import os
import time
import traceback
import logging
import math

import gymnasium as gym
import rlcarla
import numpy as np
import torch
import pygame
import carla

from torch.utils.tensorboard  import SummaryWriter
from agents.ql_diffusion       import Diffusion_QL
from rlcarla.core.obs_builder  import OBS_DIM
from rlcarla.utils.trajectory  import draw_trajectory, get_future_waypoints

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("train")

torch.set_float32_matmul_precision("medium")


# ==========================================================
# TRAIN CONFIG
# ==========================================================
class TrainConfig:

    WIDTH          = 800
    HEIGHT         = 450
    FPS            = 20

    MAX_EPISODES          = 500
    MAX_STEPS             = 1000

    START_RANDOM_STEPS    = 10000
    POLICY_RANDOM_MIX     = 0.15

    BATCH_SIZE            = 256
    TRAIN_FREQ            = 2
    SAVE_EVERY            = 10

    BUFFER_SIZE           = 200000
    ACTION_ALPHA          = 0.7

    DISCOUNT              = 0.99
    TAU                   = 0.005
    ETA                   = 0.01
    BETA_SCHEDULE         = "vp"
    N_TIMESTEPS           = 3
    LR                    = 3e-4
    ACTION_TEMPERATURE    = 0.1

    CURRICULUM = [
        (0,   "empty"),
        (100, "light"),
        (200, "medium"),
        (350, "heavy"),
    ]

    TRAFFIC_LIGHT_CURRICULUM = [
        (0,   "off"),
        (200, "on"),
    ]

    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR        = "runs/rlcarla"
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


cfg = TrainConfig()


# ==========================================================
# REPLAY BUFFER
# ==========================================================
class ReplayBuffer:

    def __init__(self, state_dim, action_dim, max_size):
        self.max_size   = max_size
        self.ptr        = 0
        self.size       = 0
        self.state      = np.zeros(
            (max_size, state_dim),  dtype=np.float32
        )
        self.action     = np.zeros(
            (max_size, action_dim), dtype=np.float32
        )
        self.next_state = np.zeros(
            (max_size, state_dim),  dtype=np.float32
        )
        self.reward     = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done   = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, s, a, ns, r, done):
        self.state[self.ptr]      = s
        self.action[self.ptr]     = a
        self.next_state[self.ptr] = ns
        self.reward[self.ptr]     = r
        self.not_done[self.ptr]   = float(not done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[idx]).to(device),
            torch.FloatTensor(self.action[idx]).to(device),
            torch.FloatTensor(self.next_state[idx]).to(device),
            torch.FloatTensor(self.reward[idx]).to(device),
            torch.FloatTensor(self.not_done[idx]).to(device),
        )

    def __len__(self):
        return self.size


# ==========================================================
# EXPLORATION
# ==========================================================
def random_action():
    throttle = np.random.uniform(0.5, 0.85)
    steer    = np.random.uniform(-0.25, 0.25)
    brake    = 0.0
    return np.array([throttle, steer, brake], dtype=np.float32)


def smooth_action(raw_action, prev_action, alpha=cfg.ACTION_ALPHA):
    return (
        alpha * raw_action + (1.0 - alpha) * prev_action
    ).astype(np.float32)


def choose_action(agent, state, prev_action, total_steps):
    if total_steps < cfg.START_RANDOM_STEPS:
        return random_action()
    if np.random.rand() < cfg.POLICY_RANDOM_MIX:
        return random_action()
    try:
        raw = np.asarray(
            agent.sample_action(state), dtype=np.float32
        )
    except Exception:
        return random_action()
    return smooth_action(raw, prev_action)


# ==========================================================
# CURRICULUM
# ==========================================================
def get_traffic_preset(episode):
    preset = "empty"
    for ep_thresh, p in cfg.CURRICULUM:
        if episode >= ep_thresh:
            preset = p
    return preset


def get_traffic_lights(episode):
    mode = "off"
    for ep_thresh, m in cfg.TRAFFIC_LIGHT_CURRICULUM:
        if episode >= ep_thresh:
            mode = m
    return mode


# ==========================================================
# 3D LIDAR POINT CLOUD VISUALIZER
# ==========================================================
def draw_point_cloud(
    screen,
    points,
    center_x,
    center_y,
    radius    = 90,
    max_range = 30.0,
):
    """
    Top-down 2D projection of 3D LiDAR points.
    Colored by height (z):
      Dark grey  → ground       (z < -1.5m)
      Grey       → near ground  (-1.5 to -0.5m)
      Green      → vehicle lvl  (-0.5 to  0.5m)
      Orange     → obstacle      (0.5 to  1.5m)
      Blue       → tall object  (> 1.5m)
    """
    font_s = pygame.font.SysFont("monospace", 10)

    # Background
    bg = pygame.Surface(
        (radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA
    )
    pygame.draw.circle(
        bg, (0, 0, 0, 175),
        (radius + 2, radius + 2), radius + 2
    )
    screen.blit(bg, (center_x - radius - 2, center_y - radius - 2))

    # Outer ring
    pygame.draw.circle(
        screen, (0, 130, 0), (center_x, center_y), radius, 1
    )

    # Range rings
    for r in [radius // 3, radius * 2 // 3]:
        pygame.draw.circle(
            screen, (0, 55, 0), (center_x, center_y), r, 1
        )

    # Crosshairs
    pygame.draw.line(
        screen, (0, 40, 0),
        (center_x - radius, center_y),
        (center_x + radius, center_y), 1
    )
    pygame.draw.line(
        screen, (0, 40, 0),
        (center_x, center_y - radius),
        (center_x, center_y + radius), 1
    )

    # Forward arrow
    pygame.draw.polygon(
        screen, (0, 255, 100),
        [
            (center_x,     center_y - radius - 7),
            (center_x - 4, center_y - radius + 1),
            (center_x + 4, center_y - radius + 1),
        ]
    )

    # Range labels
    lbl_30 = font_s.render("30m", True, (0, 90, 0))
    lbl_10 = font_s.render("10m", True, (0, 70, 0))
    screen.blit(lbl_30, (center_x + 4, center_y - radius + 2))
    screen.blit(lbl_10, (center_x + 4, center_y - radius // 3 + 2))

    # Draw points
    if len(points) > 0:
        scale = radius / max_range
        x_pts = points[:, 0]
        y_pts = points[:, 1]
        z_pts = points[:, 2]

        dist  = np.sqrt(x_pts**2 + y_pts**2)
        mask  = (dist > 0.5) & (dist < max_range)
        x_pts = x_pts[mask]
        y_pts = y_pts[mask]
        z_pts = z_pts[mask]

        for i in range(len(x_pts)):
            px = x_pts[i]
            py = y_pts[i]
            pz = z_pts[i]

            rx = int(center_x - py * scale)
            ry = int(center_y - px * scale)

            dx = rx - center_x
            dy = ry - center_y
            if math.sqrt(dx**2 + dy**2) > radius:
                continue

            if pz < -1.5:
                color = (50,  50,  50)
            elif pz < -0.5:
                color = (100, 100, 100)
            elif pz < 0.5:
                color = (0,   180,  80)
            elif pz < 1.5:
                color = (255, 140,   0)
            else:
                color = (80,  80,  255)

            pygame.draw.circle(screen, color, (rx, ry), 2)

    # Ego dot
    pygame.draw.circle(
        screen, (0, 200, 255), (center_x, center_y), 5
    )
    pygame.draw.circle(
        screen, (255, 255, 255), (center_x, center_y), 2
    )

    # Title
    title = font_s.render("LiDAR 3D", True, (0, 200, 0))
    screen.blit(
        title,
        (center_x - title.get_width() // 2,
         center_y + radius + 5)
    )

    # Height legend
    legend = [
        ((50,  50,  50),  "gnd"),
        ((100, 100, 100), "low"),
        ((0,   180,  80), "mid"),
        ((255, 140,   0), "obj"),
        ((80,  80,  255), "top"),
    ]
    lx = center_x - radius
    ly = center_y + radius - 65
    for color, label in legend:
        pygame.draw.circle(screen, color, (lx + 5, ly + 4), 4)
        lbl = font_s.render(label, True, color)
        screen.blit(lbl, (lx + 12, ly))
        ly += 14


# ==========================================================
# HUD
# ==========================================================
def draw_hud(
    screen, font, info,
    total_steps, episode, ep_reward,
    replay_size, traffic_preset,
    tl_mode, view,
):
    lines = [
        f"Episode     : {episode}",
        f"Step        : {info.get('step', 0)}",
        f"Total Steps : {total_steps}",
        f"Ep Reward   : {ep_reward:.1f}",
        f"Speed       : {info.get('speed', 0):.1f} m/s",
        f"Replay      : {replay_size}",
        f"Traffic     : {traffic_preset}",
        f"Lights      : {tl_mode}",
        f"View        : {view}",
        f"Map         : {str(info.get('map','?')).split('/')[-1]}",
    ]

    surf = pygame.Surface(
        (230, len(lines) * 22 + 10), pygame.SRCALPHA
    )
    surf.fill((0, 0, 0, 140))
    screen.blit(surf, (5, 5))

    for i, line in enumerate(lines):
        txt = font.render(line, True, (255, 255, 255))
        screen.blit(txt, (10, 10 + i * 22))


# ==========================================================
# PYGAME INIT
# ==========================================================
pygame.init()
screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
pygame.display.set_caption("RLCarla Training v3")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 16)

# ==========================================================
# ENV + AGENT + BUFFER + LOGGER
# ==========================================================
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR,        exist_ok=True)

writer = SummaryWriter(log_dir=cfg.LOG_DIR)

env = gym.make("RLCarla-v0")

agent = Diffusion_QL(
    state_dim          = OBS_DIM,
    action_dim         = 3,
    max_action         = 1.0,
    device             = cfg.DEVICE,
    discount           = cfg.DISCOUNT,
    tau                = cfg.TAU,
    eta                = cfg.ETA,
    beta_schedule      = cfg.BETA_SCHEDULE,
    n_timesteps        = cfg.N_TIMESTEPS,
    lr                 = cfg.LR,
    action_temperature = cfg.ACTION_TEMPERATURE,
)

replay       = ReplayBuffer(OBS_DIM, 3, cfg.BUFFER_SIZE)
best_reward  = -1e9
total_steps  = 0
current_view = "third_person"

logger.info(f"Device      : {cfg.DEVICE}")
logger.info(f"Obs dim     : {OBS_DIM}")
logger.info(f"Buffer size : {cfg.BUFFER_SIZE}")
logger.info(f"Max eps     : {cfg.MAX_EPISODES}")
logger.info(f"Max steps   : {cfg.MAX_STEPS}")
logger.info(f"LiDAR       : 32ch 3D (30m range)")

# ==========================================================
# TRAINING LOOP
# ==========================================================
running = True

try:
    for episode in range(cfg.MAX_EPISODES):

        if not running:
            break

        traffic_preset = get_traffic_preset(episode)
        tl_mode        = get_traffic_lights(episode)

        env.unwrapped._traffic_preset    = traffic_preset
        env.unwrapped.cfg.TRAFFIC_LIGHTS = tl_mode

        state, _ = env.reset()

        prev_action = np.zeros(3, dtype=np.float32)
        ep_reward   = 0.0
        ep_bc_loss  = []
        ep_ql_loss  = []
        ep_cr_loss  = []
        ep_info     = {}

        for step in range(cfg.MAX_STEPS):

            # ---- Pygame events ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False

            # Camera switching
            if keys[pygame.K_1]:
                current_view = "third_person"
                env.unwrapped.set_camera_view(current_view)
            elif keys[pygame.K_2]:
                current_view = "driver"
                env.unwrapped.set_camera_view(current_view)
            elif keys[pygame.K_3]:
                current_view = "front"
                env.unwrapped.set_camera_view(current_view)
            elif keys[pygame.K_4]:
                current_view = "bird_eye"
                env.unwrapped.set_camera_view(current_view)

            # Spectator mode
            if keys[pygame.K_5]:
                env.unwrapped.set_spectator_mode("follow")
            elif keys[pygame.K_6]:
                env.unwrapped.set_spectator_mode("top")
            elif keys[pygame.K_7]:
                env.unwrapped.set_spectator_mode("none")

            if not running:
                break

            # ---- Action ----
            action = choose_action(
                agent, state, prev_action, total_steps
            )

            next_state, reward, done, trunc, info = env.step(action)

            replay.add(state, action, next_state, reward, done)

            state       = next_state
            prev_action = action
            ep_reward  += reward
            total_steps += 1
            ep_info     = info

            # ---- Train ----
            if (len(replay) > cfg.BATCH_SIZE and
                    total_steps % cfg.TRAIN_FREQ == 0):

                metric = agent.train(
                    replay,
                    iterations = 1,
                    batch_size = cfg.BATCH_SIZE,
                )

                ep_bc_loss.append(np.mean(metric["bc_loss"]))
                ep_ql_loss.append(np.mean(metric["ql_loss"]))
                ep_cr_loss.append(np.mean(metric["critic_loss"]))

            # ---- Render camera ----
            frame = env.unwrapped.get_camera_frame()

            if frame is not None:
                cam_tf = env.unwrapped.get_camera_transform()
                K      = env.unwrapped.get_camera_intrinsic()

                if cam_tf is not None and K is not None:
                    waypoints = get_future_waypoints(
                        env.unwrapped.vehicle,
                        env.unwrapped.carla_map,
                        n=20, spacing=3.0,
                    )
                    frame = draw_trajectory(
                        frame, waypoints, K, cam_tf,
                        cfg.WIDTH, cfg.HEIGHT,
                    )

                surf = pygame.surfarray.make_surface(
                    frame.swapaxes(0, 1)
                )
                screen.blit(surf, (0, 0))

            # ---- HUD ----
            draw_hud(
                screen, font, ep_info,
                total_steps, episode, ep_reward,
                len(replay), traffic_preset,
                tl_mode, current_view,
            )

            # ---- 3D LiDAR Point Cloud ----
            if env.unwrapped._lidar is not None:
                points = env.unwrapped._lidar.get_points()
                draw_point_cloud(
                    screen,
                    points,
                    center_x  = cfg.WIDTH  - 115,
                    center_y  = cfg.HEIGHT - 115,
                    radius    = 90,
                    max_range = 30.0,
                )

            pygame.display.flip()
            clock.tick(cfg.FPS)

            if done or trunc:
                break

        # ---- Episode logging ----
        writer.add_scalar(
            "reward/episode",     ep_reward, episode
        )
        writer.add_scalar(
            "reward/total_steps", ep_reward, total_steps
        )
        writer.add_scalar(
            "env/episode_length", step + 1,  episode
        )

        if ep_bc_loss:
            writer.add_scalar(
                "loss/bc",     np.mean(ep_bc_loss), episode
            )
            writer.add_scalar(
                "loss/ql",     np.mean(ep_ql_loss), episode
            )
            writer.add_scalar(
                "loss/critic", np.mean(ep_cr_loss), episode
            )

        for key in [
            "forward_progress", "lane_center", "yaw_align",
            "curve_reward", "collision", "off_road", "stuck",
            "red_light", "wrong_lane", "wrong_way", "comfort",
        ]:
            if key in ep_info:
                writer.add_scalar(
                    f"reward/{key}", ep_info[key], episode
                )

        logger.info(
            f"Ep {episode:04d} | "
            f"Reward {ep_reward:8.2f} | "
            f"Steps {step+1:4d} | "
            f"Traffic {traffic_preset:6s} | "
            f"Lights {tl_mode:3s} | "
            f"Buffer {len(replay):6d} | "
            f"Done: {ep_info.get('term_reason', 'trunc')}"
        )

        if episode % cfg.SAVE_EVERY == 0:
            agent.save_model(cfg.CHECKPOINT_DIR, id=episode)
            logger.info(f"Checkpoint saved — episode {episode}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save_model(cfg.CHECKPOINT_DIR, id=9999)
            logger.info(f"New best — reward {best_reward:.2f}")

except KeyboardInterrupt:
    logger.info("Stopped by user.")

except Exception:
    traceback.print_exc()

finally:
    writer.close()
    env.close()
    pygame.quit()
    logger.info("Training finished.")
