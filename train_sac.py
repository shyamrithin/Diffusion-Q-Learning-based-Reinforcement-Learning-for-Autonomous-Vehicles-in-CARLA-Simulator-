# ==========================================================
# train_sac.py
# SAC Baseline Training Script v2
#
# Changes from v1:
#   - MAX_EPISODES = 1300 (~8 hours on RTX 4050)
#   - Traffic lights ON at episode 400 (was 1000)
#   - total_steps accounts for resume offset
#   - Early throttle bias < 20k steps
#     prevents aggressive braking exploration
#   - Curriculum adjusted for 1300 episode budget
#
# Everything else identical to Diffusion-QL setup:
#   Same env, reward, obs, map (Town03), buffer size.
#
# CARLA 0.9.16 | Gymnasium 1.3 | Python 3.10
# ==========================================================

import os
import time
import traceback
import logging
import math
import re

import gymnasium as gym
import rlcarla
import numpy as np
import torch
import pygame

from torch.utils.tensorboard import SummaryWriter
from agents.sac               import SAC
from rlcarla.core.obs_builder import OBS_DIM
from rlcarla.utils.trajectory import (
    draw_trajectory, get_future_waypoints
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("train_sac")

torch.set_float32_matmul_precision("medium")


# ==========================================================
# CONFIG
# ==========================================================
class SACConfig:
    """
    SAC training configuration v2.
    Optimised for 1300 episode budget (~8 hours RTX 4050).

    Curriculum scaled to 1300 episodes:
      ep 0-200   : empty roads
      ep 200-500 : light traffic
      ep 500-800 : medium traffic
      ep 800+    : heavy traffic
      ep 400+    : traffic lights ON
    """

    WIDTH          = 800
    HEIGHT         = 450
    FPS            = 20

    MAX_EPISODES   = 1300
    MAX_STEPS      = 1000

    START_RANDOM_STEPS = 5000
    POLICY_RANDOM_MIX  = 0.05

    BATCH_SIZE     = 256
    TRAIN_FREQ     = 2
    SAVE_EVERY     = 10

    BUFFER_SIZE    = 200000

    DISCOUNT       = 0.99
    TAU            = 0.005
    LR             = 3e-4
    GRAD_NORM      = 0.5
    HIDDEN_DIM     = 256
    AUTO_ENTROPY   = True

    CURRICULUM = [
        (0,   "empty"),
        (200, "light"),
        (500, "medium"),
        (800, "heavy"),
    ]

    TRAFFIC_LIGHT_CURRICULUM = [
        (0,   "off"),
        (400, "on"),
    ]

    CHECKPOINT_DIR = "checkpoints_sac"
    LOG_DIR        = "runs/rlcarla_sac"
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


cfg = SACConfig()


# ==========================================================
# REPLAY BUFFER
# ==========================================================
class ReplayBuffer:
    """
    Experience replay buffer with reward clipping.
    Identical to Diffusion-QL buffer for fair comparison.
    Reward clipped to [-10, 10] prevents Q explosion.
    """

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
        self.reward     = np.zeros(
            (max_size, 1), dtype=np.float32
        )
        self.not_done   = np.zeros(
            (max_size, 1), dtype=np.float32
        )

    def add(self, s, a, ns, r, done):
        r = float(np.clip(r, -10.0, 10.0))
        self.state[self.ptr]      = s
        self.action[self.ptr]     = a
        self.next_state[self.ptr] = ns
        self.reward[self.ptr]     = r
        self.not_done[self.ptr]   = float(not done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(
            0, self.size, size=batch_size
        )
        return (
            torch.FloatTensor(
                self.state[idx]).to(device),
            torch.FloatTensor(
                self.action[idx]).to(device),
            torch.FloatTensor(
                self.next_state[idx]).to(device),
            torch.FloatTensor(
                self.reward[idx]).to(device),
            torch.FloatTensor(
                self.not_done[idx]).to(device),
        )

    def __len__(self):
        return self.size


# ==========================================================
# HELPERS
# ==========================================================
def random_action():
    """
    Random exploration action.
    Never brakes — keeps vehicle moving during warmup.
    """
    return np.array([
        np.random.uniform(0.5, 0.85),
        np.random.uniform(-0.25, 0.25),
        0.0,
    ], dtype=np.float32)


def get_traffic_preset(episode):
    """Return traffic preset for current episode."""
    preset = "empty"
    for ep_thresh, p in cfg.CURRICULUM:
        if episode >= ep_thresh:
            preset = p
    return preset


def get_traffic_lights(episode):
    """Return traffic light mode for current episode."""
    mode = "off"
    for ep_thresh, m in cfg.TRAFFIC_LIGHT_CURRICULUM:
        if episode >= ep_thresh:
            mode = m
    return mode


def find_latest_checkpoint(checkpoint_dir):
    """Find latest SAC checkpoint (excluding 9999)."""
    if not os.path.exists(checkpoint_dir):
        return 0, None
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("sac_actor_") and
        f.endswith(".pth")
    ]
    episodes = []
    for f in files:
        match = re.search(r"sac_actor_(\d+)\.pth", f)
        if match:
            ep = int(match.group(1))
            if ep != 9999:
                episodes.append(ep)
    if not episodes:
        return 0, None
    return max(episodes), checkpoint_dir


# ==========================================================
# LIDAR VISUALIZER
# ==========================================================
def draw_point_cloud(screen, points, center_x, center_y,
                     size=180, max_range=30.0):
    """Square LiDAR point cloud — blue SAC theme."""
    font_s = pygame.font.SysFont("monospace", 10)

    bg = pygame.Surface((size, size), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 175))
    screen.blit(bg, (center_x-size//2, center_y-size//2))

    pygame.draw.rect(
        screen, (0, 100, 200),
        (center_x-size//2, center_y-size//2, size, size), 1
    )
    pygame.draw.line(
        screen, (0, 30, 60),
        (center_x-size//2, center_y),
        (center_x+size//2, center_y), 1
    )
    pygame.draw.line(
        screen, (0, 30, 60),
        (center_x, center_y-size//2),
        (center_x, center_y+size//2), 1
    )
    for offset in [-size//4, size//4]:
        pygame.draw.line(
            screen, (0, 20, 40),
            (center_x-size//2, center_y+offset),
            (center_x+size//2, center_y+offset), 1
        )
        pygame.draw.line(
            screen, (0, 20, 40),
            (center_x+offset, center_y-size//2),
            (center_x+offset, center_y+size//2), 1
        )

    pygame.draw.polygon(screen, (0, 150, 255), [
        (center_x,   center_y-size//2-7),
        (center_x-4, center_y-size//2+1),
        (center_x+4, center_y-size//2+1),
    ])

    lbl_30 = font_s.render("30m", True, (0, 80, 160))
    lbl_15 = font_s.render("15m", True, (0, 60, 120))
    screen.blit(lbl_30, (center_x+3, center_y-size//2+2))
    screen.blit(lbl_15, (center_x+3, center_y-size//4+2))

    if len(points) > 0:
        scale = (size//2) / max_range
        x_pts = points[:,0]
        y_pts = points[:,1]
        z_pts = points[:,2]
        dist  = np.sqrt(x_pts**2 + y_pts**2)
        mask  = (dist > 0.5) & (dist < max_range)
        x_pts = x_pts[mask]
        y_pts = y_pts[mask]
        z_pts = z_pts[mask]

        for i in range(len(x_pts)):
            rx = int(center_x - y_pts[i] * scale)
            ry = int(center_y - x_pts[i] * scale)
            if (rx < center_x-size//2 or
                    rx > center_x+size//2 or
                    ry < center_y-size//2 or
                    ry > center_y+size//2):
                continue
            pz = z_pts[i]
            if pz < -1.5:   color = (50,  50,  50)
            elif pz < -0.5: color = (100, 100, 100)
            elif pz < 0.5:  color = (0,   180,  80)
            elif pz < 1.5:  color = (255, 140,   0)
            else:           color = (80,  80,  255)
            pygame.draw.circle(
                screen, color, (rx, ry), 2
            )

    pygame.draw.circle(
        screen, (0, 150, 255), (center_x, center_y), 5
    )
    pygame.draw.circle(
        screen, (255, 255, 255), (center_x, center_y), 2
    )

    title = font_s.render(
        "LiDAR [SAC]", True, (0, 150, 255)
    )
    screen.blit(title, (
        center_x - title.get_width()//2,
        center_y + size//2 + 4
    ))

    legend = [
        ((50,  50,  50),  "gnd"),
        ((100, 100, 100), "low"),
        ((0,   180,  80), "mid"),
        ((255, 140,   0), "obj"),
        ((80,  80,  255), "top"),
    ]
    lx = center_x - size//2 + 4
    ly = center_y + size//2 - 75
    for color, label in legend:
        pygame.draw.circle(
            screen, color, (lx+4, ly+4), 3
        )
        lbl = font_s.render(label, True, color)
        screen.blit(lbl, (lx+10, ly))
        ly += 14


# ==========================================================
# HUD
# ==========================================================
def draw_hud(screen, font, info, total_steps, episode,
             ep_reward, replay_size, traffic_preset,
             tl_mode, view, start_episode):
    """Blue-themed HUD for SAC."""
    lines = [
        f"[SAC] Ep    : {episode} (+{episode-start_episode})",
        f"Step        : {info.get('step', 0)}",
        f"Total Steps : {total_steps}",
        f"Ep Reward   : {ep_reward:.1f}",
        f"Speed       : {info.get('speed', 0):.1f} m/s",
        f"Replay      : {replay_size}",
        f"Traffic     : {traffic_preset}",
        f"Lights      : {tl_mode}",
        f"View        : {view}",
        f"Map         : "
        f"{str(info.get('map','?')).split('/')[-1]}",
    ]

    surf = pygame.Surface(
        (260, len(lines)*22+10), pygame.SRCALPHA
    )
    surf.fill((0, 0, 30, 140))
    screen.blit(surf, (5, 5))

    for i, line in enumerate(lines):
        txt = font.render(line, True, (200, 220, 255))
        screen.blit(txt, (10, 10+i*22))


# ==========================================================
# PYGAME INIT
# ==========================================================
pygame.init()
screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
pygame.display.set_caption("RLCarla — SAC Baseline")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 16)

# ==========================================================
# ENV + AGENT + BUFFER
# ==========================================================
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR,        exist_ok=True)

writer = SummaryWriter(log_dir=cfg.LOG_DIR)

env = gym.make("RLCarla-v0")

agent = SAC(
    state_dim    = OBS_DIM,
    action_dim   = 3,
    max_action   = 1.0,
    device       = cfg.DEVICE,
    discount     = cfg.DISCOUNT,
    tau          = cfg.TAU,
    lr           = cfg.LR,
    grad_norm    = cfg.GRAD_NORM,
    hidden_dim   = cfg.HIDDEN_DIM,
    auto_entropy = cfg.AUTO_ENTROPY,
)

# ==========================================================
# RESUME
# ==========================================================
start_episode = 0
latest_ep, path = find_latest_checkpoint(
    cfg.CHECKPOINT_DIR
)
if latest_ep > 0:
    agent.load_model(path, id=latest_ep)
    start_episode = latest_ep
    logger.info(f"Resumed SAC from episode {latest_ep}")
else:
    logger.info("SAC starting fresh")

replay      = ReplayBuffer(OBS_DIM, 3, cfg.BUFFER_SIZE)
best_reward = -1e9
# Account for steps already done on resume
total_steps = start_episode * 200
current_view= "third_person"

logger.info(f"Algorithm     : SAC (baseline)")
logger.info(f"Device        : {cfg.DEVICE}")
logger.info(f"Obs dim       : {OBS_DIM}")
logger.info(f"Start episode : {start_episode}")
logger.info(f"Target end    : {cfg.MAX_EPISODES}")
logger.info(f"LR            : {cfg.LR}")
logger.info(f"Auto entropy  : {cfg.AUTO_ENTROPY}")
logger.info(f"Train freq    : {cfg.TRAIN_FREQ}")
logger.info(f"Reward clip   : [-10, 10]")
logger.info(f"Throttle bias : < 20k steps")

# ==========================================================
# TRAINING LOOP
# ==========================================================
running = True

try:
    for episode in range(start_episode, cfg.MAX_EPISODES):

        if not running:
            break

        traffic_preset = get_traffic_preset(episode)
        tl_mode        = get_traffic_lights(episode)

        env.unwrapped._traffic_preset    = traffic_preset
        env.unwrapped.cfg.TRAFFIC_LIGHTS = tl_mode

        state, _ = env.reset()

        ep_reward  = 0.0
        ep_cr_loss = []
        ep_ac_loss = []
        ep_info    = {}

        for step in range(cfg.MAX_STEPS):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False

            if keys[pygame.K_1]:
                current_view = "third_person"
                env.unwrapped.set_camera_view(
                    current_view
                )
            elif keys[pygame.K_2]:
                current_view = "driver"
                env.unwrapped.set_camera_view(
                    current_view
                )
            elif keys[pygame.K_3]:
                current_view = "front"
                env.unwrapped.set_camera_view(
                    current_view
                )
            elif keys[pygame.K_4]:
                current_view = "bird_eye"
                env.unwrapped.set_camera_view(
                    current_view
                )

            if keys[pygame.K_5]:
                env.unwrapped.set_spectator_mode("follow")
            elif keys[pygame.K_6]:
                env.unwrapped.set_spectator_mode("top")
            elif keys[pygame.K_7]:
                env.unwrapped.set_spectator_mode("none")

            if not running:
                break

            # Action selection
            if total_steps < cfg.START_RANDOM_STEPS:
                action = random_action()
            elif np.random.rand() < cfg.POLICY_RANDOM_MIX:
                action = random_action()
            else:
                action = agent.sample_action(state)

            # 🔥 Early training throttle bias
            # SAC entropy explores brake aggressively early
            # Force minimum throttle + cap brake < 20k steps
            if total_steps < 20000:
                action[0] = max(action[0], 0.4)
                action[2] = min(action[2], 0.1)

            next_state, reward, done, trunc, info = \
                env.step(action)

            replay.add(
                state, action, next_state, reward, done
            )

            state       = next_state
            ep_reward  += reward
            total_steps += 1
            ep_info     = info

            if (len(replay) > cfg.BATCH_SIZE and
                    total_steps % cfg.TRAIN_FREQ == 0):

                metric = agent.train(
                    replay,
                    iterations = 1,
                    batch_size = cfg.BATCH_SIZE,
                )
                ep_cr_loss.append(
                    np.mean(metric["critic_loss"])
                )
                ep_ac_loss.append(
                    np.mean(metric["actor_loss"])
                )

            # Render
            frame = env.unwrapped.get_camera_frame()
            if frame is not None:
                cam_tf = (
                    env.unwrapped.get_camera_transform()
                )
                K = (
                    env.unwrapped.get_camera_intrinsic()
                )
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

            draw_hud(
                screen, font, ep_info,
                total_steps, episode, ep_reward,
                len(replay), traffic_preset,
                tl_mode, current_view, start_episode,
            )

            if env.unwrapped._lidar is not None:
                points = (
                    env.unwrapped._lidar.get_points()
                )
                draw_point_cloud(
                    screen, points,
                    center_x  = cfg.WIDTH  - 100,
                    center_y  = cfg.HEIGHT - 100,
                    size      = 180,
                    max_range = 30.0,
                )

            pygame.display.flip()
            clock.tick(cfg.FPS)

            if done or trunc:
                break

        # Episode logging
        writer.add_scalar(
            "reward/episode",     ep_reward, episode)
        writer.add_scalar(
            "reward/total_steps", ep_reward, total_steps)
        writer.add_scalar(
            "env/episode_length", step+1,    episode)

        if ep_cr_loss:
            writer.add_scalar(
                "loss/critic",
                np.mean(ep_cr_loss), episode)
            writer.add_scalar(
                "loss/actor",
                np.mean(ep_ac_loss), episode)

        for key in [
            "forward_progress", "lane_center",
            "yaw_align", "curve_reward", "collision",
            "off_road", "stuck", "red_light",
            "wrong_lane", "wrong_way", "comfort",
            "proximity_reward",
        ]:
            if key in ep_info:
                writer.add_scalar(
                    f"reward/{key}",
                    ep_info[key], episode
                )

        logger.info(
            f"Ep {episode:04d} "
            f"(+{episode-start_episode:04d}) | "
            f"Reward {ep_reward:8.2f} | "
            f"Steps {step+1:4d} | "
            f"Traffic {traffic_preset:6s} | "
            f"Lights {tl_mode:3s} | "
            f"Buffer {len(replay):6d} | "
            f"Done: {ep_info.get('term_reason','trunc')}"
        )

        if episode % cfg.SAVE_EVERY == 0:
            agent.save_model(
                cfg.CHECKPOINT_DIR, id=episode
            )
            logger.info(
                f"[SAC] Checkpoint — episode {episode}"
            )

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save_model(
                cfg.CHECKPOINT_DIR, id=9999
            )
            logger.info(
                f"[SAC] New best — "
                f"reward {best_reward:.2f}"
            )

except KeyboardInterrupt:
    logger.info("Stopped by user.")

except Exception:
    traceback.print_exc()

finally:
    writer.close()
    env.close()
    pygame.quit()
    logger.info("SAC training finished.")