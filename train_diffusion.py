# ==========================================================
# train_diffusion.py
# RLCarla Training Script v14 — Entropy-Regularised DQL
#                               + Reward Scaling
#                               + Replay-Buffer Persistence
#
# Changes in this revision:
#   - Replay buffer is now saved to disk alongside model
#     checkpoints (every SAVE_EVERY episodes) and reloaded
#     on resume. A power cut / interruption no longer wipes
#     the ~200k-transition buffer, so resumes recover the
#     critic's experience instead of restarting from empty.
#   - MAX_EPISODES extended to 2000 (V14 was still climbing
#     at the previous 1500 horizon).
#   - Fixed the startup log line that mis-printed critic LR
#     as LR*10; the agent actually uses LR*2.
#
# Retained V14 recipe (the configuration that broke through):
#   reward_scale = 0.1 (in agent TD targets)
#   reward clip [-15,15] in reward.py, [-10,10] in buffer
#   critic LR = LR*2, stuck penalty = 5.0
#   Critic pre-training (40k), ETA schedule (0 -> 0.0005),
#   periodic critic target reset (7k), entropy auto-alpha.
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
import carla

from torch.utils.tensorboard  import SummaryWriter
from agents.ql_diffusion       import Diffusion_QL
from rlcarla.core.obs_builder  import OBS_DIM
from rlcarla.utils.trajectory  import (
    draw_trajectory, get_future_waypoints
)

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
    """
    Diffusion-QL v14 — Entropy-Regularised + Reward Scaling.

    All matched to SAC/PPO for fair comparison:
      Same curriculum, same traffic light schedule.
    """

    WIDTH          = 800
    HEIGHT         = 450
    FPS            = 20

    MAX_EPISODES          = 2000
    MAX_STEPS             = 1000

    START_RANDOM_STEPS    = 10000
    POLICY_RANDOM_MIX     = 0.15

    BATCH_SIZE            = 256
    TRAIN_FREQ            = 6
    SAVE_EVERY            = 10

    BUFFER_SIZE           = 200000
    ACTION_ALPHA          = 0.7

    DISCOUNT              = 0.99
    TAU                   = 0.005
    ETA                   = 0.001
    BETA_SCHEDULE         = "vp"
    N_TIMESTEPS           = 3
    LR                    = 3e-5
    GRAD_NORM             = 0.05
    ACTION_TEMPERATURE    = 0.05

    # Entropy regularisation
    ALPHA      = 0.2    # initial entropy coefficient
    AUTO_ALPHA = True   # auto-tune like SAC

    # Critic pre-training
    PRE_TRAIN_CRITIC_STEPS = 40000

    # ETA scheduling
    ETA_START  = 0.0
    ETA_END    = 0.0005
    ETA_WARMUP = 70000

    # Periodic critic target reset
    CRITIC_RESET_FREQ = 7000

    # Curriculum — matched to SAC/PPO
    CURRICULUM = [
        (0,   "empty"),
        (200, "light"),
        (500, "medium"),
        (800, "heavy"),
    ]

    TRAFFIC_LIGHT_CURRICULUM = [
        (0,   "off"),
        (700, "on"),
    ]

    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR        = "runs/rlcarla_v14"
    # Replay-buffer persistence file
    BUFFER_PATH    = "checkpoints/replay_buffer.npz"
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


cfg = TrainConfig()


# ==========================================================
# RESUME HELPER
# ==========================================================
def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return 0, None
    actor_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("actor_") and f.endswith(".pth")
    ]
    episodes = []
    for f in actor_files:
        match = re.search(r"actor_(\d+)\.pth", f)
        if match:
            ep = int(match.group(1))
            if ep != 9999:
                episodes.append(ep)
    if not episodes:
        return 0, None
    return max(episodes), checkpoint_dir


def resume_training(agent, checkpoint_dir):
    latest_ep, path = find_latest_checkpoint(
        checkpoint_dir
    )
    if latest_ep > 0:
        agent.load_model(path, id=latest_ep)
        logger.info(
            f"Resumed from episode {latest_ep}"
        )
        return latest_ep
    best_path = os.path.join(
        checkpoint_dir, "actor_9999.pth"
    )
    if os.path.exists(best_path):
        agent.load_model(checkpoint_dir, id=9999)
        logger.info("Resumed from best model (9999)")
        return 0
    logger.info("No checkpoint — starting fresh")
    return 0


# ==========================================================
# REPLAY BUFFER
# ==========================================================
class ReplayBuffer:
    """Replay buffer with reward clipping [-10, 10].

    Supports disk persistence (save/load) so an
    interrupted run can resume with experience intact
    instead of restarting from an empty buffer.
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

    def save(self, path):
        """Persist buffer to disk (only the filled portion).

        Writes to a temp file then renames, so a crash
        mid-write can't corrupt an existing good buffer.
        """
        try:
            tmp = path + ".tmp"
            np.savez(
                tmp,
                state      = self.state[:self.size],
                action     = self.action[:self.size],
                next_state = self.next_state[:self.size],
                reward     = self.reward[:self.size],
                not_done   = self.not_done[:self.size],
                ptr        = np.array(self.ptr),
                size       = np.array(self.size),
            )
            # np.savez appends .npz to the temp name
            os.replace(tmp + ".npz", path)
        except Exception as e:
            logger.warning(f"Buffer save failed: {e}")

    def load(self, path):
        """Reload buffer from disk. Returns True on success."""
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            n = int(data["size"])
            n = min(n, self.max_size)
            self.state[:n]      = data["state"][:n]
            self.action[:n]     = data["action"][:n]
            self.next_state[:n] = data["next_state"][:n]
            self.reward[:n]     = data["reward"][:n]
            self.not_done[:n]   = data["not_done"][:n]
            self.size = n
            self.ptr  = int(data["ptr"]) % self.max_size
            return True
        except Exception as e:
            logger.warning(f"Buffer load failed: {e}")
            return False

    def __len__(self):
        return self.size


# ==========================================================
# EXPLORATION
# ==========================================================
def random_action():
    """Biased random — always forward, never brake."""
    throttle = np.random.uniform(0.5, 0.85)
    steer    = np.random.uniform(-0.25, 0.25)
    brake    = 0.0
    return np.array(
        [throttle, steer, brake], dtype=np.float32
    )


def smooth_action(raw_action, prev_action,
                  alpha=cfg.ACTION_ALPHA):
    return (
        alpha * raw_action + (1.0 - alpha) * prev_action
    ).astype(np.float32)


def choose_action(agent, state, prev_action,
                  total_steps):
    """
    Action selection:
      Pre-training phase : random (critic warms up)
      Policy phase       : entropy-guided diffusion
    """
    if total_steps < cfg.START_RANDOM_STEPS:
        return random_action()

    if total_steps < cfg.PRE_TRAIN_CRITIC_STEPS:
        return random_action()

    if np.random.rand() < cfg.POLICY_RANDOM_MIX:
        return random_action()

    try:
        raw = np.asarray(
            agent.sample_action(state), dtype=np.float32
        )
    except Exception:
        return random_action()

    # Throttle bias after pre-training
    if total_steps < cfg.PRE_TRAIN_CRITIC_STEPS + 50000:
        raw[0] = max(raw[0], 0.4)
        raw[2] = min(raw[2], 0.1)

    return smooth_action(raw, prev_action)


# ==========================================================
# ETA SCHEDULER
# ==========================================================
def get_eta(total_steps):
    """Gradually increase QL weight."""
    if total_steps < cfg.PRE_TRAIN_CRITIC_STEPS:
        return 0.0
    warmup_steps = total_steps - cfg.PRE_TRAIN_CRITIC_STEPS
    frac         = min(
        warmup_steps / cfg.ETA_WARMUP, 1.0
    )
    return cfg.ETA_START + frac * (
        cfg.ETA_END - cfg.ETA_START
    )


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
# LIDAR VISUALIZER
# ==========================================================
def draw_point_cloud(screen, points, center_x, center_y,
                     size=180, max_range=30.0):
    font_s = pygame.font.SysFont("monospace", 10)

    bg = pygame.Surface((size, size), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 175))
    screen.blit(bg, (center_x-size//2, center_y-size//2))

    pygame.draw.rect(
        screen, (0, 130, 0),
        (center_x-size//2, center_y-size//2, size, size), 1
    )
    pygame.draw.line(
        screen, (0, 40, 0),
        (center_x-size//2, center_y),
        (center_x+size//2, center_y), 1
    )
    pygame.draw.line(
        screen, (0, 40, 0),
        (center_x, center_y-size//2),
        (center_x, center_y+size//2), 1
    )
    for offset in [-size//4, size//4]:
        pygame.draw.line(
            screen, (0, 25, 0),
            (center_x-size//2, center_y+offset),
            (center_x+size//2, center_y+offset), 1
        )
        pygame.draw.line(
            screen, (0, 25, 0),
            (center_x+offset, center_y-size//2),
            (center_x+offset, center_y+size//2), 1
        )

    pygame.draw.polygon(screen, (0, 255, 100), [
        (center_x,   center_y-size//2-7),
        (center_x-4, center_y-size//2+1),
        (center_x+4, center_y-size//2+1),
    ])

    lbl_30 = font_s.render("30m", True, (0, 90, 0))
    lbl_15 = font_s.render("15m", True, (0, 70, 0))
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
        screen, (0, 200, 255), (center_x, center_y), 5
    )
    pygame.draw.circle(
        screen, (255, 255, 255), (center_x, center_y), 2
    )

    title = font_s.render(
        "LiDAR [DQL]", True, (0, 200, 0)
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
             tl_mode, view, start_episode,
             current_eta, pre_training,
             current_alpha, current_entropy):
    """HUD with entropy and alpha display."""
    phase = "CRITIC PRE-TRAIN" if pre_training else "FULL"

    lines = [
        f"[DQL-E] Ep  : {episode} (+{episode-start_episode})",
        f"Phase       : {phase}",
        f"ETA         : {current_eta:.5f}",
        f"Alpha       : {current_alpha:.4f}",
        f"Entropy     : {current_entropy:.4f}",
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
        (270, len(lines)*22+10), pygame.SRCALPHA
    )
    surf.fill((0, 0, 0, 140))
    screen.blit(surf, (5, 5))

    for i, line in enumerate(lines):
        color = (255, 200, 0) if (
            i == 1 and pre_training
        ) else (255, 255, 255)
        txt = font.render(line, True, color)
        screen.blit(txt, (10, 10+i*22))


# ==========================================================
# PYGAME INIT
# ==========================================================
pygame.init()
screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
pygame.display.set_caption(
    "RLCarla — Entropy-Regularised DQL v14"
)
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 16)

# ==========================================================
# ENV + AGENT
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
    grad_norm          = cfg.GRAD_NORM,
    action_temperature = cfg.ACTION_TEMPERATURE,
    # Entropy regularisation
    alpha              = cfg.ALPHA,
    auto_alpha         = cfg.AUTO_ALPHA,
)

# ==========================================================
# RESUME
# ==========================================================
start_episode = resume_training(
    agent, cfg.CHECKPOINT_DIR
)

replay          = ReplayBuffer(
    OBS_DIM, 3, cfg.BUFFER_SIZE
)

# Reload the replay buffer if resuming and a saved
# buffer exists — avoids the empty-buffer recovery dip.
if start_episode > 0:
    if replay.load(cfg.BUFFER_PATH):
        logger.info(
            f"Replay buffer restored — "
            f"{len(replay)} transitions"
        )
    else:
        logger.info(
            "No saved buffer found — "
            "buffer will refill from scratch"
        )

best_reward     = -1e9
total_steps     = start_episode * 200
current_view    = "third_person"
current_alpha   = cfg.ALPHA
current_entropy = 0.0

logger.info(f"Algorithm        : DQL v14 (Entropy-Reg)")
logger.info(f"Device           : {cfg.DEVICE}")
logger.info(f"Obs dim          : {OBS_DIM}")
logger.info(f"Start episode    : {start_episode}")
logger.info(f"Target end       : {cfg.MAX_EPISODES}")
logger.info(f"Actor LR         : {cfg.LR}")
logger.info(f"Critic LR        : {cfg.LR * 2}")
logger.info(f"Alpha (init)     : {cfg.ALPHA}")
logger.info(f"Auto alpha       : {cfg.AUTO_ALPHA}")
logger.info(f"Target entropy   : {-3.0}")
logger.info(f"Grad norm        : {cfg.GRAD_NORM}")
logger.info(f"Train freq       : {cfg.TRAIN_FREQ}")
logger.info(f"Reward clip      : buffer [-10,10]")
logger.info(
    f"Critic pre-train : {cfg.PRE_TRAIN_CRITIC_STEPS}"
)
logger.info(
    f"ETA schedule     : "
    f"{cfg.ETA_START} → {cfg.ETA_END}"
)
logger.info(
    f"Critic reset     : every {cfg.CRITIC_RESET_FREQ}"
)
logger.info(f"Lights ON        : episode 700")
logger.info(f"Heavy traffic    : episode 800")

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

        prev_action  = np.zeros(3, dtype=np.float32)
        ep_reward    = 0.0
        ep_bc_loss   = []
        ep_ql_loss   = []
        ep_cr_loss   = []
        ep_entropy   = []
        ep_alpha     = []
        ep_info      = {}

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

            action = choose_action(
                agent, state, prev_action, total_steps
            )

            next_state, reward, done, trunc, info = \
                env.step(action)

            replay.add(
                state, action, next_state, reward, done
            )

            state       = next_state
            prev_action = action
            ep_reward  += reward
            total_steps += 1
            ep_info     = info

            # Update ETA
            current_eta  = get_eta(total_steps)
            agent.eta    = current_eta
            pre_training = (
                total_steps < cfg.PRE_TRAIN_CRITIC_STEPS
            )

            if len(replay) > cfg.BATCH_SIZE:

                if total_steps % cfg.TRAIN_FREQ == 0:

                    if pre_training:
                        # Critic only — no actor
                        cr_loss = agent.train_critic_only(
                            replay,
                            batch_size=cfg.BATCH_SIZE,
                        )
                        ep_cr_loss.append(cr_loss)

                    else:
                        # Full update with entropy
                        metric = agent.train(
                            replay,
                            iterations = 1,
                            batch_size = cfg.BATCH_SIZE,
                        )
                        ep_bc_loss.append(
                            np.mean(metric["bc_loss"])
                        )
                        ep_ql_loss.append(
                            np.mean(metric["ql_loss"])
                        )
                        ep_cr_loss.append(
                            np.mean(metric["critic_loss"])
                        )
                        ep_entropy.append(
                            np.mean(metric["entropy"])
                        )
                        ep_alpha.append(
                            np.mean(metric["alpha"])
                        )

                        # Update display values
                        current_alpha   = agent.alpha
                        current_entropy = np.mean(
                            metric["entropy"]
                        )

            # Periodic critic target reset
            if (total_steps > 0 and
                    total_steps %
                    cfg.CRITIC_RESET_FREQ == 0):
                agent.reset_critic_target()
                writer.add_scalar(
                    "debug/critic_reset",
                    1, total_steps
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
                current_eta, pre_training,
                current_alpha, current_entropy,
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
        writer.add_scalar(
            "debug/eta",          current_eta, episode)
        writer.add_scalar(
            "debug/alpha",        current_alpha, episode)

        if ep_bc_loss:
            writer.add_scalar(
                "loss/bc",
                np.mean(ep_bc_loss), episode)
            writer.add_scalar(
                "loss/ql",
                np.mean(ep_ql_loss), episode)
        if ep_cr_loss:
            writer.add_scalar(
                "loss/critic",
                np.mean(ep_cr_loss), episode)
        if ep_entropy:
            writer.add_scalar(
                "debug/entropy",
                np.mean(ep_entropy), episode)

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

        phase_str = (
            "PRE-TRAIN" if pre_training else "FULL"
        )
        logger.info(
            f"Ep {episode:04d} "
            f"(+{episode-start_episode:04d}) | "
            f"Reward {ep_reward:8.2f} | "
            f"Steps {step+1:4d} | "
            f"Traffic {traffic_preset:6s} | "
            f"Lights {tl_mode:3s} | "
            f"ETA {current_eta:.5f} | "
            f"Alpha {current_alpha:.4f} | "
            f"Entropy {current_entropy:.4f} | "
            f"Phase {phase_str} | "
            f"Buffer {len(replay):6d} | "
            f"Done: {ep_info.get('term_reason','trunc')}"
        )

        if episode % cfg.SAVE_EVERY == 0:
            agent.save_model(
                cfg.CHECKPOINT_DIR, id=episode
            )
            # Persist the replay buffer alongside the model
            # so an interruption resumes with experience.
            replay.save(cfg.BUFFER_PATH)
            logger.info(
                f"Checkpoint saved — episode {episode} "
                f"(+ buffer, {len(replay)} transitions)"
            )

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save_model(
                cfg.CHECKPOINT_DIR, id=9999
            )
            logger.info(
                f"New best — reward {best_reward:.2f}"
            )

except KeyboardInterrupt:
    logger.info("Stopped by user.")

except Exception:
    traceback.print_exc()

finally:
    # Save buffer on exit so even a manual stop is recoverable
    try:
        replay.save(cfg.BUFFER_PATH)
        logger.info(
            f"Buffer saved on exit — "
            f"{len(replay)} transitions"
        )
    except Exception:
        pass
    writer.close()
    env.close()
    pygame.quit()
    logger.info("DQL v14 training finished.")