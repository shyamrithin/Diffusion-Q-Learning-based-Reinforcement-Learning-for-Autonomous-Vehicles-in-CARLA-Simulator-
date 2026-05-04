# ==========================================================
# train_ppo.py
# PPO Baseline Training Script
#
# Trains PPO agent in identical RLCarla-v0 environment.
# Key difference from SAC/Diffusion-QL:
#   ON-policy — collects ROLLOUT_SIZE steps then updates.
#   No replay buffer — data discarded after each update.
#
# Differences from train_sac.py:
#   - Uses PPO agent (agents/ppo.py)
#   - Rollout-based collection (not step-by-step)
#   - No replay buffer
#   - Logs to runs/rlcarla_ppo
#   - Checkpoints as ppo_policy_N.pth
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
from agents.ppo               import PPO
from rlcarla.core.obs_builder import OBS_DIM
from rlcarla.utils.trajectory import (
    draw_trajectory, get_future_waypoints
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("train_ppo")

torch.set_float32_matmul_precision("medium")


# ==========================================================
# CONFIG
# ==========================================================
class PPOConfig:
    """
    PPO training configuration.
    Matches Diffusion-QL/SAC env setup for fair comparison.
    PPO-specific hyperparameters tuned for CARLA.
    """

    WIDTH          = 800
    HEIGHT         = 450
    FPS            = 20

    MAX_EPISODES   = 2000
    MAX_STEPS      = 1000

    # PPO collects full rollout before updating
    ROLLOUT_SIZE   = 2048   # steps per update
    BATCH_SIZE     = 256
    PPO_EPOCHS     = 10
    SAVE_EVERY     = 10

    DISCOUNT       = 0.99
    GAE_LAMBDA     = 0.95
    LR             = 3e-4
    GRAD_NORM      = 0.5
    CLIP_EPSILON   = 0.2
    VF_COEF        = 0.5
    ENT_COEF       = 0.01
    HIDDEN_DIM     = 256

    # Same curriculum
    CURRICULUM = [
        (0,    "empty"),
        (500,  "light"),
        (1000, "medium"),
        (2000, "heavy"),
    ]

    TRAFFIC_LIGHT_CURRICULUM = [
        (0,    "off"),
        (1000, "on"),
    ]

    CHECKPOINT_DIR = "checkpoints_ppo"
    LOG_DIR        = "runs/rlcarla_ppo"
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


cfg = PPOConfig()


# ==========================================================
# HELPERS
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


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return 0, None
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("ppo_policy_") and
        f.endswith(".pth")
    ]
    episodes = []
    for f in files:
        match = re.search(r"ppo_policy_(\d+)\.pth", f)
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
    """Square LiDAR panel — PPO version."""
    font_s = pygame.font.SysFont("monospace", 10)
    bg = pygame.Surface((size, size), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 175))
    screen.blit(bg, (center_x-size//2, center_y-size//2))
    pygame.draw.rect(
        screen, (130, 0, 0),  # red border for PPO
        (center_x-size//2, center_y-size//2, size, size), 1
    )
    pygame.draw.line(
        screen, (40, 0, 0),
        (center_x-size//2, center_y),
        (center_x+size//2, center_y), 1
    )
    pygame.draw.line(
        screen, (40, 0, 0),
        (center_x, center_y-size//2),
        (center_x, center_y+size//2), 1
    )
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
            pygame.draw.circle(screen, color, (rx, ry), 2)
    pygame.draw.circle(
        screen, (255, 100, 0), (center_x, center_y), 5
    )
    title = font_s.render("LiDAR [PPO]", True, (200, 0, 0))
    screen.blit(title, (
        center_x - title.get_width()//2,
        center_y + size//2 + 4
    ))


# ==========================================================
# HUD
# ==========================================================
def draw_hud(screen, font, info, total_steps, episode,
             ep_reward, rollout_steps, traffic_preset,
             tl_mode, view, start_episode):
    lines = [
        f"[PPO] Episode  : {episode} (+{episode-start_episode})",
        f"Step        : {info.get('step', 0)}",
        f"Total Steps : {total_steps}",
        f"Ep Reward   : {ep_reward:.1f}",
        f"Speed       : {info.get('speed', 0):.1f} m/s",
        f"Rollout     : {rollout_steps}/{cfg.ROLLOUT_SIZE}",
        f"Traffic     : {traffic_preset}",
        f"Lights      : {tl_mode}",
        f"View        : {view}",
        f"Map         : {str(info.get('map','?')).split('/')[-1]}",
    ]
    surf = pygame.Surface(
        (260, len(lines)*22+10), pygame.SRCALPHA
    )
    surf.fill((30, 0, 0, 140))   # red tint for PPO
    screen.blit(surf, (5, 5))
    for i, line in enumerate(lines):
        txt = font.render(line, True, (255, 200, 200))
        screen.blit(txt, (10, 10+i*22))


# ==========================================================
# PYGAME INIT
# ==========================================================
pygame.init()
screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
pygame.display.set_caption("RLCarla — PPO Baseline")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 16)

# ==========================================================
# ENV + AGENT
# ==========================================================
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR,        exist_ok=True)

writer = SummaryWriter(log_dir=cfg.LOG_DIR)

env = gym.make("RLCarla-v0")

agent = PPO(
    state_dim    = OBS_DIM,
    action_dim   = 3,
    max_action   = 1.0,
    device       = cfg.DEVICE,
    discount     = cfg.DISCOUNT,
    lr           = cfg.LR,
    grad_norm    = cfg.GRAD_NORM,
    hidden_dim   = cfg.HIDDEN_DIM,
    clip_epsilon = cfg.CLIP_EPSILON,
    ppo_epochs   = cfg.PPO_EPOCHS,
    vf_coef      = cfg.VF_COEF,
    ent_coef     = cfg.ENT_COEF,
    gae_lambda   = cfg.GAE_LAMBDA,
    rollout_size = cfg.ROLLOUT_SIZE,
    batch_size   = cfg.BATCH_SIZE,
)

# Resume if checkpoint exists
start_episode = 0
latest_ep, path = find_latest_checkpoint(cfg.CHECKPOINT_DIR)
if latest_ep > 0:
    agent.load_model(path, id=latest_ep)
    start_episode = latest_ep
    logger.info(f"Resumed PPO from episode {latest_ep}")
else:
    logger.info("PPO starting fresh")

best_reward  = -1e9
total_steps  = 0
current_view = "third_person"

logger.info(f"Algorithm     : PPO (baseline)")
logger.info(f"Device        : {cfg.DEVICE}")
logger.info(f"Obs dim       : {OBS_DIM}")
logger.info(f"Start episode : {start_episode}")
logger.info(f"Target end    : {cfg.MAX_EPISODES}")
logger.info(f"LR            : {cfg.LR}")
logger.info(f"Rollout size  : {cfg.ROLLOUT_SIZE}")
logger.info(f"PPO epochs    : {cfg.PPO_EPOCHS}")

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
        ep_info    = {}
        ppo_metrics= None

        for step in range(cfg.MAX_STEPS):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False

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

            if keys[pygame.K_5]:
                env.unwrapped.set_spectator_mode("follow")
            elif keys[pygame.K_6]:
                env.unwrapped.set_spectator_mode("top")
            elif keys[pygame.K_7]:
                env.unwrapped.set_spectator_mode("none")

            if not running:
                break

            # PPO: get action + value + log_prob
            action, value, log_prob = agent.sample_action(state)

            next_state, reward, done, trunc, info = \
                env.step(action)

            # Clip reward for stability
            clipped_reward = float(
                np.clip(reward, -10.0, 10.0)
            )

            # Collect into rollout buffer
            agent.collect(
                state, action, clipped_reward,
                value, log_prob, float(done or trunc)
            )

            # PPO update when rollout is full
            if agent.ready_to_update():
                ppo_metrics = agent.update(next_state)

            state       = next_state
            ep_reward  += reward
            total_steps += 1
            ep_info     = info

            # Render
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

            draw_hud(
                screen, font, ep_info,
                total_steps, episode, ep_reward,
                agent._step, traffic_preset,
                tl_mode, current_view, start_episode,
            )

            if env.unwrapped._lidar is not None:
                points = env.unwrapped._lidar.get_points()
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

        # Logging
        writer.add_scalar(
            "reward/episode",     ep_reward, episode)
        writer.add_scalar(
            "reward/total_steps", ep_reward, total_steps)
        writer.add_scalar(
            "env/episode_length", step+1,    episode)

        if ppo_metrics:
            writer.add_scalar(
                "loss/actor",
                np.mean(ppo_metrics["actor_loss"]), episode)
            writer.add_scalar(
                "loss/critic",
                np.mean(ppo_metrics["critic_loss"]), episode)
            writer.add_scalar(
                "loss/entropy",
                np.mean(ppo_metrics["entropy"]), episode)

        logger.info(
            f"Ep {episode:04d} (+{episode-start_episode:04d}) | "
            f"Reward {ep_reward:8.2f} | "
            f"Steps {step+1:4d} | "
            f"Traffic {traffic_preset:6s} | "
            f"Done: {ep_info.get('term_reason','trunc')}"
        )

        if episode % cfg.SAVE_EVERY == 0:
            agent.save_model(cfg.CHECKPOINT_DIR, id=episode)
            logger.info(
                f"[PPO] Checkpoint saved — episode {episode}"
            )

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save_model(cfg.CHECKPOINT_DIR, id=9999)
            logger.info(
                f"[PPO] New best — reward {best_reward:.2f}"
            )

except KeyboardInterrupt:
    logger.info("Stopped by user.")

except Exception:
    traceback.print_exc()

finally:
    writer.close()
    env.close()
    pygame.quit()
    logger.info("PPO training finished.")
