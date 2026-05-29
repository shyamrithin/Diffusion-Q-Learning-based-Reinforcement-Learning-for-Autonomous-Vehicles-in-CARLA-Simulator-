# ==========================================================
# infer_sac.py
# SAC Inference Script for RLCarla
#
# Runs trained SAC agent in CARLA simulator.
# Records per-episode metrics for paper results.
#
# Usage:
#   python3 infer_sac.py --traffic light
#   python3 infer_sac.py --traffic medium
#   python3 infer_sac.py --traffic heavy
#   python3 infer_sac.py --traffic empty
#
# CARLA 0.9.16 | Python 3.10
# ==========================================================

import os
import sys
import time
import math
import logging
import argparse
import traceback
import numpy as np
import torch
import pygame
import carla

import gymnasium as gym
import rlcarla

from agents.sac               import SAC
from rlcarla.core.obs_builder import OBS_DIM

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("infer_sac")

torch.set_float32_matmul_precision("medium")

# ==========================================================
# ARGS
# ==========================================================
parser = argparse.ArgumentParser(description="SAC Inference")
parser.add_argument("--checkpoint", type=str,
                    default="checkpoints_sac")
parser.add_argument("--traffic", type=str, default="empty",
                    choices=["empty","light","medium",
                             "heavy","chaos"])
parser.add_argument("--episodes", type=int, default=999999)
parser.add_argument("--width",    type=int, default=1280)
parser.add_argument("--height",   type=int, default=720)
parser.add_argument("--spectator", type=str, default="follow",
                    choices=["follow","top","none"])
parser.add_argument("--lights", type=str, default="off",
                    choices=["on","off"])
args = parser.parse_args()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
CHECKPOINT    = args.checkpoint
WIDTH         = args.width
HEIGHT        = args.height

TRAFFIC_ORDER = ["empty","light","medium","heavy","chaos"]

# ==========================================================
# PYGAME INIT
# ==========================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RLCarla — SAC Inference")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 16)
font_l = pygame.font.SysFont("monospace", 20)

# ==========================================================
# LOAD MODEL
# ==========================================================
def load_model(checkpoint_dir):
    actor_path = os.path.join(
        checkpoint_dir, "sac_actor_9999.pth"
    )
    if not os.path.exists(actor_path):
        raise FileNotFoundError(
            f"SAC checkpoint not found: {actor_path}"
        )
    agent = SAC(
        state_dim  = OBS_DIM,
        action_dim = 3,
        max_action = 1.0,
        device     = DEVICE,
    )
    agent.load_model(checkpoint_dir, id=9999)
    agent.actor.eval()
    agent.critic.eval()
    logger.info(f"SAC loaded from {actor_path}")
    return agent


# ==========================================================
# ACTION SELECTION
# ==========================================================
def get_action(agent, state, prev_action, alpha=0.7):
    with torch.no_grad():
        state_t = torch.FloatTensor(
            state.reshape(1, -1)
        ).to(DEVICE)
        action, _ = agent.actor.sample(state_t)

    raw = action.cpu().numpy().flatten().astype(np.float32)

    # Action smoothing
    smoothed = (alpha * raw +
                (1.0 - alpha) * prev_action).astype(np.float32)

    # Clip to valid ranges
    result = np.zeros(3, dtype=np.float32)
    result[0] = float(np.clip(smoothed[0],  0.0,  1.0))
    result[1] = float(np.clip(smoothed[1], -1.0,  1.0))
    result[2] = float(np.clip(smoothed[2],  0.0,  1.0))
    return result


# ==========================================================
# HUD
# ==========================================================
def draw_hud(screen, font, font_l,
             episode, step, ep_reward,
             speed, collisions, offroads,
             traffic, term_reason=None):

    lines = [
        ("Algorithm",  "SAC"),
        ("Episode",    str(episode)),
        ("Step",       str(step)),
        ("Reward",     f"{ep_reward:.1f}"),
        ("Speed",      f"{speed * 3.6:.1f} km/h"),
        ("Collisions", str(collisions)),
        ("Offroads",   str(offroads)),
        ("Traffic",    traffic),
    ]

    panel_w = 220
    panel_h = len(lines) * 24 + 14
    surf = pygame.Surface(
        (panel_w, panel_h), pygame.SRCALPHA
    )
    surf.fill((0, 0, 0, 160))
    screen.blit(surf, (8, 8))

    for i, (label, value) in enumerate(lines):
        lbl = font.render(
            f"{label:<12}", True, (180, 180, 180)
        )
        val = font.render(value, True, (255, 255, 255))
        y   = 14 + i * 24
        screen.blit(lbl, (14,  y))
        screen.blit(val, (130, y))

    if term_reason:
        color_map = {
            "collision" : (255,  60,  60),
            "offroad"   : (255, 160,   0),
            "wrong_way" : (255, 100,   0),
            "stuck"     : (255, 220,   0),
            "max_steps" : ( 60, 220,  60),
        }
        color  = color_map.get(term_reason, (200, 200, 200))
        label  = term_reason.upper().replace("_", " ")
        banner = font_l.render(label, True, color)
        bx     = WIDTH  // 2 - banner.get_width()  // 2
        by     = HEIGHT // 2 - banner.get_height() // 2
        bg2    = pygame.Surface(
            (banner.get_width() + 40,
             banner.get_height() + 20),
            pygame.SRCALPHA
        )
        bg2.fill((0, 0, 0, 200))
        screen.blit(bg2, (bx - 20, by - 10))
        screen.blit(banner, (bx, by))


# ==========================================================
# MAIN
# ==========================================================
def main():
    agent = load_model(CHECKPOINT)

    env = gym.make("RLCarla-v0")

    current_traffic   = args.traffic
    current_lights    = args.lights
    current_spectator = args.spectator
    traffic_idx       = TRAFFIC_ORDER.index(current_traffic)

    env.unwrapped._traffic_preset    = current_traffic
    env.unwrapped.cfg.TRAFFIC_LIGHTS = current_lights
    env.unwrapped.cfg.SPECTATOR_MODE = current_spectator

    running = True
    episode = 0

    stats = {
        "episodes"   : 0,
        "total_steps": 0,
        "best_reward": -1e9,
        "collisions" : 0,
        "offroads"   : 0,
        "wrong_ways" : 0,
        "completions": 0,
        "total_reward": 0.0,
    }

    logger.info(
        f"Starting SAC inference | "
        f"Traffic: {current_traffic} | "
        f"Checkpoint: {CHECKPOINT}"
    )

    try:
        while running and episode < args.episodes:

            episode += 1
            env.unwrapped._traffic_preset    = current_traffic
            env.unwrapped.cfg.TRAFFIC_LIGHTS = current_lights
            env.unwrapped.cfg.SPECTATOR_MODE = current_spectator

            state, _ = env.reset()
            env.unwrapped.set_camera_view("third_person")

            prev_action = np.zeros(3, dtype=np.float32)
            ep_reward   = 0.0
            step        = 0
            term_reason = None

            logger.info(
                f"Ep {episode} | "
                f"Traffic: {current_traffic}"
            )

            while True:

                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    running = False

                # Traffic controls
                if keys[pygame.K_t]:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        traffic_idx = max(0, traffic_idx-1)
                    else:
                        traffic_idx = min(
                            len(TRAFFIC_ORDER)-1,
                            traffic_idx+1
                        )
                    current_traffic = TRAFFIC_ORDER[traffic_idx]
                    env.unwrapped.set_traffic_preset(
                        current_traffic
                    )
                    time.sleep(0.2)

                # Reset
                if keys[pygame.K_r]:
                    break

                if not running:
                    break

                # Get action
                action = get_action(
                    agent, state, prev_action
                )

                next_state, reward, done, trunc, info = \
                    env.step(action)

                state       = next_state
                prev_action = action
                ep_reward  += reward
                step       += 1
                stats["total_steps"] += 1

                # Render camera
                frame = env.unwrapped.get_camera_frame()
                if frame is not None:
                    if (frame.shape[1] != WIDTH or
                            frame.shape[0] != HEIGHT):
                        import cv2
                        frame = cv2.resize(
                            frame, (WIDTH, HEIGHT)
                        )
                    surf = pygame.surfarray.make_surface(
                        frame.swapaxes(0, 1)
                    )
                    screen.blit(surf, (0, 0))

                # HUD
                speed = info.get("speed", 0.0)
                if env.unwrapped.vehicle:
                    speed = env.unwrapped.vehicle\
                        .get_velocity()
                    speed = math.sqrt(
                        speed.x**2 +
                        speed.y**2 +
                        speed.z**2
                    )

                draw_hud(
                    screen, font, font_l,
                    episode, step, ep_reward,
                    speed,
                    stats["collisions"],
                    stats["offroads"],
                    current_traffic,
                    info.get("term_reason"),
                )

                pygame.display.flip()
                clock.tick(30)

                if done or trunc:
                    term_reason = info.get(
                        "term_reason", "done"
                    )
                    break

            # Episode summary
            stats["episodes"]    += 1
            stats["total_reward"] += ep_reward
            stats["best_reward"]   = max(
                stats["best_reward"], ep_reward
            )

            if term_reason == "collision":
                stats["collisions"] += 1
            elif term_reason == "offroad":
                stats["offroads"]   += 1
            elif term_reason == "wrong_way":
                stats["wrong_ways"] += 1
            elif term_reason == "max_steps":
                stats["completions"] += 1

            avg_reward = (stats["total_reward"] /
                          stats["episodes"])

            logger.info(
                f"Ep {episode:04d} | "
                f"Reward {ep_reward:8.2f} | "
                f"Steps {step:4d} | "
                f"End: {term_reason} | "
                f"Avg: {avg_reward:.2f} | "
                f"Best: {stats['best_reward']:.2f} | "
                f"Col: {stats['collisions']} | "
                f"Off: {stats['offroads']}"
            )

            if term_reason in (
                "collision", "offroad",
                "wrong_lane", "wrong_way"
            ):
                time.sleep(1.5)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception:
        traceback.print_exc()
    finally:
        logger.info("=" * 55)
        logger.info("SAC INFERENCE SUMMARY")
        logger.info(f"  Traffic     : {current_traffic}")
        logger.info(f"  Episodes    : {stats['episodes']}")
        logger.info(f"  Total Steps : {stats['total_steps']}")
        logger.info(
            f"  Avg Reward  : "
            f"{stats['total_reward']/max(1,stats['episodes']):.2f}"
        )
        logger.info(f"  Best Reward : {stats['best_reward']:.2f}")
        logger.info(f"  Collisions  : {stats['collisions']}")
        logger.info(f"  Off Roads   : {stats['offroads']}")
        logger.info(f"  Wrong Ways  : {stats['wrong_ways']}")
        logger.info(f"  Completions : {stats['completions']}")
        logger.info("=" * 55)
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()