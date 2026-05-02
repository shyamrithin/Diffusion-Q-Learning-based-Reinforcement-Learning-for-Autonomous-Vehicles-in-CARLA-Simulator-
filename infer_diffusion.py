# ==========================================================
# infer_diffusion.py
# RLCarla Inference Script — Full Rewrite v2
# Frame stacking (564D) + Action smoothing
# Switchable cameras, traffic density, trajectory overlay
# CARLA 0.9.16 compatible
# ==========================================================

import os
import time
import traceback
import logging
import argparse

import gymnasium as gym
import rlcarla
import numpy as np
import torch
import pygame
import carla

from agents.ql_diffusion      import Diffusion_QL
from rlcarla.core.obs_builder  import OBS_DIM
from rlcarla.utils.trajectory  import draw_trajectory, get_future_waypoints

# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("infer")

# ==========================================================
# PERFORMANCE
# ==========================================================
torch.set_float32_matmul_precision("medium")

# ==========================================================
# ARGS
# ==========================================================
parser = argparse.ArgumentParser(description="RLCarla Inference v2")
parser.add_argument("--checkpoint", type=str, default="checkpoints",
                    help="Directory containing actor_9999.pth")
parser.add_argument("--traffic", type=str, default="light",
                    choices=["empty","light","medium","heavy","chaos"],
                    help="Starting traffic preset")
parser.add_argument("--map", type=str, default=None,
                    help="CARLA map to load. None = use current.")
parser.add_argument("--episodes", type=int, default=999999,
                    help="Number of episodes (default: infinite)")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--no-trajectory", action="store_true",
                    help="Disable trajectory overlay")
parser.add_argument("--deterministic", action="store_true",
                    help="Argmax Q instead of softmax sampling")
parser.add_argument("--spectator", type=str, default="follow",
                    choices=["follow","top","none"],
                    help="CARLA world spectator mode")
parser.add_argument("--lights", type=str, default="on",
                    choices=["on","off"],
                    help="Traffic lights on/off (off=frozen green)")
parser.add_argument("--alpha", type=float, default=0.7,
                    help="Action smoothing alpha (0=max smooth, 1=none)")
args = parser.parse_args()

# ==========================================================
# CONSTANTS
# ==========================================================
WIDTH        = args.width
HEIGHT       = args.height
FPS          = 30
CHECKPOINT   = args.checkpoint
SHOW_TRAJ    = not args.no_trajectory
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAFFIC_ORDER = ["empty", "light", "medium", "heavy", "chaos"]
VIEW_ORDER    = ["third_person", "driver", "front", "bird_eye"]
SPECTATOR_ORDER = ["follow", "top", "none"]

# ==========================================================
# PYGAME INIT
# ==========================================================
pygame.init()
screen  = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RLCarla — Inference v2")
clock   = pygame.time.Clock()
font    = pygame.font.SysFont("monospace", 16)
font_l  = pygame.font.SysFont("monospace", 20)


# ==========================================================
# HUD
# ==========================================================
class HUD:

    def __init__(self, screen, font, font_l, width, height):
        self.screen  = screen
        self.font    = font
        self.font_l  = font_l
        self.W       = width
        self.H       = height
        self._alerts = []

    def alert(self, msg, duration=2.0):
        self._alerts.append((msg, time.time() + duration))

    def draw(
        self,
        episode, step, ep_reward,
        speed, speed_limit,
        view_name, traffic_preset,
        map_name, is_red_light,
        tl_mode, spectator_mode,
        alpha, term_reason=None,
    ):
        # ---- Left stats panel ----
        lines = [
            ("Episode",    str(episode)),
            ("Step",       str(step)),
            ("Reward",     f"{ep_reward:.1f}"),
            ("Speed",      f"{speed * 3.6:.1f} km/h"),
            ("Limit",      f"{speed_limit:.0f} km/h"),
            ("View",       view_name),
            ("Traffic",    traffic_preset),
            ("Lights",     tl_mode),
            ("Spectator",  spectator_mode),
            ("Smooth α",   f"{alpha:.2f}"),
            ("Map",        map_name.split("/")[-1]),
        ]

        panel_w = 220
        panel_h = len(lines) * 24 + 14
        surf    = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 160))
        self.screen.blit(surf, (8, 8))

        for i, (label, value) in enumerate(lines):
            lbl = self.font.render(f"{label:<10}", True, (180, 180, 180))
            val = self.font.render(value,           True, (255, 255, 255))
            y   = 14 + i * 24
            self.screen.blit(lbl, (14,  y))
            self.screen.blit(val, (120, y))

        # ---- Red light warning ----
        if is_red_light:
            warn = self.font_l.render(
                "  RED LIGHT", True, (255, 60, 60)
            )
            self.screen.blit(
                warn,
                (self.W // 2 - warn.get_width() // 2, 14)
            )

        # ---- Controls legend ----
        controls = [
            "1/2/3/4   : Camera view",
            "5/6/7     : Spectator mode",
            "T / Sft+T : Traffic density",
            "L         : Toggle lights",
            "A / Sft+A : Smoothing alpha",
            "R         : Reset episode",
            "ESC       : Quit",
        ]
        cy = self.H - len(controls) * 20 - 10
        bg = pygame.Surface(
            (230, len(controls) * 20 + 8), pygame.SRCALPHA
        )
        bg.fill((0, 0, 0, 130))
        self.screen.blit(bg, (8, cy - 4))
        for line in controls:
            txt = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(txt, (12, cy))
            cy += 20

        # ---- Episode done banner ----
        if term_reason:
            reasons = {
                "collision" : ("COLLISION",   (255,  60,  60)),
                "offroad"   : ("OFF ROAD",    (255, 160,   0)),
                "wrong_lane": ("WRONG LANE",  (255, 200,   0)),
                "wrong_way" : ("WRONG WAY",   (255, 100,   0)),
                "stuck"     : ("STUCK",       (255, 220,   0)),
                "max_steps" : ("EPISODE END", ( 60, 220,  60)),
            }
            label, color = reasons.get(
                term_reason,
                (term_reason.upper(), (200, 200, 200))
            )
            banner = self.font_l.render(label, True, color)
            bx     = self.W // 2 - banner.get_width()  // 2
            by     = self.H // 2 - banner.get_height() // 2
            bg2    = pygame.Surface(
                (banner.get_width() + 40, banner.get_height() + 20),
                pygame.SRCALPHA
            )
            bg2.fill((0, 0, 0, 200))
            self.screen.blit(bg2, (bx - 20, by - 10))
            self.screen.blit(banner, (bx, by))

        # ---- Temporary alerts ----
        now          = time.time()
        self._alerts = [(m, e) for m, e in self._alerts if e > now]
        ay = 70
        for msg, _ in self._alerts:
            atxt = self.font_l.render(msg, True, (100, 255, 100))
            self.screen.blit(
                atxt,
                (self.W // 2 - atxt.get_width() // 2, ay)
            )
            ay += 28


# ==========================================================
# LOAD MODEL
# ==========================================================
def load_best_model(checkpoint_dir):
    actor_path  = os.path.join(checkpoint_dir, "actor_9999.pth")
    critic_path = os.path.join(checkpoint_dir, "critic_9999.pth")

    if not os.path.exists(actor_path):
        raise FileNotFoundError(
            f"Best model not found at {actor_path}\n"
            f"Train first with: python3 train_diffusion.py"
        )

    agent = Diffusion_QL(
        state_dim          = OBS_DIM,   # 564
        action_dim         = 3,
        max_action         = 1.0,
        device             = DEVICE,
        discount           = 0.99,
        tau                = 0.005,
        eta                = 0.01,
        beta_schedule      = "vp",
        n_timesteps        = 5,
        action_temperature = 0.05,      # lower temp at inference = more confident
    )
    agent.load_model(checkpoint_dir, id=9999)
    agent.actor.eval()
    agent.critic.eval()
    agent.ema_model.eval()

    logger.info(f"Loaded best model from {actor_path}")
    return agent


# ==========================================================
# ACTION SELECTION
# ==========================================================
def get_action(agent, state, prev_action, alpha, deterministic=False):
    """
    Get action from policy with smoothing applied.

    deterministic=True  → argmax Q (best for benchmarking)
    deterministic=False → softmax sampling (more natural driving)
    """
    with torch.no_grad():
        state_t   = torch.FloatTensor(
            state.reshape(1, -1)
        ).to(DEVICE)
        state_rpt = torch.repeat_interleave(state_t, repeats=50, dim=0)

        action    = agent.actor.sample(state_rpt)
        q_values  = agent.critic_target.q_min(
            state_rpt, action
        ).flatten()

        if deterministic:
            idx = q_values.argmax()
        else:
            probs = torch.softmax(q_values / 0.05, dim=0)
            idx   = torch.multinomial(probs, 1).squeeze()

    raw = action[idx].cpu().numpy().flatten().astype(np.float32)

    # Apply smoothing
    smoothed = (alpha * raw + (1.0 - alpha) * prev_action).astype(np.float32)
    return smoothed


# ==========================================================
# MAIN
# ==========================================================
def main():

    agent = load_best_model(CHECKPOINT)

    env = gym.make("RLCarla-v0")
    hud = HUD(screen, font, font_l, WIDTH, HEIGHT)

    # Runtime state
    current_view      = VIEW_ORDER[0]
    current_traffic   = args.traffic
    current_spectator = args.spectator
    current_lights    = args.lights
    current_alpha     = args.alpha

    traffic_idx   = TRAFFIC_ORDER.index(current_traffic)
    spectator_idx = SPECTATOR_ORDER.index(current_spectator)

    env.unwrapped._traffic_preset    = current_traffic
    env.unwrapped.cfg.TRAFFIC_LIGHTS = current_lights

    running = True
    episode = 0

    stats = {
        "episodes"   : 0,
        "total_steps": 0,
        "best_reward": -1e9,
        "collisions" : 0,
        "offroads"   : 0,
        "wrong_lanes": 0,
        "wrong_ways" : 0,
        "completions": 0,
    }

    try:
        while running and episode < args.episodes:

            episode += 1

            env.unwrapped._traffic_preset    = current_traffic
            env.unwrapped.cfg.TRAFFIC_LIGHTS = current_lights
            env.unwrapped.cfg.SPECTATOR_MODE = current_spectator

            state, _ = env.reset(map_name=args.map)
            env.unwrapped.set_camera_view(current_view)

            prev_action = np.zeros(3, dtype=np.float32)
            ep_reward   = 0.0
            step        = 0
            term_reason = None
            force_reset = False

            logger.info(
                f"Ep {episode} | Traffic: {current_traffic} | "
                f"Lights: {current_lights} | View: {current_view}"
            )

            while True:

                # ---- Events ----
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                keys     = pygame.key.get_pressed()
                key_mods = pygame.key.get_mods()

                if keys[pygame.K_ESCAPE]:
                    running = False

                # Camera
                if keys[pygame.K_1]:
                    current_view = "third_person"
                    env.unwrapped.set_camera_view(current_view)
                    hud.alert("Camera: Third Person")
                elif keys[pygame.K_2]:
                    current_view = "driver"
                    env.unwrapped.set_camera_view(current_view)
                    hud.alert("Camera: Driver")
                elif keys[pygame.K_3]:
                    current_view = "front"
                    env.unwrapped.set_camera_view(current_view)
                    hud.alert("Camera: Front")
                elif keys[pygame.K_4]:
                    current_view = "bird_eye"
                    env.unwrapped.set_camera_view(current_view)
                    hud.alert("Camera: Bird Eye")

                # Spectator mode
                if keys[pygame.K_5]:
                    current_spectator = "follow"
                    env.unwrapped.set_spectator_mode("follow")
                    hud.alert("Spectator: Follow")
                elif keys[pygame.K_6]:
                    current_spectator = "top"
                    env.unwrapped.set_spectator_mode("top")
                    hud.alert("Spectator: Top")
                elif keys[pygame.K_7]:
                    current_spectator = "none"
                    env.unwrapped.set_spectator_mode("none")
                    hud.alert("Spectator: None")

                # Traffic density
                if keys[pygame.K_t]:
                    if key_mods & pygame.KMOD_SHIFT:
                        traffic_idx   = max(0, traffic_idx - 1)
                    else:
                        traffic_idx   = min(
                            len(TRAFFIC_ORDER) - 1, traffic_idx + 1
                        )
                    current_traffic = TRAFFIC_ORDER[traffic_idx]
                    env.unwrapped.set_traffic_preset(current_traffic)
                    hud.alert(f"Traffic: {current_traffic}")
                    time.sleep(0.2)

                # Traffic lights toggle
                if keys[pygame.K_l]:
                    current_lights = "off" \
                        if current_lights == "on" else "on"
                    env.unwrapped.set_traffic_lights(current_lights)
                    hud.alert(f"Lights: {current_lights}")
                    time.sleep(0.2)

                # Action smoothing alpha
                if keys[pygame.K_a]:
                    if key_mods & pygame.KMOD_SHIFT:
                        current_alpha = max(0.0, current_alpha - 0.1)
                    else:
                        current_alpha = min(1.0, current_alpha + 0.1)
                    current_alpha = round(current_alpha, 1)
                    hud.alert(f"Smooth α: {current_alpha}")
                    time.sleep(0.15)

                # Force reset
                if keys[pygame.K_r]:
                    force_reset = True
                    hud.alert("Resetting...")

                if not running or force_reset:
                    break

                # ---- Action ----
                action = get_action(
                    agent, state, prev_action,
                    alpha         = current_alpha,
                    deterministic = args.deterministic,
                )

                next_state, reward, done, trunc, info = env.step(action)

                state       = next_state
                prev_action = action
                ep_reward  += reward
                step       += 1
                stats["total_steps"] += 1

                # ---- Render ----
                frame = env.unwrapped.get_camera_frame()

                if frame is not None:
                    if (frame.shape[1] != WIDTH or
                            frame.shape[0] != HEIGHT):
                        import cv2
                        frame = cv2.resize(frame, (WIDTH, HEIGHT))

                    if SHOW_TRAJ:
                        cam_tf = env.unwrapped.get_camera_transform()
                        K      = env.unwrapped.get_camera_intrinsic()

                        if cam_tf is not None and K is not None:
                            waypoints = get_future_waypoints(
                                env.unwrapped.vehicle,
                                env.unwrapped.carla_map,
                                n=25, spacing=3.0,
                            )
                            frame = draw_trajectory(
                                frame, waypoints, K, cam_tf,
                                WIDTH, HEIGHT,
                            )

                    surf = pygame.surfarray.make_surface(
                        frame.swapaxes(0, 1)
                    )
                    screen.blit(surf, (0, 0))

                # ---- HUD ----
                speed       = info.get("speed", 0.0)
                speed_limit = 50.0
                is_red      = False

                if env.unwrapped.vehicle:
                    speed_limit = env.unwrapped.vehicle.get_speed_limit()
                    is_red = (
                        env.unwrapped.vehicle.get_traffic_light_state()
                        == carla.TrafficLightState.Red
                    )

                map_name = (
                    env.unwrapped.carla_map.name
                    if env.unwrapped.carla_map else "?"
                )

                hud.draw(
                    episode        = episode,
                    step           = step,
                    ep_reward      = ep_reward,
                    speed          = speed,
                    speed_limit    = speed_limit,
                    view_name      = current_view,
                    traffic_preset = current_traffic,
                    map_name       = map_name,
                    is_red_light   = is_red,
                    tl_mode        = current_lights,
                    spectator_mode = current_spectator,
                    alpha          = current_alpha,
                    term_reason    = info.get("term_reason"),
                )

                pygame.display.flip()
                clock.tick(FPS)

                if done or trunc:
                    term_reason = info.get("term_reason", "done")
                    break

            # ---- Episode stats ----
            stats["episodes"]    += 1
            stats["best_reward"]  = max(
                stats["best_reward"], ep_reward
            )

            if term_reason == "collision":
                stats["collisions"]  += 1
            elif term_reason in ("offroad",):
                stats["offroads"]    += 1
            elif term_reason == "wrong_lane":
                stats["wrong_lanes"] += 1
            elif term_reason == "wrong_way":
                stats["wrong_ways"]  += 1
            elif term_reason == "max_steps":
                stats["completions"] += 1

            logger.info(
                f"Ep {episode:04d} | "
                f"Reward {ep_reward:8.2f} | "
                f"Steps {step:4d} | "
                f"End: {term_reason} | "
                f"Best: {stats['best_reward']:.2f}"
            )

            if term_reason in ("collision", "offroad",
                               "wrong_lane", "wrong_way"):
                time.sleep(1.5)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")

    except Exception:
        traceback.print_exc()

    finally:
        logger.info("=" * 50)
        logger.info("INFERENCE SUMMARY")
        logger.info(f"  Episodes    : {stats['episodes']}")
        logger.info(f"  Total Steps : {stats['total_steps']}")
        logger.info(f"  Best Reward : {stats['best_reward']:.2f}")
        logger.info(f"  Collisions  : {stats['collisions']}")
        logger.info(f"  Off Roads   : {stats['offroads']}")
        logger.info(f"  Wrong Lanes : {stats['wrong_lanes']}")
        logger.info(f"  Wrong Ways  : {stats['wrong_ways']}")
        logger.info(f"  Completions : {stats['completions']}")
        logger.info("=" * 50)

        env.close()
        pygame.quit()
        logger.info("Inference finished.")


if __name__ == "__main__":
    main()
