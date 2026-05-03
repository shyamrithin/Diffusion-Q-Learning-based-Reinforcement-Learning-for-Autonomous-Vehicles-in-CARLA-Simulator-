import os
import time
import traceback
import logging
import argparse
import math

import gymnasium as gym
import rlcarla
import numpy as np
import torch
import pygame
import carla

from agents.ql_diffusion      import Diffusion_QL
from rlcarla.core.obs_builder  import OBS_DIM
from rlcarla.utils.trajectory  import draw_trajectory, get_future_waypoints

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("infer")

torch.set_float32_matmul_precision("medium")

# ==========================================================
# ARGS
# ==========================================================
parser = argparse.ArgumentParser(description="RLCarla Inference v2")
parser.add_argument("--checkpoint", type=str, default="checkpoints")
parser.add_argument("--traffic", type=str, default="empty",
                    choices=["empty","light","medium","heavy","chaos"])
parser.add_argument("--map", type=str, default=None)
parser.add_argument("--episodes", type=int, default=999999)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--no-trajectory", action="store_true")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--spectator", type=str, default="follow",
                    choices=["follow","top","none"])
parser.add_argument("--lights", type=str, default="off",
                    choices=["on","off"])
parser.add_argument("--alpha", type=float, default=0.7)
args = parser.parse_args()

# ==========================================================
# CONSTANTS
# ==========================================================
WIDTH         = args.width
HEIGHT        = args.height
FPS           = 30
CHECKPOINT    = args.checkpoint
SHOW_TRAJ     = not args.no_trajectory
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAFFIC_ORDER   = ["empty", "light", "medium", "heavy", "chaos"]
VIEW_ORDER      = ["third_person", "driver", "front", "bird_eye"]
SPECTATOR_ORDER = ["follow", "top", "none"]

# ==========================================================
# PYGAME INIT
# ==========================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RLCarla — Inference v2")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont("monospace", 16)
font_l = pygame.font.SysFont("monospace", 20)


# ==========================================================
# 3D LIDAR POINT CLOUD
# ==========================================================
def draw_point_cloud(screen, points, center_x, center_y,
                     radius=90, max_range=30.0):
    font_s = pygame.font.SysFont("monospace", 10)

    bg = pygame.Surface(
        (radius*2+4, radius*2+4), pygame.SRCALPHA
    )
    pygame.draw.circle(bg, (0,0,0,175),
                       (radius+2, radius+2), radius+2)
    screen.blit(bg, (center_x-radius-2, center_y-radius-2))

    pygame.draw.circle(screen, (0,130,0),
                       (center_x,center_y), radius, 1)
    for r in [radius//3, radius*2//3]:
        pygame.draw.circle(screen, (0,55,0),
                           (center_x,center_y), r, 1)

    pygame.draw.line(screen, (0,40,0),
                     (center_x-radius, center_y),
                     (center_x+radius, center_y), 1)
    pygame.draw.line(screen, (0,40,0),
                     (center_x, center_y-radius),
                     (center_x, center_y+radius), 1)

    pygame.draw.polygon(screen, (0,255,100), [
        (center_x,     center_y-radius-7),
        (center_x-4,   center_y-radius+1),
        (center_x+4,   center_y-radius+1),
    ])

    lbl_30 = font_s.render("30m", True, (0,90,0))
    lbl_10 = font_s.render("10m", True, (0,70,0))
    screen.blit(lbl_30, (center_x+4, center_y-radius+2))
    screen.blit(lbl_10, (center_x+4, center_y-radius//3+2))

    if len(points) > 0:
        scale = radius / max_range
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

            dx = rx - center_x
            dy = ry - center_y
            if math.sqrt(dx**2 + dy**2) > radius:
                continue

            pz = z_pts[i]
            if pz < -1.5:   color = (50,  50,  50)
            elif pz < -0.5: color = (100, 100, 100)
            elif pz < 0.5:  color = (0,   180,  80)
            elif pz < 1.5:  color = (255, 140,   0)
            else:           color = (80,  80,  255)

            pygame.draw.circle(screen, color, (rx, ry), 2)

    pygame.draw.circle(screen, (0,200,255), (center_x,center_y), 5)
    pygame.draw.circle(screen, (255,255,255), (center_x,center_y), 2)

    title = font_s.render("LiDAR 3D", True, (0,200,0))
    screen.blit(title, (
        center_x - title.get_width()//2,
        center_y + radius + 5
    ))

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
        pygame.draw.circle(screen, color, (lx+5, ly+4), 4)
        lbl = font_s.render(label, True, color)
        screen.blit(lbl, (lx+12, ly))
        ly += 14


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
        lines = [
            ("Episode",   str(episode)),
            ("Step",      str(step)),
            ("Reward",    f"{ep_reward:.1f}"),
            ("Speed",     f"{speed * 3.6:.1f} km/h"),
            ("Limit",     f"{speed_limit:.0f} km/h"),
            ("View",      view_name),
            ("Traffic",   traffic_preset),
            ("Lights",    tl_mode),
            ("Spectator", spectator_mode),
            ("Smooth α",  f"{alpha:.2f}"),
            ("Map",       map_name.split("/")[-1]),
        ]

        panel_w = 220
        panel_h = len(lines) * 24 + 14
        surf    = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 160))
        self.screen.blit(surf, (8, 8))

        for i, (label, value) in enumerate(lines):
            lbl = self.font.render(f"{label:<10}", True, (180,180,180))
            val = self.font.render(value,           True, (255,255,255))
            y   = 14 + i * 24
            self.screen.blit(lbl, (14,  y))
            self.screen.blit(val, (120, y))

        if is_red_light:
            warn = self.font_l.render("⚠  RED LIGHT", True, (255,60,60))
            self.screen.blit(warn, (
                self.W//2 - warn.get_width()//2, 14
            ))

        controls = [
            "1/2/3/4   : Camera view",
            "5/6/7     : Spectator mode",
            "T / Sft+T : Traffic +/-",
            "L         : Toggle lights",
            "A / Sft+A : Smooth alpha +/-",
            "R         : Reset episode",
            "ESC       : Quit",
        ]
        cy = self.H - len(controls)*20 - 10
        bg = pygame.Surface(
            (230, len(controls)*20+8), pygame.SRCALPHA
        )
        bg.fill((0,0,0,130))
        self.screen.blit(bg, (8, cy-4))
        for line in controls:
            txt = self.font.render(line, True, (200,200,200))
            self.screen.blit(txt, (12, cy))
            cy += 20

        if term_reason:
            reasons = {
                "collision" : ("COLLISION",   (255, 60,  60)),
                "offroad"   : ("OFF ROAD",    (255,160,   0)),
                "wrong_lane": ("WRONG LANE",  (255,200,   0)),
                "wrong_way" : ("WRONG WAY",   (255,100,   0)),
                "stuck"     : ("STUCK",       (255,220,   0)),
                "max_steps" : ("EPISODE END", ( 60,220,  60)),
            }
            label, color = reasons.get(
                term_reason, (term_reason.upper(), (200,200,200))
            )
            banner = self.font_l.render(label, True, color)
            bx     = self.W//2 - banner.get_width()//2
            by     = self.H//2 - banner.get_height()//2
            bg2    = pygame.Surface(
                (banner.get_width()+40, banner.get_height()+20),
                pygame.SRCALPHA
            )
            bg2.fill((0,0,0,200))
            self.screen.blit(bg2, (bx-20, by-10))
            self.screen.blit(banner, (bx, by))

        now          = time.time()
        self._alerts = [(m,e) for m,e in self._alerts if e > now]
        ay = 70
        for msg, _ in self._alerts:
            atxt = self.font_l.render(msg, True, (100,255,100))
            self.screen.blit(atxt, (
                self.W//2 - atxt.get_width()//2, ay
            ))
            ay += 28


# ==========================================================
# LOAD MODEL
# ==========================================================
def load_best_model(checkpoint_dir):
    actor_path = os.path.join(checkpoint_dir, "actor_9999.pth")

    if not os.path.exists(actor_path):
        raise FileNotFoundError(
            f"Best model not found at {actor_path}\n"
            f"Train first with: python3 train_diffusion.py"
        )

    agent = Diffusion_QL(
        state_dim          = OBS_DIM,
        action_dim         = 3,
        max_action         = 1.0,
        device             = DEVICE,
        discount           = 0.99,
        tau                = 0.005,
        eta                = 0.01,
        beta_schedule      = "vp",
        n_timesteps        = 3,        # must match training
        action_temperature = 0.05,
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
    with torch.no_grad():
        state_t   = torch.FloatTensor(
            state.reshape(1,-1)
        ).to(DEVICE)
        state_rpt = torch.repeat_interleave(state_t, repeats=30, dim=0)

        action   = agent.actor.sample(state_rpt)
        q_values = agent.critic_target.q_min(
            state_rpt, action
        ).flatten()

        if deterministic:
            idx = q_values.argmax()
        else:
            probs = torch.softmax(q_values / 0.05, dim=0)
            idx   = torch.multinomial(probs, 1).squeeze()

    raw      = action[idx].cpu().numpy().flatten().astype(np.float32)
    smoothed = (alpha * raw + (1.0 - alpha) * prev_action).astype(np.float32)
    return smoothed


# ==========================================================
# MAIN
# ==========================================================
def main():

    agent = load_best_model(CHECKPOINT)

    env = gym.make("RLCarla-v0")
    hud = HUD(screen, font, font_l, WIDTH, HEIGHT)

    current_view      = VIEW_ORDER[0]
    current_traffic   = args.traffic
    current_spectator = args.spectator
    current_lights    = args.lights
    current_alpha     = args.alpha

    traffic_idx   = TRAFFIC_ORDER.index(current_traffic)

    env.unwrapped._traffic_preset    = current_traffic
    env.unwrapped.cfg.TRAFFIC_LIGHTS = current_lights
    env.unwrapped.cfg.SPECTATOR_MODE = current_spectator

    # 🔥 FIX — set map before reset, not as kwarg
    if args.map is not None:
        env.unwrapped._map_name = args.map

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

            # 🔥 FIX — plain reset(), no map_name kwarg
            state, _ = env.reset()
            env.unwrapped.set_camera_view(current_view)

            prev_action = np.zeros(3, dtype=np.float32)
            ep_reward   = 0.0
            step        = 0
            term_reason = None
            force_reset = False

            logger.info(
                f"Ep {episode} | "
                f"Traffic: {current_traffic} | "
                f"Lights: {current_lights} | "
                f"View: {current_view}"
            )

            while True:

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

                # Spectator
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

                # Traffic
                if keys[pygame.K_t]:
                    if key_mods & pygame.KMOD_SHIFT:
                        traffic_idx = max(0, traffic_idx - 1)
                    else:
                        traffic_idx = min(
                            len(TRAFFIC_ORDER)-1, traffic_idx+1
                        )
                    current_traffic = TRAFFIC_ORDER[traffic_idx]
                    env.unwrapped.set_traffic_preset(current_traffic)
                    hud.alert(f"Traffic: {current_traffic}")
                    time.sleep(0.2)

                # Lights
                if keys[pygame.K_l]:
                    current_lights = "off" \
                        if current_lights == "on" else "on"
                    env.unwrapped.set_traffic_lights(current_lights)
                    hud.alert(f"Lights: {current_lights}")
                    time.sleep(0.2)

                # Smoothing alpha
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

                # ---- Render camera ----
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

                # ---- 3D LiDAR ----
                if (env.unwrapped._lidar is not None and
                        step % 2 == 0):
                    points = env.unwrapped._lidar.get_points()
                    draw_point_cloud(
                        screen, points,
                        center_x  = WIDTH  - 115,
                        center_y  = HEIGHT - 115,
                        radius    = 90,
                        max_range = 30.0,
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
            elif term_reason == "offroad":
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

            if term_reason in (
                "collision", "offroad", "wrong_lane", "wrong_way"
            ):
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