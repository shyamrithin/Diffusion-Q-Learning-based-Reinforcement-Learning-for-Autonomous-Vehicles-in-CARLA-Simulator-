# ==========================================================
# record_signal_demo.py
# Live demonstration driver for the traffic-signal demo.
#
# Runs the trained DQL-E policy through a route with LIVE traffic
# and CYCLING traffic lights, gated by SignalAwareWrapper (a
# rule-based supervisory layer). Drives continuously so you can
# screen-record it. No video is written by this script.
#
# HONEST SCOPE (read this):
#   - DRIVING (steering, base throttle, traffic avoidance) is the
#     learned DQL-E policy.
#   - STOPPING AT LIGHTS and STOPPING BEHIND TRAFFIC is done by the
#     rule-based SignalAwareWrapper, NOT learned behaviour. DQL-E
#     was trained with lights frozen green and never learned to
#     respond to red. Describe the demo accordingly.
#
# This script is SELF-CONTAINED: it does not import record_eval.py
# (that module parses argv at import time). The DQL-E loading below
# — including the critic_target sync fix — is copied from
# record_eval.load_agent() so behaviour matches the evaluations.
#
# Usage:
#   conda activate diffusioncarla
#   # CARLA server running in another terminal
#   python3 record_signal_demo.py --route route_1_roundabout \
#       --traffic medium --ckpt_id 9999
#
#   --route    route_1_roundabout | route_2_curve | route_3_straight
#   --traffic  empty | light | medium | heavy
#   --loops    how many times to drive the route (default 3)
#   --verbose  print per-step gate telemetry
# ==========================================================

import argparse
import logging
import numpy as np
import torch
import carla
import gymnasium as gym
import rlcarla
from rlcarla.core.obs_builder import OBS_DIM
import route_utils as ru
from signal_aware_wrapper import SignalAwareWrapper

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("signal_demo")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------
# Optional pygame render: camera view + large light-status HUD.
# Also paces the loop to ~30 FPS so playback is real-time
# (without it, the loop runs faster than wall-clock and the
# CARLA view looks sped up).
# ----------------------------------------------------------
_PG = {"screen": None, "font": None, "bigfont": None,
       "clock": None}


def init_render(width, height):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("DQL-E + signal supervisor (demo)")
    _PG["screen"] = screen
    _PG["font"] = pygame.font.SysFont("monospace", 18)
    _PG["bigfont"] = pygame.font.SysFont("monospace", 40, bold=True)
    _PG["clock"] = pygame.time.Clock()


def render_frame(env, width, height, light_state, light_dist,
                 thr, steer, brk, step):
    """Draw camera view + HUD. Returns False if window closed."""
    import pygame
    screen = _PG["screen"]
    if screen is None:
        return True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    if pygame.key.get_pressed()[pygame.K_ESCAPE]:
        return False

    frame = None
    try:
        frame = env.unwrapped.get_camera_frame()
    except Exception:
        frame = None
    if frame is not None:
        try:
            import cv2
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
        except Exception:
            pass
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
    else:
        screen.fill((20, 20, 20))

    # ---- BIG light-status panel (top-centre) ----
    colors = {"Red": (220, 40, 40), "Yellow": (230, 200, 40),
              "Green": (40, 200, 80)}
    label = light_state if light_state else "NO LIGHT"
    col = colors.get(light_state, (130, 130, 130))
    panel_w, panel_h = 360, 90
    px = (width - panel_w) // 2
    bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 180))
    screen.blit(bg, (px, 12))
    # colored signal dot
    pygame.draw.circle(screen, col, (px + 48, 12 + panel_h // 2), 30)
    pygame.draw.circle(screen, (255, 255, 255),
                       (px + 48, 12 + panel_h // 2), 30, 2)
    big = _PG["bigfont"].render(label.upper(), True, col)
    screen.blit(big, (px + 95, 12 + 24))
    if light_dist != float("inf"):
        d = _PG["font"].render(f"{light_dist:.0f} m ahead", True,
                               (230, 230, 230))
        screen.blit(d, (px + 98, 12 + 62))

    # ---- telemetry HUD (top-left) ----
    lines = [
        "DQL-E Traffic light demo",
        f"step {step}",
        f"throttle {thr:.2f}   brake {brk:.2f}   steer {steer:+.2f}",
    ]
    hud = pygame.Surface((520, len(lines) * 24 + 12),
                         pygame.SRCALPHA)
    hud.fill((0, 0, 0, 160))
    screen.blit(hud, (8, 8))
    for i, ln in enumerate(lines):
        screen.blit(_PG["font"].render(ln, True, (255, 255, 255)),
                    (14, 14 + i * 24))

    pygame.display.flip()
    _PG["clock"].tick(30)   # real-time pacing
    return True

# ---------------- args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--route", default="route_1_roundabout",
                    choices=list(ru.ROUTES.keys()))
parser.add_argument("--traffic", default="medium",
                    choices=["empty", "light", "medium", "heavy"])
parser.add_argument("--ckpt_id", default="9999")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--loops", type=int, default=3)
parser.add_argument("--max_steps", type=int, default=1500)
parser.add_argument("--spawn", type=int, default=None,
                    help="spawn point index override (e.g. from "
                         "find_traffic_lights.py). If set, the ego "
                         "spawns here instead of the route start.")
parser.add_argument("--freeroam", action="store_true",
                    help="no destination flag; drive continuously "
                         "until max_steps (good for recording).")
parser.add_argument("--script_light", action="store_true",
                    help="force the light the ego approaches to stay "
                         "RED, then flip GREEN after --green_after "
                         "steps of the ego being stopped. Guarantees "
                         "a clean red->green on camera.")
parser.add_argument("--green_after", type=int, default=80,
                    help="steps the ego is held at red before the "
                         "scripted light turns green (default 80).")
parser.add_argument("--stop_after_pass", type=int, default=None,
                    help="end the episode this many steps after the "
                         "scripted light turns green (so the take "
                         "ends cleanly once the ego clears the light).")
parser.add_argument("--render", action="store_true",
                    help="open a pygame window showing the camera "
                         "view + a large RED/YELLOW/GREEN light HUD. "
                         "Also paces the sim to real-time (~30 FPS).")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--num_vehicles", type=int, default=None,
                    help="override NPC vehicle count for denser "
                         "traffic (else uses --traffic preset).")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

CKPT_DIR = args.checkpoint or "checkpoints"


def _to_env_action(a):
    return np.array([
        float(np.clip(a[0],  0.0, 1.0)),
        float(np.clip(a[1], -1.0, 1.0)),
        float(np.clip(a[2],  0.0, 1.0)),
    ], dtype=np.float32)


# ---------------- DQL-E loader (copied from record_eval) ----------------
def load_dqle():
    from agents.ql_diffusion import Diffusion_QL
    agent = Diffusion_QL(
        state_dim=OBS_DIM, action_dim=3, max_action=1.0,
        device=DEVICE, discount=0.99, tau=0.005, eta=0.001,
        beta_schedule="vp", n_timesteps=3, lr=3e-5,
        grad_norm=0.05, action_temperature=0.05,
        alpha=0.2, auto_alpha=True,
    )
    agent.load_model(CKPT_DIR, id=int(args.ckpt_id))
    # CRITICAL: sync critic_target<-critic, else sample_action()
    # scores its candidates with a random critic -> idle bug.
    import copy as _copy
    agent.critic_target = _copy.deepcopy(agent.critic)
    try:
        agent.actor.eval()
    except Exception:
        pass

    def act_fn(state, prev, alpha=0.7):
        raw = np.asarray(agent.sample_action(state), dtype=np.float32)
        sm = (alpha * raw + (1 - alpha) * prev)
        return _to_env_action(sm)

    logger.info(f"Loaded DQL-E from {CKPT_DIR} id={args.ckpt_id}")
    return agent, act_fn


# ---------------- main ----------------
def main():
    env = gym.make("RLCarla-v0")   # matches record_eval.main()

    # gym.make wraps the env in a TimeLimit using the registration's
    # max_episode_steps (1500), which truncates the episode (with no
    # term_reason) regardless of our --max_steps. Lift it for long
    # demo takes by reaching the TimeLimit wrapper's counter.
    try:
        e = env
        while e is not None:
            if hasattr(e, "_max_episode_steps"):
                e._max_episode_steps = args.max_steps + 100
                break
            e = getattr(e, "env", None)
    except Exception:
        pass

    agent, act_fn = load_dqle()

    world = env.unwrapped.world
    cmap = env.unwrapped.carla_map

    if args.render:
        init_render(args.width, args.height)
    # optional denser traffic
    if args.num_vehicles is not None:
        try:
            env.unwrapped.cfg.NUM_VEHICLES = args.num_vehicles
            logger.info(f"NPC vehicle count -> {args.num_vehicles}")
        except Exception:
            logger.info("could not override NUM_VEHICLES")

    spawn_idx = ru.ROUTES[args.route]["spawn"]
    dest_idx = ru.ROUTES[args.route]["dest"]
    ref_xy, start_tf = ru.generate_reference_route(
        world, cmap, spawn_idx, dest_idx
    )
    dest_xy = ref_xy[-1] if len(ref_xy) else None
    DEST_RADIUS = 6.0

    # spawn override: start at a specific spawn point (e.g. one near
    # a traffic light, from find_traffic_lights.py) instead of the
    # route start. In freeroam there is no destination anyway.
    if args.spawn is not None:
        sps = cmap.get_spawn_points()
        if not (0 <= args.spawn < len(sps)):
            raise SystemExit(f"--spawn {args.spawn} out of range "
                             f"(0..{len(sps)-1})")
        start_tf = sps[args.spawn]
        logger.info(f"spawn override: spawn point {args.spawn} "
                    f"at ({start_tf.location.x:.1f},"
                    f"{start_tf.location.y:.1f})")
    else:
        logger.info(f"[{args.route}] {len(ref_xy)} wps "
                    f"spawn={spawn_idx} dest={dest_idx}")

    for loop in range(1, args.loops + 1):
        env.unwrapped.set_eval_spawn(start_tf)
        env.unwrapped._traffic_preset = args.traffic
        # KEY DIFFERENCE vs eval: cycling lights, not frozen green
        env.unwrapped.cfg.TRAFFIC_LIGHTS = "normal"
        # the wrapper intentionally stops the car at red lights and
        # behind traffic; disable the stuck-detector so a legitimate
        # wait does not end the demo episode
        env.unwrapped.cfg.STUCK_LIMIT = 100000
        # the env has its own MAX_STEPS cap (default 1000) that ends
        # the episode before our --max_steps; raise it so long demo
        # takes run to completion.
        env.unwrapped.cfg.MAX_STEPS = args.max_steps + 10
        state, _ = env.reset()
        try:
            env.unwrapped.set_camera_view("third_person")
        except Exception:
            pass

        # build the wrapper now that the vehicle + traffic exist
        veh = env.unwrapped.vehicle
        traffic = env.unwrapped._traffic
        wrapper = SignalAwareWrapper(veh, traffic, world=world,
                                     verbose=args.verbose)

        prev = np.zeros(3, dtype=np.float32)
        logger.info(f"=== Demo loop {loop}/{args.loops} "
                    f"route={args.route} traffic={args.traffic} ===")

        # light-scripting state: once the ego is near a light we force
        # it RED, count stopped steps, then flip it GREEN for a clean
        # red->green sequence on camera.
        scripted_tl = None
        stopped_steps = 0
        turned_green = False
        steps_since_green = 0
        info = {}
        quit_demo = False

        for step in range(args.max_steps):
            # --- script the approached light (optional) ---
            if args.script_light and not turned_green:
                tl, tld = wrapper.nearest_light_ahead()
                if tl is not None and tld <= 30.0:
                    if scripted_tl is None:
                        scripted_tl = tl
                        try:
                            scripted_tl.set_state(
                                carla.TrafficLightState.Red)
                            scripted_tl.freeze(True)  # hold the state
                            logger.info(f"[{loop}] scripting light "
                                        f"{tl.id}: held RED")
                        except Exception as e:
                            logger.info(f"light script err: {e}")
                    # count how long ego has been ~stopped at it
                    spd = info.get("speed_kmh", 0.0) if step else 0.0
                    if spd < 1.0:
                        stopped_steps += 1
                    if stopped_steps >= args.green_after:
                        try:
                            scripted_tl.set_state(
                                carla.TrafficLightState.Green)
                            logger.info(f"[{loop}] light "
                                        f"{scripted_tl.id} -> GREEN")
                        except Exception as e:
                            logger.info(f"green err: {e}")
                        turned_green = True

            raw = act_fn(state, prev)        # DQL-E action
            thr, steer, brk = wrapper.apply(raw)   # gated
            gated = np.array([thr, steer, brk], dtype=np.float32)

            state, reward, done, trunc, info = env.step(gated)
            prev = gated

            # light telemetry so the terminal shows what's happening
            ls_state, ls_dist = None, float("inf")
            try:
                ls_state, ls_dist = wrapper._light_state_and_distance()
                ls_state = (str(ls_state).split(".")[-1]
                            if ls_state is not None else None)
            except Exception:
                pass
            if step % 10 == 0 or args.verbose:
                lstr = (f"{ls_state}@{ls_dist:.0f}m"
                        if ls_state else "none")
                logger.info(f"[{loop}] step {step:4d} light={lstr} "
                            f"thr={thr:.2f} steer={steer:+.2f} "
                            f"brake={brk:.2f}")
                if args.verbose:
                    try:
                        g = wrapper.debug_light_geom()
                        if g:
                            logger.info(f"      geom {g}")
                    except Exception:
                        pass

            # render window (camera + big light HUD); paces real-time
            if args.render:
                ok = render_frame(env, args.width, args.height,
                                  ls_state, ls_dist, thr, steer, brk,
                                  step)
                if not ok:
                    quit_demo = True
                    break

            # end loop early if destination reached — but NOT in
            # freeroam or spawn-override mode (we want a long take
            # that drives through lights, not an early finish)
            if (not args.freeroam and args.spawn is None
                    and dest_xy is not None):
                loc = veh.get_transform().location
                import math
                if math.hypot(loc.x - dest_xy[0],
                              loc.y - dest_xy[1]) <= DEST_RADIUS:
                    logger.info(f"[{loop}] reached destination "
                                f"at step {step}")
                    break
            # end the take cleanly once the ego has cleared the
            # scripted light (avoids driving into downstream junctions
            # where light association is unreliable)
            if turned_green and args.stop_after_pass is not None:
                steps_since_green += 1
                if steps_since_green >= args.stop_after_pass:
                    logger.info(f"[{loop}] take complete — "
                                f"{args.stop_after_pass} steps after "
                                f"green; ending cleanly")
                    break

            if done or trunc:
                logger.info(f"[{loop}] episode ended "
                            f"({info.get('term_reason')}) at step {step}")
                break

        if quit_demo:
            logger.info("window closed — stopping demo")
            break

    env.close()
    if args.render:
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
    logger.info("demo finished")


if __name__ == "__main__":
    main()