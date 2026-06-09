# ==========================================================
# record_eval.py
# Unified fixed-route evaluation + CSV recording for ALL
# three agents (DQL-E, SAC, PPO) on identical routes.
#
# Why this exists:
#   For a FAIR comparison every agent must drive the SAME
#   start point and be measured against the SAME reference
#   path. This script spawns the ego at a fixed route start,
#   runs the chosen agent open-loop, and logs per-step
#   telemetry plus route-completion and trajectory-error
#   metrics computed against a GlobalRoutePlanner reference.
#
# Produces, per (agent, route, traffic):
#   results/<agent>_<traffic>/<route>_ep01.csv ...   (per-step)
#   results/<agent>_<traffic>/<route>_summary.csv    (per-ep)
#   All agents share IDENTICAL CSV columns so the same
#   post-processing works across DQL-E / SAC / PPO.
#
# Usage examples:
#   python3 record_eval.py --agent dqle --traffic heavy \
#       --route route_1_roundabout --episodes 10
#   python3 record_eval.py --agent sac  --traffic empty \
#       --route all --episodes 10
#   python3 record_eval.py --agent ppo  --traffic medium \
#       --route all --episodes 10
#
# NOTE: Fill ROUTES indices in route_utils.py first
#       (run `python3 route_utils.py` with CARLA up).
#
# CARLA 0.9.16 | Python 3.10
# ==========================================================

import os
import sys
import math
import time
import csv
import logging
import argparse
import traceback

import numpy as np
import torch
import pygame
import carla
import gymnasium as gym
import rlcarla

from rlcarla.core.obs_builder import OBS_DIM
import route_utils as ru

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("record_eval")
torch.set_float32_matmul_precision("medium")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# ARGS
# ==========================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent", required=True,
    choices=["dqle", "sac", "ppo"],
    help="Which agent to evaluate",
)
parser.add_argument(
    "--checkpoint", default=None,
    help="Checkpoint dir (defaults per agent)",
)
parser.add_argument(
    "--ckpt_id", default="9999",
    help="Checkpoint id to load (e.g. 9999 for best, "
         "or 2000 for a specific episode)",
)
parser.add_argument(
    "--traffic", default="empty",
    choices=["empty", "light", "medium", "heavy", "chaos"],
)
parser.add_argument(
    "--route", default="all",
    help="Route key from route_utils.ROUTES, or 'all'",
)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--width",  type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument(
    "--noise", type=float, default=0.0,
    help="Gaussian obs noise std fraction (0.0, 0.05, "
         "0.10, 0.20) for robustness experiments",
)
parser.add_argument(
    "--no_render", action="store_true",
    help="Disable pygame rendering for faster eval",
)
args = parser.parse_args()

DEFAULT_CKPT = {
    "dqle": "checkpoints",
    "sac":  "checkpoints_sac",
    "ppo":  "checkpoints_ppo",
}
CKPT_DIR = args.checkpoint or DEFAULT_CKPT[args.agent]


# ==========================================================
# AGENT LOADING (per-agent adapters → common interface)
# ==========================================================
def load_agent():
    """Return (agent, act_fn). act_fn(state, prev) -> action
    in env format [throttle(0..1), steer(-1..1), brake(0..1)].
    """
    if args.agent == "dqle":
        from agents.ql_diffusion import Diffusion_QL
        agent = Diffusion_QL(
            state_dim=OBS_DIM, action_dim=3,
            max_action=1.0, device=DEVICE,
        )
        agent.load_model(CKPT_DIR, id=int(args.ckpt_id))
        try:
            agent.actor.eval()
        except Exception:
            pass

        def act_fn(state, prev, alpha=0.7):
            raw = np.asarray(
                agent.sample_action(state), dtype=np.float32
            )
            sm = (alpha * raw + (1 - alpha) * prev)
            return _to_env_action(sm)

        logger.info(
            f"Loaded DQL-E from {CKPT_DIR} id={args.ckpt_id}"
        )
        return agent, act_fn

    if args.agent == "sac":
        from agents.sac import SAC
        agent = SAC(
            state_dim=OBS_DIM, action_dim=3,
            max_action=1.0, device=DEVICE,
        )
        agent.load_model(CKPT_DIR, id=int(args.ckpt_id))
        agent.actor.eval()

        def act_fn(state, prev, alpha=0.7):
            with torch.no_grad():
                st = torch.FloatTensor(
                    state.reshape(1, -1)
                ).to(DEVICE)
                action, _ = agent.actor.sample(st)
            raw = action.cpu().numpy().flatten().astype(
                np.float32
            )
            sm = (alpha * raw + (1 - alpha) * prev)
            return _to_env_action(sm)

        logger.info(
            f"Loaded SAC from {CKPT_DIR} id={args.ckpt_id}"
        )
        return agent, act_fn

    if args.agent == "ppo":
        from agents.ppo import PPO
        agent = PPO(
            state_dim=OBS_DIM, action_dim=3,
            max_action=1.0, device=DEVICE,
        )
        agent.load_model(CKPT_DIR, id=int(args.ckpt_id))
        try:
            agent.policy.eval()
        except Exception:
            pass

        def act_fn(state, prev, alpha=1.0):
            # PPO outputs continuous policy directly; use the
            # deterministic mean for evaluation. No smoothing
            # (alpha=1.0) to match how PPO was trained.
            raw = np.asarray(
                agent.get_action(state), dtype=np.float32
            )
            return _to_env_action(raw)

        logger.info(
            f"Loaded PPO from {CKPT_DIR} id={args.ckpt_id}"
        )
        return agent, act_fn

    raise ValueError(args.agent)


def _to_env_action(a):
    """Clamp a raw 3-vector to env action ranges."""
    return np.array([
        float(np.clip(a[0],  0.0, 1.0)),
        float(np.clip(a[1], -1.0, 1.0)),
        float(np.clip(a[2],  0.0, 1.0)),
    ], dtype=np.float32)


def _maybe_noise(state):
    """Add Gaussian noise to the observation for robustness
    experiments. Noise std is a fraction of the per-feature
    scale (obs is already roughly normalised in [-1,1]).
    """
    if args.noise <= 0.0:
        return state
    return (state + np.random.normal(
        0.0, args.noise, size=state.shape
    ).astype(np.float32))


# ==========================================================
# CSV
# ==========================================================
def save_csv(records, path):
    if not records:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    logger.info(f"Saved {len(records)} rows → {path}")


# ==========================================================
# RENDER (optional)
# ==========================================================
def init_render():
    if args.no_render:
        return None, None
    pygame.init()
    screen = pygame.display.set_mode(
        (args.width, args.height)
    )
    pygame.display.set_caption(
        f"{args.agent.upper()} — eval"
    )
    font = pygame.font.SysFont("monospace", 16)
    return screen, font


def render_frame(screen, font, env, args_traffic, route_key,
                 ep, episodes, step, ep_reward, action,
                 last_speed):
    if screen is None:
        return True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        return False
    frame = env.unwrapped.get_camera_frame()
    if frame is not None:
        try:
            import cv2
            if (frame.shape[1] != args.width or
                    frame.shape[0] != args.height):
                frame = cv2.resize(
                    frame, (args.width, args.height)
                )
        except Exception:
            pass
        surf = pygame.surfarray.make_surface(
            frame.swapaxes(0, 1)
        )
        screen.blit(surf, (0, 0))
    lines = [
        f"{args.agent.upper()} | {args_traffic.upper()}",
        f"Route: {route_key}",
        f"Ep: {ep}/{episodes}",
        f"Step: {step}",
        f"Reward: {ep_reward:.1f}",
        f"Speed: {last_speed:.1f} km/h",
        f"Thr: {action[0]:.2f} Str: {action[1]:.2f} "
        f"Brk: {action[2]:.2f}",
    ]
    bg = pygame.Surface(
        (260, len(lines) * 22 + 10), pygame.SRCALPHA
    )
    bg.fill((0, 0, 0, 160))
    screen.blit(bg, (8, 8))
    for i, line in enumerate(lines):
        txt = font.render(line, True, (255, 255, 255))
        screen.blit(txt, (14, 14 + i * 22))
    pygame.display.flip()
    pygame.time.Clock().tick(30)
    return True


# ==========================================================
# SINGLE ROUTE EVALUATION
# ==========================================================
def eval_route(env, act_fn, screen, font, route_key,
               spawn_idx, dest_idx):
    world = env.unwrapped.world
    cmap  = env.unwrapped.carla_map

    # Reference path for this route (ground-truth measurement)
    ref_xy, start_tf = ru.generate_reference_route(
        world, cmap, spawn_idx, dest_idx
    )
    logger.info(
        f"[{route_key}] reference path: {len(ref_xy)} wps "
        f"spawn={spawn_idx} dest={dest_idx}"
    )

    out_dir = f"results/{args.agent}_{args.traffic}"
    if args.noise > 0:
        out_dir += f"_noise{int(args.noise*100)}"
    os.makedirs(out_dir, exist_ok=True)

    summaries = []

    for ep in range(1, args.episodes + 1):
        # Force fixed spawn for this route. We set a hint the
        # env can use; if the env ignores it we still measure
        # against the reference and note the spawn.
        env.unwrapped._eval_spawn_transform = start_tf
        env.unwrapped._traffic_preset    = args.traffic
        env.unwrapped.cfg.TRAFFIC_LIGHTS = "off"

        state, _ = env.reset()
        if screen is not None:
            env.unwrapped.set_camera_view("third_person")

        prev_action = np.zeros(3, dtype=np.float32)
        ep_reward = 0.0
        step = 0
        records = []
        agent_xy = []
        term_reason = None
        last_speed = 0.0

        logger.info(
            f"[{route_key}] Episode {ep}/{args.episodes} "
            f"agent={args.agent} traffic={args.traffic} "
            f"noise={args.noise}"
        )

        while step < args.max_steps:
            obs_in = _maybe_noise(state)
            action = act_fn(obs_in, prev_action)
            next_state, reward, done, trunc, info = \
                env.step(action)

            if env.unwrapped.vehicle:
                loc = env.unwrapped.vehicle.get_location()
                vel = env.unwrapped.vehicle.get_velocity()
                tf  = env.unwrapped.vehicle.get_transform()
                spd = math.sqrt(
                    vel.x**2 + vel.y**2 + vel.z**2
                ) * 3.6
                last_speed = spd
                agent_xy.append([loc.x, loc.y])
                records.append({
                    "agent"    : args.agent,
                    "route"    : route_key,
                    "traffic"  : args.traffic,
                    "noise"    : args.noise,
                    "episode"  : ep,
                    "step"     : step,
                    "x"        : round(loc.x, 3),
                    "y"        : round(loc.y, 3),
                    "heading"  : round(tf.rotation.yaw, 2),
                    "speed_kmh": round(spd, 3),
                    "throttle" : round(float(action[0]), 4),
                    "steer"    : round(float(action[1]), 4),
                    "brake"    : round(float(action[2]), 4),
                    "reward"   : round(reward, 4),
                    "ep_reward": round(ep_reward, 2),
                    "collision": int(
                        info.get("collision_flag", False)
                    ),
                    "offroad"  : int(
                        info.get("offroad_flag", False)
                    ),
                    "term"     : "",
                })

            state = next_state
            prev_action = action
            ep_reward += reward
            step += 1

            ok = render_frame(
                screen, font, env, args.traffic, route_key,
                ep, args.episodes, step, ep_reward,
                action, last_speed,
            )
            if not ok:
                if env is not None:
                    env.close()
                if screen is not None:
                    pygame.quit()
                sys.exit(0)

            if done or trunc:
                term_reason = info.get("term_reason", "done")
                if records:
                    records[-1]["term"] = term_reason
                break

        if term_reason is None:
            term_reason = "max_steps"

        agent_xy = np.array(agent_xy, dtype=np.float32) \
            if agent_xy else np.zeros((0, 2), np.float32)

        # Metrics vs reference
        comp_frac, _ = ru.route_completion(agent_xy, ref_xy)
        mean_err, max_err = ru.trajectory_error(
            agent_xy, ref_xy
        )
        steer_series = [r["steer"]     for r in records]
        speed_series = [r["speed_kmh"] for r in records]
        head_series  = [r["heading"]   for r in records]

        # Save per-step CSV
        csv_path = os.path.join(
            out_dir, f"{route_key}_ep{ep:02d}.csv"
        )
        save_csv(records, csv_path)

        # success = reached end of reference path (high
        # completion) AND not terminated by collision/offroad
        reached = comp_frac >= 0.90
        success = int(
            reached and term_reason not in
            ("collision", "offroad", "wrong_way")
        )

        summary = {
            "agent"        : args.agent,
            "route"        : route_key,
            "traffic"      : args.traffic,
            "noise"        : args.noise,
            "episode"      : ep,
            "steps"        : step,
            "reward"       : round(ep_reward, 2),
            "term_reason"  : term_reason,
            "success"      : success,
            "route_completion": round(comp_frac, 4),
            "collision"    : int(term_reason == "collision"),
            "offroad"      : int(term_reason == "offroad"),
            "completion_flag": int(term_reason == "max_steps"),
            "mean_traj_err": round(mean_err, 3),
            "max_traj_err" : round(max_err, 3),
            "avg_speed"    : round(
                float(np.mean(speed_series))
                if speed_series else 0.0, 2),
            "steer_oscillation": round(
                ru.steering_oscillation(steer_series), 5),
            "yaw_oscillation"  : round(
                ru.yaw_oscillation(head_series), 4),
            "avg_jerk"     : round(
                ru.compute_jerk(speed_series), 4),
        }
        summaries.append(summary)
        logger.info(
            f"[{route_key}] Ep {ep} | reward={ep_reward:.1f} "
            f"| steps={step} | end={term_reason} | "
            f"completion={comp_frac*100:.1f}% | "
            f"traj_err={mean_err:.2f}m | success={success}"
        )
        time.sleep(0.5)

    # Save per-route summary
    sum_path = os.path.join(
        out_dir, f"{route_key}_summary.csv"
    )
    save_csv(summaries, sum_path)
    return summaries


# ==========================================================
# MAIN
# ==========================================================
def main():
    # Validate routes are filled in
    if args.route == "all":
        route_keys = list(ru.ROUTES.keys())
    else:
        route_keys = [args.route]

    for rk in route_keys:
        if rk not in ru.ROUTES:
            raise ValueError(
                f"Unknown route '{rk}'. "
                f"Available: {list(ru.ROUTES.keys())}"
            )
        if ru.ROUTES[rk]["spawn"] is None:
            raise RuntimeError(
                f"Route '{rk}' has no spawn/dest indices. "
                "Fill ROUTES in route_utils.py "
                "(run `python3 route_utils.py` with CARLA up)."
            )

    agent, act_fn = load_agent()
    screen, font = init_render()
    env = gym.make("RLCarla-v0")

    all_summaries = []
    try:
        for rk in route_keys:
            spawn_idx = ru.ROUTES[rk]["spawn"]
            dest_idx  = ru.ROUTES[rk]["dest"]
            s = eval_route(
                env, act_fn, screen, font, rk,
                spawn_idx, dest_idx,
            )
            all_summaries.extend(s)
    except Exception:
        traceback.print_exc()
    finally:
        # Combined summary across routes
        if all_summaries:
            combo_dir = f"results/{args.agent}_{args.traffic}"
            if args.noise > 0:
                combo_dir += f"_noise{int(args.noise*100)}"
            os.makedirs(combo_dir, exist_ok=True)
            save_csv(
                all_summaries,
                os.path.join(combo_dir, "ALL_summary.csv"),
            )
            _print_overview(all_summaries)
        env.close()
        if screen is not None:
            pygame.quit()


def _print_overview(summaries):
    print("\n" + "=" * 60)
    print(f"{args.agent.upper()} | traffic={args.traffic} | "
          f"noise={args.noise} | {len(summaries)} episodes")
    print("=" * 60)
    succ = np.mean([s["success"] for s in summaries]) * 100
    comp = np.mean(
        [s["route_completion"] for s in summaries]
    ) * 100
    coll = np.mean([s["collision"] for s in summaries]) * 100
    rew  = np.mean([s["reward"] for s in summaries])
    terr = np.mean([s["mean_traj_err"] for s in summaries])
    osc  = np.mean(
        [s["steer_oscillation"] for s in summaries]
    )
    jerk = np.mean([s["avg_jerk"] for s in summaries])
    print(f"  Success rate      : {succ:.1f}%")
    print(f"  Route completion  : {comp:.1f}%")
    print(f"  Collision rate    : {coll:.1f}%")
    print(f"  Avg reward        : {rew:.1f}")
    print(f"  Mean traj error   : {terr:.2f} m")
    print(f"  Steer oscillation : {osc:.4f}")
    print(f"  Avg jerk          : {jerk:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()