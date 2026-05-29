import sys
sys.path.insert(0, "/home/shyam/Carla/PythonAPI/carla")
# ==========================================================
# run_eval_sac.py
# SAC Evaluation Runner
#
# Identical evaluation logic to DQL-E.
# Only difference: loads SAC agent.
#
# Usage:
#   python3 run_eval_sac.py
#   python3 run_eval_sac.py --scenario heavy
#   python3 run_eval_sac.py --noise
#   python3 run_eval_sac.py --slow
#
# CARLA 0.9.16 | Python 3.10
# ==========================================================

import os
import sys
import json
import time
import math
import logging
import argparse
import numpy as np
import carla
import gymnasium as gym
import rlcarla
import torch
import pygame

from agents.sac               import SAC
from rlcarla.core.obs_builder import OBS_DIM
from eval_engine import (
    SPAWN_POINT, DESTINATION, TRAFFIC_PRESETS,
    NOISE_SIGMAS, get_route_waypoints,
    get_nearest_waypoint, check_route_complete,
    spawn_traffic, destroy_traffic,
    spawn_slow_surrounding_vehicles,
    add_sensor_noise, StepRecorder,
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("eval_sac")

CHECKPOINT_DIR = "checkpoints_sac"
OUTPUT_DIR     = "eval_results/sac"
N_RUNS         = 3
MAX_STEPS      = 2000
DEVICE         = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_agent():
    """Load best SAC checkpoint."""
    agent = SAC(
        state_dim  = OBS_DIM,
        action_dim = 3,
        max_action = 1.0,
        device     = DEVICE,
    )
    actor_path = os.path.join(
        CHECKPOINT_DIR, "sac_actor_9999.pth"
    )
    if not os.path.exists(actor_path):
        raise FileNotFoundError(
            f"SAC checkpoint not found: {actor_path}"
        )
    agent.load_model(CHECKPOINT_DIR, id=9999)
    agent.actor.eval()
    logger.info("SAC model loaded (sac_actor_9999.pth)")
    return agent


def get_action(agent, obs):
    """Get deterministic action from SAC."""
    with torch.no_grad():
        state_t = torch.FloatTensor(
            obs.reshape(1, -1)
        ).to(DEVICE)
        action, _, _ = agent.actor.sample(state_t)
    action = action.cpu().numpy().flatten()
    action[0] = float(np.clip(action[0],  0.0, 1.0))
    action[1] = float(np.clip(action[1], -1.0, 1.0))
    action[2] = float(np.clip(action[2],  0.0, 1.0))
    return action


def run_episode(env, agent, world, carla_map,
                route_wps, scenario,
                noise_sigma=0.0,
                slow_mode=False,
                run_id=0):
    """Run one SAC evaluation episode."""
    recorder = StepRecorder()

    preset  = TRAFFIC_PRESETS[scenario]
    ego_loc = carla.Location(
        x=SPAWN_POINT["x"],
        y=SPAWN_POINT["y"],
        z=SPAWN_POINT["z"],
    )

    traffic_ids = spawn_traffic(
        env.unwrapped.client,
        world, carla_map,
        n_vehicles   = preset["vehicles"],
        n_walkers    = preset["walkers"],
        ego_location = ego_loc,
        seed         = 42 + run_id,
    )

    obs, _ = env.reset()
    vehicle = env.unwrapped.vehicle

    slow_ids = []
    if slow_mode:
        slow_ids = spawn_slow_surrounding_vehicles(
            world, carla_map, vehicle,
            n_vehicles      = 4,
            target_speed_kmh= 20.0,
            seed            = 42 + run_id,
        )
        for _ in range(30):
            world.tick()

    ep_reward = 0.0
    reached   = False

    for step in range(MAX_STEPS):

        noisy_obs = add_sensor_noise(
            obs, noise_sigma
        ) if noise_sigma > 0 else obs

        action = get_action(agent, noisy_obs)

        next_obs, reward, done, trunc, info = \
            env.step(action)

        loc = vehicle.get_location()
        gt_x, gt_y, deviation = get_nearest_waypoint(
            route_wps, loc.x, loc.y
        )

        recorder.record(
            step, vehicle,
            gt_x, gt_y, deviation,
            action, reward, info
        )

        obs        = next_obs
        ep_reward += reward

        if check_route_complete(
            loc.x, loc.y,
            DESTINATION["x"], DESTINATION["y"]
        ):
            logger.info(
                f"[Run {run_id}] Route complete "
                f"at step {step}!"
            )
            reached = True
            recorder.records[-1]["term_reason"] = \
                "route_complete"
            break

        if done or trunc:
            break

    destroy_traffic(
        env.unwrapped.client,
        traffic_ids + slow_ids
    )

    logger.info(
        f"[Run {run_id}] reward={ep_reward:.1f} "
        f"steps={step+1} reached={reached}"
    )

    return recorder


def run_scenario(agent, env, world, carla_map,
                 route_wps, scenario,
                 noise_sigma=0.0,
                 slow_mode=False,
                 output_dir=None):
    """Run N_RUNS for one scenario."""
    summaries = []

    tag = scenario
    if noise_sigma > 0:
        tag += f"_noise{noise_sigma}"
    if slow_mode:
        tag = "slow_traffic"

    os.makedirs(output_dir, exist_ok=True)

    for run_id in range(N_RUNS):
        logger.info(
            f"\n{'='*50}\n"
            f"SAC | Scenario: {tag} | "
            f"Run {run_id+1}/{N_RUNS}\n"
            f"{'='*50}"
        )

        recorder = run_episode(
            env, agent, world, carla_map,
            route_wps, scenario,
            noise_sigma = noise_sigma,
            slow_mode   = slow_mode,
            run_id      = run_id,
        )

        csv_path = os.path.join(
            output_dir,
            f"{tag}_run{run_id+1}.csv"
        )
        recorder.save_csv(csv_path)

        summary = recorder.get_summary()
        summary["run_id"]   = run_id + 1
        summary["scenario"] = tag
        summaries.append(summary)

        time.sleep(2.0)

    avg_summary = {
        "scenario"          : tag,
        "algorithm"         : "SAC",
        "n_runs"            : N_RUNS,
        "route_complete_pct": round(
            np.mean([
                s["route_complete"] * 100
                for s in summaries
            ]), 1
        ),
        "avg_wp_deviation_m": round(
            np.mean([
                s["avg_wp_deviation_m"]
                for s in summaries
            ]), 3
        ),
        "avg_speed_kmh"     : round(
            np.mean([
                s["avg_speed_kmh"]
                for s in summaries
            ]), 2
        ),
        "total_collisions"  : int(np.sum([
            s["collision_events"]
            for s in summaries
        ])),
        "avg_throttle"      : round(
            np.mean([
                s["avg_throttle"]
                for s in summaries
            ]), 3
        ),
        "avg_abs_steer"     : round(
            np.mean([
                s["avg_abs_steer"]
                for s in summaries
            ]), 3
        ),
        "avg_brake"         : round(
            np.mean([
                s["avg_brake"]
                for s in summaries
            ]), 3
        ),
        "runs"              : summaries,
    }

    summary_path = os.path.join(
        output_dir, f"{tag}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(avg_summary, f, indent=2)

    logger.info(
        f"\nSAC {tag} Summary:\n"
        f"  Route Complete : "
        f"{avg_summary['route_complete_pct']}%\n"
        f"  Avg WP Dev     : "
        f"{avg_summary['avg_wp_deviation_m']}m\n"
        f"  Collisions     : "
        f"{avg_summary['total_collisions']}"
    )

    return avg_summary


def main(args):
    pygame.init()
    screen = pygame.display.set_mode((800, 450))
    pygame.display.set_caption("SAC Evaluation")
    clock  = pygame.time.Clock()

    env = gym.make("RLCarla-v0")
    env.unwrapped.cfg.MAPS = ["Town03"]
    obs, _ = env.reset()

    world     = env.unwrapped.world
    carla_map = env.unwrapped.carla_map

    start_loc = carla.Location(
        x=SPAWN_POINT["x"],
        y=SPAWN_POINT["y"],
        z=SPAWN_POINT["z"],
    )
    dest_loc = carla.Location(
        x=DESTINATION["x"],
        y=DESTINATION["y"],
        z=0.0,
    )
    route_wps = get_route_waypoints(
        world, carla_map, start_loc, dest_loc
    )

    agent       = load_agent()
    output_dir  = OUTPUT_DIR
    all_results = {}

    if args.slow:
        result = run_scenario(
            agent, env, world, carla_map,
            route_wps, "heavy",
            slow_mode  = True,
            output_dir = os.path.join(
                output_dir, "slow"
            ),
        )
        all_results["slow"] = result

    elif args.noise:
        for sigma in NOISE_SIGMAS:
            result = run_scenario(
                agent, env, world, carla_map,
                route_wps, "medium",
                noise_sigma = sigma,
                output_dir  = os.path.join(
                    output_dir, "noise"
                ),
            )
            all_results[f"noise_{sigma}"] = result

    else:
        scenarios = (
            [args.scenario]
            if args.scenario != "all"
            else ["light", "medium", "heavy"]
        )
        for scenario in scenarios:
            result = run_scenario(
                agent, env, world, carla_map,
                route_wps, scenario,
                output_dir = os.path.join(
                    output_dir, scenario
                ),
            )
            all_results[scenario] = result

    combined_path = os.path.join(
        output_dir, "all_results.json"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAC Evaluation"
    )
    parser.add_argument(
        "--scenario", type=str, default="all",
        choices=["light","medium","heavy","all"]
    )
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--slow",  action="store_true")
    args = parser.parse_args()
    main(args)