import sys
sys.path.insert(0, "/home/shyam/Carla/PythonAPI/carla")
# ==========================================================
# run_eval_dql.py
# DQL-E Evaluation Runner
#
# Runs DQL-E agent on fixed Town03 route
# across all scenarios and noise levels.
# Saves per-step CSVs and summary JSON.
#
# Usage:
#   python3 run_eval_dql.py
#   python3 run_eval_dql.py --scenario heavy
#   python3 run_eval_dql.py --noise
#   python3 run_eval_dql.py --slow
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

from agents.ql_diffusion      import Diffusion_QL
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
logger = logging.getLogger("eval_dql")

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR     = "eval_results/dql"
N_RUNS         = 3
MAX_STEPS      = 2000
DEVICE         = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_agent():
    """Load best DQL-E checkpoint."""
    agent = Diffusion_QL(
        state_dim          = OBS_DIM,
        action_dim         = 3,
        max_action         = 1.0,
        device             = DEVICE,
        discount           = 0.99,
        tau                = 0.005,
        eta                = 0.001,
        beta_schedule      = "vp",
        n_timesteps        = 3,
        lr                 = 3e-5,
        grad_norm          = 0.1,
        action_temperature = 0.1,
        alpha              = 0.2,
        auto_alpha         = True,
    )
    actor_path = os.path.join(
        CHECKPOINT_DIR, "actor_9999.pth"
    )
    if not os.path.exists(actor_path):
        raise FileNotFoundError(
            f"DQL checkpoint not found: {actor_path}"
        )
    agent.load_model(CHECKPOINT_DIR, id=9999)
    agent.actor.eval()
    agent.critic.eval()
    logger.info("DQL-E model loaded (actor_9999.pth)")
    return agent


# Action smoothing state
_prev_action = np.zeros(3, dtype=np.float32)

def get_action(agent, obs):
    """Get deterministic action from DQL-E."""
    global _prev_action
    with torch.no_grad():
        state_t   = torch.FloatTensor(
            obs.reshape(1, -1)
        ).to(DEVICE)
        state_rpt = torch.repeat_interleave(
            state_t, repeats=50, dim=0
        )
        actions   = agent.actor.sample(state_rpt)
        q_values  = agent.critic_target.q_min(
            state_rpt, actions
        ).flatten()
        idx       = q_values.argmax()
    raw = actions[idx].cpu().numpy().flatten()

    # Action smoothing — same as training
    ACTION_ALPHA = 0.7
    raw = (ACTION_ALPHA * raw +
           (1 - ACTION_ALPHA) * _prev_action)
    _prev_action = raw.copy()

    # Clip to valid ranges
    action = np.zeros(3, dtype=np.float32)
    action[0] = float(np.clip(raw[0],  0.0, 1.0))
    action[1] = float(np.clip(raw[1], -1.0, 1.0))
    action[2] = float(np.clip(raw[2],  0.0, 1.0))

    # Force minimum throttle if barely moving
    if action[2] > 0.5:
        action[2] = 0.0
    if action[0] < 0.3:
        action[0] = 0.3

    return action


def run_episode(env, agent, world, carla_map,
                route_wps, scenario,
                noise_sigma=0.0,
                slow_mode=False,
                run_id=0):
    """
    Run one evaluation episode.
    Returns StepRecorder with all data.
    """
    recorder = StepRecorder()

    # Spawn NPC traffic
    preset = TRAFFIC_PRESETS[scenario]
    ego_loc = carla.Location(
        x=SPAWN_POINT["x"],
        y=SPAWN_POINT["y"],
        z=SPAWN_POINT["z"],
    )

    traffic_ids = spawn_traffic(
        env.unwrapped.client,
        world, carla_map,
        n_vehicles  = preset["vehicles"],
        n_walkers   = preset["walkers"],
        ego_location= ego_loc,
        seed        = 42 + run_id,
    )

    # Reset env to fixed spawn
    obs, _ = env.reset()
    vehicle = env.unwrapped.vehicle

    # Spawn slow surrounding vehicles if Part 2
    slow_ids = []
    if slow_mode:
        slow_ids = spawn_slow_surrounding_vehicles(
            world, carla_map, vehicle,
            n_vehicles      = 4,
            target_speed_kmh= 20.0,
            seed            = 42 + run_id,
        )
        # Let NPCs settle
        for _ in range(30):
            world.tick()

    dest_loc = carla.Location(
        x=DESTINATION["x"],
        y=DESTINATION["y"],
        z=0.0,
    )

    ep_reward = 0.0
    reached   = False

    for step in range(MAX_STEPS):

        # Add sensor noise if Part 3
        noisy_obs = add_sensor_noise(
            obs, noise_sigma
        ) if noise_sigma > 0 else obs

        # Get action
        action = get_action(agent, noisy_obs)

        # Step environment
        next_obs, reward, done, trunc, info = \
            env.step(action)

        # Get ground truth waypoint
        loc   = vehicle.get_location()
        gt_x, gt_y, deviation = get_nearest_waypoint(
            route_wps, loc.x, loc.y
        )

        # Record step
        recorder.record(
            step, vehicle,
            gt_x, gt_y, deviation,
            action, reward, info
        )

        obs        = next_obs
        ep_reward += reward

        # Check destination reached
        if check_route_complete(
            loc.x, loc.y,
            DESTINATION["x"], DESTINATION["y"]
        ):
            logger.info(
                f"[Run {run_id}] Route complete "
                f"at step {step}!"
            )
            reached = True
            # Mark last record
            recorder.records[-1]["term_reason"] = \
                "route_complete"
            break

        if done or trunc:
            logger.info(
                f"[Run {run_id}] Episode ended: "
                f"{info.get('term_reason','?')} "
                f"at step {step}"
            )
            break

    # Cleanup
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
    """
    Run N_RUNS episodes for one scenario.
    Save CSV per run + summary JSON.
    """
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
            f"DQL-E | Scenario: {tag} | "
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

        # Save per-run CSV
        csv_path = os.path.join(
            output_dir,
            f"{tag}_run{run_id+1}.csv"
        )
        recorder.save_csv(csv_path)

        summary = recorder.get_summary()
        summary["run_id"]   = run_id + 1
        summary["scenario"] = tag
        summaries.append(summary)

        logger.info(
            f"[Summary] "
            f"steps={summary['total_steps']} "
            f"wp_dev={summary['avg_wp_deviation_m']}m "
            f"collisions={summary['collision_events']}"
        )

        # Brief pause between runs
        time.sleep(2.0)

    # Save averaged summary
    avg_summary = {
        "scenario"          : tag,
        "algorithm"         : "DQL-E",
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
        f"\nDQL-E {tag} Summary:\n"
        f"  Route Complete : "
        f"{avg_summary['route_complete_pct']}%\n"
        f"  Avg WP Dev     : "
        f"{avg_summary['avg_wp_deviation_m']}m\n"
        f"  Collisions     : "
        f"{avg_summary['total_collisions']}\n"
        f"  Avg Speed      : "
        f"{avg_summary['avg_speed_kmh']} km/h"
    )

    return avg_summary


def main(args):
    """Main evaluation runner."""

    pygame.init()
    screen = pygame.display.set_mode((800, 450))
    pygame.display.set_caption("DQL-E Evaluation")
    clock  = pygame.time.Clock()

    env = gym.make("RLCarla-v0")

    # Force fixed spawn at roundabout approach
    import types, carla as _carla
    def _fixed_spawn(self):
        bp_lib = self.world.get_blueprint_library()
        bp     = self.world.get_blueprint_library().filter(
            "vehicle.tesla.model3"
        )[0]
        if bp.has_attribute("color"):
            bp.set_attribute("color", "0,120,255")
        sp = _carla.Transform(
            _carla.Location(x=-118.1, y=0.3, z=0.5),
            _carla.Rotation(yaw=0.3)
        )
        v = self.world.try_spawn_actor(bp, sp)
        if v:
            self.vehicle = v
            self.actor_list.append(v)
            return True
        return False
    env.unwrapped._spawn_vehicle = types.MethodType(
        _fixed_spawn, env.unwrapped
    )

    # Force Town03
    env.unwrapped.cfg.MAPS = ["Town03"]
    obs, _ = env.reset()

    world     = env.unwrapped.world
    carla_map = env.unwrapped.carla_map

    # Plan route once — shared by all runs
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

    agent      = load_agent()
    output_dir = OUTPUT_DIR
    all_results= {}

    if args.slow:
        # Part 2 — slow traffic special case
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
        # Part 3 — noise robustness
        for sigma in NOISE_SIGMAS:
            tag = f"noise_{sigma}"
            result = run_scenario(
                agent, env, world, carla_map,
                route_wps, "medium",
                noise_sigma = sigma,
                output_dir  = os.path.join(
                    output_dir, "noise"
                ),
            )
            all_results[tag] = result

    else:
        # Part 1 — traffic scenarios
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

    # Save combined results
    combined_path = os.path.join(
        output_dir, "all_results.json"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(
        f"\nAll DQL-E results saved to {output_dir}"
    )

    env.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DQL-E Evaluation"
    )
    parser.add_argument(
        "--scenario",
        type    = str,
        default = "all",
        choices = ["light", "medium", "heavy", "all"],
    )
    parser.add_argument(
        "--noise",
        action = "store_true",
        help   = "Run noise robustness evaluation",
    )
    parser.add_argument(
        "--slow",
        action = "store_true",
        help   = "Run slow traffic scenario",
    )
    args = parser.parse_args()
    main(args)