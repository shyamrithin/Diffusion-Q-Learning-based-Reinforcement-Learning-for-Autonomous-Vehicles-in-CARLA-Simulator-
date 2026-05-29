import sys
sys.path.insert(0, "/home/shyam/Carla/PythonAPI/carla")
# ==========================================================
# eval_engine.py
# RLCarla Evaluation Engine
#
# Core evaluation logic shared by all three agents.
# Handles:
#   - Fixed spawn + destination in Town03
#   - CARLA global route planner (A* pathfinding)
#   - Per-step data recording
#   - CSV export
#   - 3 runs per scenario, averaged
#
# Scenarios:
#   light  : 10 vehicles,  5 walkers
#   medium : 30 vehicles, 15 walkers
#   heavy  : 60 vehicles, 30 walkers
#
# CARLA 0.9.16 | Python 3.10
# ==========================================================

import os
import csv
import math
import time
import logging
import numpy as np
import carla

logger = logging.getLogger(__name__)

# ==========================================================
# ==========================================================
# FIXED ROUTE — Town03 Roundabout
# Route: West approach → Central roundabout → East exit
# Tests: straight road, roundabout entry, navigation, exit
# Distance: ~150m
# ==========================================================
SPAWN_POINT = {
    "x"   : -118.1,
    "y"   :   0.3,
    "z"   :   0.5,
    "yaw" :   0.3,
}

DESTINATION = {
    "x" :  35.5,
    "y" :  -7.8,
}

TRAFFIC_PRESETS = {
    "light"  : {"vehicles": 10,  "walkers": 0},
    "medium" : {"vehicles": 30,  "walkers": 0},
    "heavy"  : {"vehicles": 60,  "walkers": 0},
}

# Noise sigmas for Part 3
NOISE_SIGMAS = [0.0, 0.01, 0.05, 0.1]

# ==========================================================
# ROUTE PLANNER
# ==========================================================
def get_route_waypoints(world, carla_map,
                        start_loc, end_loc,
                        sampling_resolution=2.0):
    """
    Use CARLA global route planner to get
    ground truth waypoints from start to end.
    Returns list of (x, y) tuples.
    """
    try:
        import sys
        sys.path.append("/home/shyam/Carla/PythonAPI/carla")
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        grp = GlobalRoutePlanner(
            carla_map, sampling_resolution
        )
        route = grp.trace_route(start_loc, end_loc)
        waypoints = [
            (wp.transform.location.x,
             wp.transform.location.y)
            for wp, _ in route
        ]
        logger.info(
            f"[Route] {len(waypoints)} waypoints planned"
        )
        return waypoints
    except Exception as e:
        logger.error(f"[Route] Planner failed: {e}")
        # Fallback: linear interpolation
        waypoints = []
        steps = 100
        for i in range(steps + 1):
            t = i / steps
            x = start_loc.x + t * (
                end_loc.x - start_loc.x
            )
            y = start_loc.y + t * (
                end_loc.y - start_loc.y
            )
            waypoints.append((x, y))
        return waypoints


def get_nearest_waypoint(waypoints, pos_x, pos_y):
    """
    Find nearest ground truth waypoint to
    current vehicle position.
    Returns (gt_x, gt_y, deviation_metres).
    """
    min_dist = float("inf")
    nearest  = waypoints[0]

    for wp in waypoints:
        d = math.sqrt(
            (wp[0] - pos_x)**2 +
            (wp[1] - pos_y)**2
        )
        if d < min_dist:
            min_dist = d
            nearest  = wp

    return nearest[0], nearest[1], min_dist


def check_route_complete(pos_x, pos_y,
                         dest_x, dest_y,
                         threshold=8.0):
    """Check if vehicle reached destination."""
    d = math.sqrt(
        (pos_x - dest_x)**2 +
        (pos_y - dest_y)**2
    )
    return d < threshold


# ==========================================================
# TRAFFIC SPAWNER
# ==========================================================
def spawn_traffic(client, world, carla_map,
                  n_vehicles, n_walkers,
                  ego_location, safe_radius=15.0,
                  seed=42):
    """
    Spawn NPC vehicles and walkers.
    Keeps safe_radius around ego spawn clear.
    Uses fixed seed for reproducibility.
    Returns list of spawned actor IDs.
    """
    import random
    random.seed(seed)

    bp_lib      = world.get_blueprint_library()
    spawn_pts   = carla_map.get_spawn_points()
    random.shuffle(spawn_pts)

    traffic_mgr = client.get_trafficmanager(8000)
    traffic_mgr.set_global_distance_to_leading_vehicle(
        2.0
    )
    traffic_mgr.set_synchronous_mode(True)
    traffic_mgr.set_random_device_seed(seed)

    actor_ids = []

    # Spawn vehicles
    vehicle_bps = bp_lib.filter("vehicle.*")
    spawned     = 0
    for sp in spawn_pts:
        if spawned >= n_vehicles:
            break
        dist = math.sqrt(
            (sp.location.x - ego_location.x)**2 +
            (sp.location.y - ego_location.y)**2
        )
        if dist < safe_radius:
            continue
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute(
                "color", "255,0,0"
            )
        actor = world.try_spawn_actor(bp, sp)
        if actor:
            actor.set_autopilot(True, 8000)
            actor_ids.append(actor.id)
            spawned += 1

    logger.info(
        f"[Traffic] Spawned {spawned}/{n_vehicles}"
        f" vehicles"
    )

    # Spawn walkers (simplified)
    walker_bp  = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find(
        "controller.ai.walker"
    )
    w_spawned  = 0

    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        bp = random.choice(walker_bp)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        tf     = carla.Transform(loc)
        walker = world.try_spawn_actor(bp, tf)
        if walker:
            ctrl = world.spawn_actor(
                controller_bp,
                carla.Transform(),
                attach_to=walker
            )
            ctrl.start()
            ctrl.go_to_location(
                world.get_random_location_from_navigation()
            )
            actor_ids.append(walker.id)
            actor_ids.append(ctrl.id)
            w_spawned += 1

    logger.info(
        f"[Traffic] Spawned {w_spawned}/{n_walkers}"
        f" walkers"
    )

    return actor_ids


def destroy_traffic(client, actor_ids):
    """Destroy all spawned NPCs."""
    if actor_ids:
        client.apply_batch([
            carla.command.DestroyActor(aid)
            for aid in actor_ids
        ])
    logger.info("[Traffic] All NPCs destroyed")


# ==========================================================
# SLOW TRAFFIC SCENARIO — Part 2
# ==========================================================
def spawn_slow_surrounding_vehicles(
    world, carla_map, ego_vehicle,
    n_vehicles=4, target_speed_kmh=20.0,
    seed=42
):
    """
    Spawn vehicles closely surrounding ego vehicle.
    Set them to crawl at target_speed_kmh.
    Used for Part 2 special case evaluation.

    Positions:
      Front  : 10m ahead
      Rear   : 10m behind
      Left   : 4m left
      Right  : 4m right
    """
    import random
    random.seed(seed)

    bp_lib  = world.get_blueprint_library()
    ego_tf  = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = math.radians(ego_tf.rotation.yaw)

    # Forward/right unit vectors
    fwd_x   = math.cos(ego_yaw)
    fwd_y   = math.sin(ego_yaw)
    right_x = math.cos(ego_yaw - math.pi/2)
    right_y = math.sin(ego_yaw - math.pi/2)

    offsets = [
        ( 12.0,   0.0, "front"),
        (-12.0,   0.0, "rear"),
        (  0.0,   3.8, "right"),
        (  0.0,  -3.8, "left"),
    ]

    client      = world.get_snapshot().timestamp
    traffic_mgr = world.get_actors().filter(
        "traffic.traffic_manager*"
    )

    actor_ids = []
    vehicle_bps = bp_lib.filter("vehicle.*")

    # Use traffic manager for slow speed
    tm = carla.Client("localhost", 2000)
    tm = tm.get_trafficmanager(8000)

    for fwd_offset, right_offset, label in offsets:
        spawn_x = (ego_loc.x +
                   fwd_x * fwd_offset +
                   right_x * right_offset)
        spawn_y = (ego_loc.y +
                   fwd_y * fwd_offset +
                   right_y * right_offset)
        spawn_z = ego_loc.z + 0.3

        sp = carla.Transform(
            carla.Location(
                x=spawn_x, y=spawn_y, z=spawn_z
            ),
            carla.Rotation(yaw=ego_tf.rotation.yaw)
        )

        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", "255,165,0")

        actor = world.try_spawn_actor(bp, sp)
        if actor:
            actor.set_autopilot(True, 8000)
            # Set speed limit to target_speed_kmh
            # percentage reduction from speed limit
            tm.vehicle_percentage_speed_difference(
                actor,
                100 - (target_speed_kmh / 50.0 * 100)
            )
            actor_ids.append(actor.id)
            logger.info(
                f"[Slow] Spawned {label} vehicle"
            )

    return actor_ids


# ==========================================================
# SENSOR NOISE — Part 3
# ==========================================================
def add_sensor_noise(obs, sigma,
                     lidar_dims=(9, 81),
                     ego_dims=(0, 9)):
    """
    Add Gaussian noise to specific observation dims.

    Noisy components:
      LiDAR dims 9-80 (72D polar histogram × 1 frame)
      Ego state dims 0-8 (speed, heading, etc.)

    Args:
      obs       : full observation vector (564D)
      sigma     : noise standard deviation
      lidar_dims: (start, end) indices for LiDAR
      ego_dims  : (start, end) indices for ego state

    Returns:
      Noisy observation vector
    """
    if sigma <= 0:
        return obs

    noisy_obs = obs.copy()

    # LiDAR noise — simulates sensor degradation
    lidar_start, lidar_end = lidar_dims
    noisy_obs[lidar_start:lidar_end] += (
        np.random.normal(0, sigma,
                         lidar_end - lidar_start)
    )
    noisy_obs[lidar_start:lidar_end] = np.clip(
        noisy_obs[lidar_start:lidar_end], 0.0, 1.0
    )

    # Ego state noise — simulates IMU/GPS drift
    ego_start, ego_end = ego_dims
    noisy_obs[ego_start:ego_end] += (
        np.random.normal(0, sigma,
                         ego_end - ego_start)
    )

    return noisy_obs


# ==========================================================
# STEP DATA RECORDER
# ==========================================================
class StepRecorder:
    """Records per-step evaluation data."""

    def __init__(self):
        self.records = []

    def record(self, step, vehicle,
               gt_x, gt_y, wp_deviation,
               action, reward, info):
        """Record one step of evaluation data."""
        loc = vehicle.get_location()
        vel = vehicle.get_velocity()
        tf  = vehicle.get_transform()

        speed = math.sqrt(
            vel.x**2 + vel.y**2 + vel.z**2
        ) * 3.6  # m/s → km/h

        self.records.append({
            "step"           : step,
            "x"              : round(loc.x, 3),
            "y"              : round(loc.y, 3),
            "z"              : round(loc.z, 3),
            "heading"        : round(
                tf.rotation.yaw, 3
            ),
            "speed_kmh"      : round(speed, 3),
            "gt_x"           : round(gt_x, 3),
            "gt_y"           : round(gt_y, 3),
            "wp_deviation_m" : round(wp_deviation, 3),
            "throttle"       : round(
                float(action[0]), 4
            ),
            "steer"          : round(
                float(action[1]), 4
            ),
            "brake"          : round(
                float(action[2]), 4
            ),
            "reward"         : round(reward, 4),
            "collision"      : int(
                info.get("collision_flag", False)
            ),
            "offroad"        : int(
                info.get("offroad_flag", False)
            ),
            "term_reason"    : info.get(
                "term_reason", ""
            ),
        })

    def save_csv(self, filepath):
        """Save all records to CSV file."""
        if not self.records:
            logger.warning("[Recorder] No data to save")
            return

        os.makedirs(
            os.path.dirname(filepath), exist_ok=True
        )
        fieldnames = list(self.records[0].keys())

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(self.records)

        logger.info(
            f"[Recorder] Saved {len(self.records)}"
            f" steps to {filepath}"
        )

    def get_summary(self):
        """Compute summary statistics."""
        if not self.records:
            return {}

        wp_devs   = [
            r["wp_deviation_m"] for r in self.records
        ]
        speeds    = [
            r["speed_kmh"] for r in self.records
        ]
        throttles = [
            r["throttle"] for r in self.records
        ]
        steers    = [
            abs(r["steer"]) for r in self.records
        ]
        brakes    = [
            r["brake"] for r in self.records
        ]

        collisions = sum(
            r["collision"] for r in self.records
        )
        offroads   = sum(
            r["offroad"] for r in self.records
        )

        term = self.records[-1]["term_reason"] \
            if self.records else ""

        return {
            "total_steps"       : len(self.records),
            "route_complete"    : int(
                term == "route_complete"
            ),
            "collision_events"  : collisions,
            "offroad_steps"     : offroads,
            "avg_wp_deviation_m": round(
                np.mean(wp_devs), 3
            ),
            "max_wp_deviation_m": round(
                np.max(wp_devs), 3
            ),
            "avg_speed_kmh"     : round(
                np.mean(speeds), 3
            ),
            "avg_throttle"      : round(
                np.mean(throttles), 3
            ),
            "avg_abs_steer"     : round(
                np.mean(steers), 3
            ),
            "avg_brake"         : round(
                np.mean(brakes), 3
            ),
        }