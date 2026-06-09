# ==========================================================
# route_utils.py
# Fixed-route evaluation utilities for RLCarla
#
# Purpose:
#   Provides a COMMON, reproducible set of evaluation routes
#   so that DQL-E, SAC, and PPO are all compared on identical
#   start points and reference paths. Training was open-loop
#   exploration (no goal); for evaluation we attach a
#   GlobalRoutePlanner reference path between a fixed spawn
#   and destination. The agent still drives open-loop (it has
#   no route in its observation) — the reference path is used
#   ONLY for measurement:
#     * route_completion: how far along the reference the
#       agent travelled before terminating
#     * trajectory_error: mean distance from the agent's
#       path to the nearest reference waypoint (lane/path
#       deviation)
#
# IMPORTANT (honest framing for the paper):
#   Agents are NOT goal-conditioned. The planned route is an
#   evaluation reference, not a navigation target. Report it
#   that way.
#
# CARLA 0.9.16 | Python 3.10
# ==========================================================

import os
import sys
import math
import numpy as np

# ----------------------------------------------------------
# Make CARLA's GlobalRoutePlanner importable.
# Found on this machine at:
#   /home/shyam/Carla/PythonAPI/carla/agents/navigation/
# We add the PythonAPI/carla dir to sys.path so that
# `from agents.navigation...` resolves.
# ----------------------------------------------------------
_CARLA_AGENTS_PATHS = [
    "/home/shyam/Carla/PythonAPI/carla",
    "/home/shyam/carla/PythonAPI/carla",
    os.path.expanduser("~/Carla/PythonAPI/carla"),
    os.path.expanduser("~/carla/PythonAPI/carla"),
]
for _p in _CARLA_AGENTS_PATHS:
    if os.path.isdir(os.path.join(_p, "agents", "navigation")):
        if _p not in sys.path:
            sys.path.append(_p)
        break

try:
    from agents.navigation.global_route_planner import (
        GlobalRoutePlanner,
    )
    _GRP_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _GRP_AVAILABLE = False
    _GRP_IMPORT_ERROR = _e


# ==========================================================
# ROUTE DEFINITIONS
# ==========================================================
# (spawn_index, destination_index) into
# world.get_map().get_spawn_points().
#
# >>> FILL THESE IN once CARLA is running. Use
#     visualize_spawn_points() below to pick indices that
#     start near / pass through a roundabout and an
#     intersection on Town03. <<<
#
# Keep the SAME three routes for ALL agents.
ROUTES = {
    "route_1_roundabout":   {"spawn": None, "dest": None},
    "route_2_intersection": {"spawn": None, "dest": None},
    "route_3_straight":     {"spawn": None, "dest": None},
}

# Sampling resolution (metres) for the reference path.
ROUTE_SAMPLE_RES = 2.0


# ==========================================================
# REFERENCE ROUTE GENERATION
# ==========================================================
def generate_reference_route(world, carla_map,
                             spawn_idx, dest_idx,
                             sample_res=ROUTE_SAMPLE_RES):
    """Generate a reference waypoint path between two spawn
    points using CARLA's GlobalRoutePlanner.

    Returns:
        ref_xy : (N, 2) numpy array of (x, y) reference points
        start_transform : carla.Transform for spawning the ego
    """
    if not _GRP_AVAILABLE:
        raise RuntimeError(
            "GlobalRoutePlanner not importable. "
            f"Last error: {_GRP_IMPORT_ERROR}"
        )

    spawn_points = carla_map.get_spawn_points()
    start_tf = spawn_points[spawn_idx]
    end_tf   = spawn_points[dest_idx]

    grp = GlobalRoutePlanner(carla_map, sample_res)
    route = grp.trace_route(
        start_tf.location, end_tf.location
    )
    # route is a list of (waypoint, road_option) tuples
    ref_xy = np.array(
        [
            [wp.transform.location.x,
             wp.transform.location.y]
            for wp, _ in route
        ],
        dtype=np.float32,
    )
    return ref_xy, start_tf


# ==========================================================
# METRICS
# ==========================================================
def _cumulative_arc_length(ref_xy):
    """Cumulative distance along the reference path."""
    if len(ref_xy) < 2:
        return np.zeros(len(ref_xy), dtype=np.float32)
    seg = np.linalg.norm(np.diff(ref_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def route_completion(agent_xy, ref_xy):
    """Fraction of the reference route the agent covered.

    For each agent position we find the nearest reference
    waypoint; the furthest-along nearest waypoint reached
    defines completion. This is robust to the agent not
    perfectly tracking the path.

    Returns:
        completion_frac : float in [0, 1]
        max_idx         : index of furthest reference wp reached
    """
    if len(agent_xy) == 0 or len(ref_xy) == 0:
        return 0.0, 0

    arc = _cumulative_arc_length(ref_xy)
    total = arc[-1] if arc[-1] > 1e-6 else 1.0

    max_arc = 0.0
    max_idx = 0
    # To avoid the agent "completing" by being near a late
    # waypoint early (e.g. routes that loop), we track the
    # furthest monotonic progress within a proximity gate.
    PROX_GATE = 8.0  # metres; agent must be within this of ref
    for px, py in agent_xy:
        d = np.hypot(ref_xy[:, 0] - px, ref_xy[:, 1] - py)
        idx = int(np.argmin(d))
        if d[idx] <= PROX_GATE and arc[idx] > max_arc:
            max_arc = arc[idx]
            max_idx = idx
    return float(max_arc / total), max_idx


def trajectory_error(agent_xy, ref_xy):
    """Mean and max lateral deviation of the agent path from
    the reference path (nearest-waypoint distance).

    Returns:
        mean_err, max_err  (metres)
    """
    if len(agent_xy) == 0 or len(ref_xy) == 0:
        return 0.0, 0.0
    errs = []
    for px, py in agent_xy:
        d = np.hypot(ref_xy[:, 0] - px, ref_xy[:, 1] - py)
        errs.append(float(np.min(d)))
    errs = np.array(errs, dtype=np.float32)
    return float(errs.mean()), float(errs.max())


# ==========================================================
# SMOOTHNESS METRICS (from control/heading logs)
# ==========================================================
def steering_oscillation(steer_series):
    """Mean absolute change in steering between steps.
    Lower = smoother steering.
    """
    s = np.asarray(steer_series, dtype=np.float32)
    if len(s) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(s))))


def compute_jerk(speed_kmh_series, dt=0.05):
    """Approximate mean absolute jerk (m/s^3) from a speed
    series. Speed given in km/h; dt is the sim step seconds.

    accel  = d(speed)/dt
    jerk   = d(accel)/dt
    """
    v = np.asarray(speed_kmh_series, dtype=np.float32) / 3.6  # -> m/s
    if len(v) < 3:
        return 0.0
    accel = np.diff(v) / dt
    jerk  = np.diff(accel) / dt
    return float(np.mean(np.abs(jerk)))


def yaw_oscillation(heading_series):
    """Mean absolute change in heading (deg) between steps."""
    h = np.asarray(heading_series, dtype=np.float32)
    if len(h) < 2:
        return 0.0
    dh = np.diff(h)
    # wrap to [-180, 180]
    dh = (dh + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(dh)))


# ==========================================================
# HELPER: visualize / list spawn points (run with CARLA up)
# ==========================================================
def visualize_spawn_points(host="localhost", port=2000,
                           draw_seconds=60.0):
    """Connect to CARLA and draw spawn-point indices in the
    world so you can pick route start/end indices. Also prints
    them. Run this once (with CARLA running) to choose the
    ROUTES indices above.
    """
    import carla  # local import; only needed when CARLA up
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    cmap = world.get_map()
    sps = cmap.get_spawn_points()
    print(f"Town map: {cmap.name} | spawn points: {len(sps)}")
    for i, sp in enumerate(sps):
        loc = sp.location
        print(f"  [{i:3d}] x={loc.x:8.1f} y={loc.y:8.1f} "
              f"z={loc.z:5.1f} yaw={sp.rotation.yaw:6.1f}")
        world.debug.draw_string(
            loc + carla.Location(z=1.5), str(i),
            draw_shadow=False,
            color=carla.Color(255, 0, 0),
            life_time=draw_seconds, persistent_lines=True,
        )
    print("\nIndices drawn in the CARLA window for "
          f"{draw_seconds:.0f}s. Pick spawn/dest indices "
          "near a roundabout and an intersection, then fill "
          "ROUTES in route_utils.py.")
    return sps


if __name__ == "__main__":
    # Convenience: `python3 route_utils.py` lists spawn points
    # (requires CARLA running).
    visualize_spawn_points()