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
# --- REVISION (routes locked + planner import fixed) -------
#   1) ROUTES filled with the three locked, visually-verified
#      evaluation routes (Town03):
#        R1 roundabout : spawn 123 -> dest 41
#                        (S->N sweep through central roundabout
#                         — multimodal yield/merge junction)
#        R2 curve      : spawn 107 -> dest 170
#                        (long smooth corner arc, maxStep ~3 deg
#                         — turning ability / trajectory smooth.)
#        R3 straight   : spawn 249 -> dest 21
#                        (dead-straight arterial — lane-keeping)
#
#   2) GlobalRoutePlanner import FIXED. The previous version
#      did `from agents.navigation...` after adding CARLA's
#      path to sys.path — but this repo has its OWN local
#      `agents/` package (agents/ql_diffusion.py, ...), which
#      SHADOWS CARLA's `agents` package when running from the
#      repo dir. The import therefore failed and the old
#      try/except SILENTLY set _GRP_AVAILABLE=False, so
#      generate_reference_route() never produced a path
#      (trajectory_error / route_completion would be empty).
#      We now load CARLA's planner by pointing the package
#      __path__ at CARLA's real directories, so the local
#      `agents/` cannot shadow it and the whole import chain
#      (global_route_planner -> local_planner -> controller
#      -> ...) resolves to CARLA's files.
#
# CARLA 0.9.16 | Python 3.10
# ==========================================================

import os
import sys
import math
import types
import numpy as np


# ----------------------------------------------------------
# Robust CARLA GlobalRoutePlanner import.
#
# PROBLEM: this repo has a LOCAL `agents/` package. Run from
# the repo dir, `import agents` resolves to the local one
# (which has no `navigation` submodule), so the normal
# `from agents.navigation...` import fails — and a try/except
# wrapper hides it. Adding CARLA's path does NOT help because
# the name `agents` is already taken by the local package.
#
# FIX: override the `agents` and `agents.navigation` package
# entries in sys.modules with module objects whose __path__
# points at CARLA's REAL directories. The normal import
# machinery then resolves every (possibly chained) submodule
# of the planner to CARLA's files by path.
# ----------------------------------------------------------
_CARLA_AGENTS_PATHS = [
    "/home/shyam/Carla/PythonAPI/carla",
    "/home/shyam/carla/PythonAPI/carla",
    os.path.expanduser("~/Carla/PythonAPI/carla"),
    os.path.expanduser("~/carla/PythonAPI/carla"),
]


def _load_global_route_planner():
    nav = None
    for _p in _CARLA_AGENTS_PATHS:
        cand = os.path.join(_p, "agents", "navigation")
        if os.path.isdir(cand):
            nav = cand
            break
    if nav is None:
        raise ImportError(
            "Could not locate CARLA agents/navigation dir. "
            "Edit _CARLA_AGENTS_PATHS in route_utils.py."
        )

    agents_root = os.path.dirname(nav)  # .../agents

    # Override the package names so chained absolute imports
    # inside the planner resolve to CARLA's files by path,
    # regardless of the repo's local `agents/` shadowing.
    pkg_agents = types.ModuleType("agents")
    pkg_agents.__path__ = [agents_root]
    sys.modules["agents"] = pkg_agents

    pkg_nav = types.ModuleType("agents.navigation")
    pkg_nav.__path__ = [nav]
    sys.modules["agents.navigation"] = pkg_nav

    from agents.navigation.global_route_planner import (
        GlobalRoutePlanner as _GRP,
    )
    return _GRP


try:
    GlobalRoutePlanner = _load_global_route_planner()
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
# LOCKED routes (Town03) — visually verified in the CARLA
# window via draw_routes.py. Keep the SAME three routes for
# ALL agents (DQL-E, SAC, PPO).
#
#   R1 roundabout : 123 -> 41   sweeps through central
#                               roundabout (multimodal merge)
#   R2 curve      : 107 -> 170  long smooth corner arc
#                               (turning / smoothness)
#   R3 straight   : 249 -> 21   dead-straight arterial
#                               (lane-keeping)
ROUTES = {
    "route_1_roundabout": {"spawn": 123, "dest": 41},
    "route_2_curve":      {"spawn": 107, "dest": 170},
    "route_3_straight":   {"spawn": 249, "dest": 21},
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


def generate_route_by_name(world, carla_map, route_name,
                           sample_res=ROUTE_SAMPLE_RES):
    """Convenience wrapper: generate a reference route by its
    ROUTES key (e.g. 'route_1_roundabout').

    Returns (ref_xy, start_transform) — same as
    generate_reference_route.
    """
    if route_name not in ROUTES:
        raise KeyError(
            f"Unknown route '{route_name}'. "
            f"Available: {list(ROUTES.keys())}"
        )
    spec = ROUTES[route_name]
    if spec["spawn"] is None or spec["dest"] is None:
        raise ValueError(
            f"Route '{route_name}' has unfilled indices."
        )
    return generate_reference_route(
        world, carla_map,
        spec["spawn"], spec["dest"], sample_res,
    )


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


def verify_routes(host="localhost", port=2000):
    """Sanity check: confirm the planner is importable and
    that each locked route traces a non-empty reference path.
    Run with CARLA up:  python3 -c \
        'import route_utils; route_utils.verify_routes()'
    """
    if not _GRP_AVAILABLE:
        print("GlobalRoutePlanner NOT available — import "
              f"failed: {_GRP_IMPORT_ERROR}")
        return
    import carla
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    cmap = world.get_map()
    print(f"Planner OK. Map: {cmap.name}")
    for name, spec in ROUTES.items():
        try:
            ref_xy, start_tf = generate_reference_route(
                world, cmap, spec["spawn"], spec["dest"]
            )
            arc = _cumulative_arc_length(ref_xy)
            print(f"  {name:20s} {spec['spawn']:>3}->"
                  f"{spec['dest']:>3} | {len(ref_xy):4d} wp "
                  f"| {arc[-1]:7.1f} m")
        except Exception as e:
            print(f"  {name:20s} FAILED: {e}")


if __name__ == "__main__":
    # Convenience: `python3 route_utils.py` lists spawn points
    # (requires CARLA running).
    visualize_spawn_points()