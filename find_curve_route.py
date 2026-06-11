# ==========================================================
# find_curve_route.py
# Route finder for fixed-route evaluation (Town03)
#
# Uses CARLA's GlobalRoutePlanner (the A*/shortest-path
# planner over the road-topology graph — the SAME planner
# route_utils.generate_reference_route() uses) to trace the
# ground-truth path between candidate spawn/dest index pairs,
# then ranks them so you can pick a genuine LONG SWEEPING
# CURVE for Route 2.
#
# How to read the output, per candidate pair:
#   length_m       — planned path length in metres
#   total_turn_deg — sum of |heading change| along the path
#                    (total bend; a sweeping curve has a
#                     substantial total, e.g. 40-110 deg)
#   max_step_deg   — largest single-step heading jump
#                    (HIGH = sharp turn / junction / U-turn;
#                     a SWEEPING curve keeps this LOW, e.g.
#                     < ~8 deg per 2 m step)
#   sweep          — total_turn_deg / length_m * 100
#                    (gentle sweep = low-to-moderate; a sharp
#                     cornering route spikes this)
#
# PICK FOR R2: a pair with length > ~60 m, total_turn
# moderate-to-high, and max_step LOW (no junction spike).
# That is a long sweeping curve the open-loop lane-follower
# will naturally track — so trajectory_error vs this path is
# a valid turning-ability measurement.
#
# Usage:
#   1) Start CARLA (Town03 loaded).
#   2) Edit CANDIDATES below with (spawn_idx, dest_idx) pairs
#      to test (seed guesses provided — add your own).
#   3) python3 find_curve_route.py
#   4) Read the ranked table; pick the sweeping-curve pair.
#
# CARLA 0.9.16 | Python 3.10 | diffusioncarla
# ==========================================================

import sys
import math

# --- Robust GlobalRoutePlanner import ----------------------
# PROBLEM: this repo has its OWN local `agents/` package
# (agents/ql_diffusion.py, ...). Running from the repo dir,
# Python resolves `import agents` to the LOCAL package, which
# has no `navigation` submodule, so the normal
# `from agents.navigation...` import fails (and any try/except
# wrapper silently hides it). Adding CARLA's path does not
# help because the name `agents` is already taken.
#
# FIX: load CARLA's modules directly from their FILE PATHS via
# importlib and register them under aliased names, so the
# local `agents` package can't shadow them. We load
# local_planner first (global_route_planner imports RoadOption
# from it) and inject the needed names before loading
# global_route_planner.
import os
import importlib.util


def _load_global_route_planner():
    carla_roots = [
        "/home/shyam/Carla/PythonAPI/carla",
        "/home/shyam/carla/PythonAPI/carla",
        os.path.expanduser("~/Carla/PythonAPI/carla"),
        os.path.expanduser("~/carla/PythonAPI/carla"),
    ]
    nav = None
    for root in carla_roots:
        cand = os.path.join(root, "agents", "navigation")
        if os.path.isdir(cand):
            nav = cand
            break
    if nav is None:
        raise ImportError(
            "Could not locate CARLA agents/navigation dir. "
            "Edit carla_roots in _load_global_route_planner()."
        )

    import types

    # Register 'agents' and 'agents.navigation' as packages
    # whose __path__ points at CARLA's real navigation dir, so
    # that ANY chained absolute import inside the planner
    # (local_planner -> controller -> ...) resolves to CARLA's
    # files by path, regardless of the repo's local `agents/`
    # package shadowing the name on sys.path.
    agents_root = os.path.dirname(nav)  # .../agents

    pkg_agents = types.ModuleType("agents")
    pkg_agents.__path__ = [agents_root]
    sys.modules["agents"] = pkg_agents

    pkg_nav = types.ModuleType("agents.navigation")
    pkg_nav.__path__ = [nav]
    sys.modules["agents.navigation"] = pkg_nav

    # With the package __path__ entries set above, the normal
    # import machinery will now find CARLA's submodules by
    # path. Import the planner the ordinary way.
    from agents.navigation.global_route_planner import (
        GlobalRoutePlanner as _GRP
    )
    return _GRP


import carla
GlobalRoutePlanner = _load_global_route_planner()


# ----------------------------------------------------------
# Candidate (spawn_idx, dest_idx) pairs to evaluate.
# Seeded with guesses; ADD YOUR OWN. Indices refer to the
# spawn-point list printed by route_utils.py (Town03, 265).
# ----------------------------------------------------------
CANDIDATES = [
    # confirmed routes (for reference / sanity check)
    (123,  41),   # R1 roundabout (S -> N through circle)
    (249,  21),   # R3 straight arterial (eastbound)

    # R2 sweeping-curve candidates — guesses to test:
    (1,   147),   # arterial -> may bend
    (249, 201),   # arterial east then toward y=62 road
    (120, 113),   # westbound -> N-S road (likely junction turn)
    (208,  53),   # elevated highway run (often sweeping)
    (54,   76),   # elevated curve section
    (235, 234),   # interchange curve
    (87,   89),   # y=60 road eastbound run
    (175, 253),   # diagonal -> may sweep
]

SAMPLING_RES = 2.0   # metres between planned waypoints


def heading_change_deg(wps):
    """Per-step absolute heading change (deg) along a route
    of (waypoint, road_option) tuples."""
    yaws = []
    for wp, _ in wps:
        yaws.append(wp.transform.rotation.yaw)
    deltas = []
    for i in range(1, len(yaws)):
        d = yaws[i] - yaws[i - 1]
        d = (d + 180.0) % 360.0 - 180.0   # wrap [-180,180]
        deltas.append(abs(d))
    return deltas


def route_length(wps):
    total = 0.0
    for i in range(1, len(wps)):
        a = wps[i - 1][0].transform.location
        b = wps[i][0].transform.location
        total += math.sqrt(
            (a.x - b.x) ** 2 + (a.y - b.y) ** 2
        )
    return total


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world   = client.get_world()
    cmap    = world.get_map()
    spawns  = cmap.get_spawn_points()
    grp     = GlobalRoutePlanner(cmap, SAMPLING_RES)

    print(f"Town03 spawn points: {len(spawns)}")
    print(f"Sampling resolution: {SAMPLING_RES} m\n")

    rows = []
    for (a, b) in CANDIDATES:
        if a >= len(spawns) or b >= len(spawns):
            print(f"  ({a:3d}->{b:3d})  SKIP — index OOB")
            continue
        try:
            wps = grp.trace_route(
                spawns[a].location, spawns[b].location
            )
        except Exception as e:
            print(f"  ({a:3d}->{b:3d})  FAIL — {e}")
            continue
        if not wps or len(wps) < 2:
            print(f"  ({a:3d}->{b:3d})  empty route")
            continue

        deltas    = heading_change_deg(wps)
        length    = route_length(wps)
        total_turn = sum(deltas)
        max_step   = max(deltas) if deltas else 0.0
        sweep      = (total_turn / length * 100.0
                      if length > 0 else 0.0)
        rows.append((a, b, length, total_turn,
                     max_step, sweep, len(wps)))

    # rank: prefer long, gently-sweeping (low max_step),
    # meaningful total_turn
    print("\n" + "=" * 78)
    print(f"{'spawn':>5} {'dest':>5} {'len_m':>8} "
          f"{'totTurn':>8} {'maxStep':>8} {'sweep':>7} "
          f"{'nWP':>5}")
    print("-" * 78)
    for (a, b, length, tt, ms, sw, n) in rows:
        flag = ""
        if length > 60 and 30 < tt < 130 and ms < 9:
            flag = "  <-- SWEEPING CURVE candidate"
        elif ms >= 20:
            flag = "  (sharp turn/junction)"
        elif tt < 15:
            flag = "  (essentially straight)"
        print(f"{a:>5} {b:>5} {length:>8.1f} "
              f"{tt:>8.1f} {ms:>8.1f} {sw:>7.2f} "
              f"{n:>5}{flag}")
    print("=" * 78)
    print(
        "\nPick the SWEEPING CURVE candidate for R2: long "
        "length, moderate total turn, LOW max-step.\n"
        "Then send me the (spawn, dest) pair and I'll write "
        "all three routes into ROUTES."
    )


if __name__ == "__main__":
    main()