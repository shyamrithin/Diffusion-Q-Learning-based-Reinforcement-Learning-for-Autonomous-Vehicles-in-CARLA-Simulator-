# ==========================================================
# scan_corner.py
# Auto-scan for the R2 sweeping curve on Town03's outer ring.
#
# Strategy: the only LONG, SMOOTH (junction-free) curves in
# Town03 are the four rounded corners of the outer ring road.
# This scans spawn points in the BOTTOM-LEFT corner region,
# plans a route from each to every other nearby ring spawn,
# and keeps routes that look like a true sweeping curve:
#   * length in a usable band (~70-180 m)
#   * substantial total turn (real bend, ~35-110 deg)
#   * LOW max single-step heading change (no junction spike)
# Ranked best-first. Pick the top row for R2.
#
# Uses the same robust planner import as find_curve_route.py
# (bypasses the local agents/ package shadowing).
#
# Usage:  (CARLA up, Town03)  python3 scan_corner.py
# CARLA 0.9.16 | Python 3.10 | diffusioncarla
# ==========================================================

import sys, os, math, types

# --- robust GlobalRoutePlanner import (see find_curve_route) -
def _load_grp():
    roots = [
        "/home/shyam/Carla/PythonAPI/carla",
        "/home/shyam/carla/PythonAPI/carla",
        os.path.expanduser("~/Carla/PythonAPI/carla"),
        os.path.expanduser("~/carla/PythonAPI/carla"),
    ]
    nav = None
    for r in roots:
        c = os.path.join(r, "agents", "navigation")
        if os.path.isdir(c):
            nav = c; break
    if nav is None:
        raise ImportError("CARLA agents/navigation not found.")
    agents_root = os.path.dirname(nav)
    pa = types.ModuleType("agents"); pa.__path__ = [agents_root]
    sys.modules["agents"] = pa
    pn = types.ModuleType("agents.navigation"); pn.__path__ = [nav]
    sys.modules["agents.navigation"] = pn
    from agents.navigation.global_route_planner import (
        GlobalRoutePlanner as G)
    return G

import carla
GlobalRoutePlanner = _load_grp()

# Bottom-left corner + bottom ring (CARLA coords). High +y =
# bottom of map. The corner arc sweeps from the left side
# (low x, mid-high y) around through the bottom edge (rising
# x, high y). Wide box to catch the whole sweep; the scan
# filters for the curved portion by shape, not position.
X_MIN, X_MAX = -160.0, 130.0
Y_MIN, Y_MAX = 80.0, 230.0

SAMPLING_RES = 2.0
LEN_MIN, LEN_MAX   = 70.0, 260.0
TURN_MIN, TURN_MAX = 35.0, 140.0
MAX_STEP_LIMIT     = 16.0   # ring arcs can show ~12-15/step


def hdg_deltas(wps):
    yaws = [wp.transform.rotation.yaw for wp, _ in wps]
    out = []
    for i in range(1, len(yaws)):
        d = (yaws[i] - yaws[i-1] + 180.0) % 360.0 - 180.0
        out.append(abs(d))
    return out

def length(wps):
    t = 0.0
    for i in range(1, len(wps)):
        a = wps[i-1][0].transform.location
        b = wps[i][0].transform.location
        t += math.hypot(a.x-b.x, a.y-b.y)
    return t

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    cmap = client.get_world().get_map()
    sps  = cmap.get_spawn_points()
    grp  = GlobalRoutePlanner(cmap, SAMPLING_RES)

    region = [i for i, sp in enumerate(sps)
              if X_MIN <= sp.location.x <= X_MAX
              and Y_MIN <= sp.location.y <= Y_MAX]
    print(f"Ring-corner spawn candidates in box: {region}\n")
    if not region:
        print("No spawns in box — widen X/Y_MIN/MAX.")
        return

    results = []
    for a in region:
        for b in region:
            if a == b:
                continue
            try:
                wps = grp.trace_route(
                    sps[a].location, sps[b].location)
            except Exception:
                continue
            if not wps or len(wps) < 2:
                continue
            d  = hdg_deltas(wps)
            L  = length(wps)
            tt = sum(d)
            ms = max(d) if d else 0.0
            results.append((a, b, L, tt, ms, len(wps)))

    # keep sweeping-curve shaped, rank by total turn then length
    good = [r for r in results
            if LEN_MIN <= r[2] <= LEN_MAX
            and TURN_MIN <= r[3] <= TURN_MAX
            and r[4] <= MAX_STEP_LIMIT]
    good.sort(key=lambda r: (-r[3], -r[2]))

    print("=" * 70)
    print(f"{'spawn':>5} {'dest':>5} {'len_m':>8} "
          f"{'totTurn':>8} {'maxStep':>8} {'nWP':>5}")
    print("-" * 70)
    show = good if good else sorted(
        results, key=lambda r: (r[4], -r[3]))[:25]
    for (a, b, L, tt, ms, n) in show:
        tag = "  <-- R2 PICK" if (a, b) == (show[0][0], show[0][1]) and good else ""
        print(f"{a:>5} {b:>5} {L:>8.1f} {tt:>8.1f} "
              f"{ms:>8.1f} {n:>5}{tag}")
    print("=" * 70)
    if good:
        a, b = good[0][0], good[0][1]
        print(f"\nBest sweeping curve: spawn {a} -> dest {b}. "
              f"Send me this pair to write into ROUTES.")
    else:
        print("\nNo clean sweep in box; showing smoothest "
              "routes. Widen the box or raise MAX_STEP_LIMIT.")

if __name__ == "__main__":
    main()