# ==========================================================
# plot_traj_town.py
# Overlay a recorded eval trajectory + its reference route
# on the Town03 ROAD-NETWORK outline, so you can see exactly
# where the agent drove vs the planned path, in map context.
#
# Draws:
#   - faint grey: all Town03 drivable lane centres (the map)
#   - green thick: the reference route (planner ground truth)
#   - agent path coloured by step (blue=start -> red=end)
#   - START (green dot), END (red X), collision rings
#
# Usage (CARLA must be up — needs the map waypoints):
#   python3 plot_traj_town.py \
#       results/dqle_empty/route_1_roundabout_ep01.csv \
#       route_1_roundabout
#
# Saves <csv>_town.png
# CARLA 0.9.16 | Python 3.10
# ==========================================================

import sys, os, csv, types
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("usage: python3 plot_traj_town.py <csv> <route_key>")
    sys.exit(1)

csv_path  = sys.argv[1]
route_key = sys.argv[2]

# --- robust GlobalRoutePlanner import (same as route_utils) -
def _load_grp():
    roots = [
        "/home/shyam/Carla/PythonAPI/carla",
        os.path.expanduser("~/Carla/PythonAPI/carla"),
    ]
    nav = None
    for r in roots:
        c = os.path.join(r, "agents", "navigation")
        if os.path.isdir(c):
            nav = c; break
    if nav is None:
        raise ImportError("CARLA agents/navigation not found")
    ar = os.path.dirname(nav)
    pa = types.ModuleType("agents"); pa.__path__ = [ar]
    sys.modules["agents"] = pa
    pn = types.ModuleType("agents.navigation"); pn.__path__ = [nav]
    sys.modules["agents.navigation"] = pn
    from agents.navigation.global_route_planner import (
        GlobalRoutePlanner as G)
    return G

import carla
import route_utils as ru
GlobalRoutePlanner = _load_grp()

# --- load agent trajectory ---------------------------------
xs, ys, steps, coll = [], [], [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        xs.append(float(row["x"]))
        ys.append(float(row["y"]))
        steps.append(int(row["step"]))
        coll.append(int(row.get("collision", 0)))
xs = np.array(xs); ys = np.array(ys); steps = np.array(steps)

# --- connect, get map + reference --------------------------
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
cmap = world.get_map()

# Town03 road outline: sample all lane-centre waypoints
topology_wps = cmap.generate_waypoints(4.0)  # every 4 m
road_x = [wp.transform.location.x for wp in topology_wps]
road_y = [wp.transform.location.y for wp in topology_wps]

# reference route
spec = ru.ROUTES[route_key]
grp = GlobalRoutePlanner(cmap, 2.0)
route = grp.trace_route(
    cmap.get_spawn_points()[spec["spawn"]].location,
    cmap.get_spawn_points()[spec["dest"]].location,
)
ref_x = [wp.transform.location.x for wp, _ in route]
ref_y = [wp.transform.location.y for wp, _ in route]

# --- plot --------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 13))

# town road network (faint)
ax.scatter(road_x, road_y, s=1, color="#cccccc", zorder=1)

# reference route (green)
ax.plot(ref_x, ref_y, "-", color="#11aa11", linewidth=5,
        alpha=0.55, zorder=3, label="Reference route (planner)")
ax.scatter(ref_x[-1], ref_y[-1], s=400, marker="*",
           color="#11aa11", edgecolors="black", zorder=6,
           label="Intended DEST")

# agent path (coloured by step)
sc = ax.scatter(xs, ys, c=steps, cmap="coolwarm", s=14,
                zorder=4)
ax.plot(xs, ys, "-", color="gray", linewidth=0.5,
        alpha=0.5, zorder=3)
plt.colorbar(sc, label="step (blue=start -> red=end)")

ax.scatter(xs[0], ys[0], s=220, marker="o", color="lime",
           edgecolors="black", zorder=7, label="Agent START")
ax.scatter(xs[-1], ys[-1], s=260, marker="X", color="red",
           edgecolors="black", zorder=7, label="Agent END")

ci = [i for i, c in enumerate(coll) if c]
if ci:
    ax.scatter(xs[ci], ys[ci], s=320, facecolors="none",
               edgecolors="red", linewidths=3, zorder=8,
               label="collision")

ax.set_xlabel("CARLA x (m)"); ax.set_ylabel("CARLA y (m)")
ax.set_title(f"{os.path.basename(csv_path)} on Town03\n"
             f"{len(xs)} steps, ended step {steps[-1]}")
ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(alpha=0.2)
ax.legend(loc="best")
fig.tight_layout()
out = os.path.splitext(csv_path)[0] + "_town.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"saved {out}")