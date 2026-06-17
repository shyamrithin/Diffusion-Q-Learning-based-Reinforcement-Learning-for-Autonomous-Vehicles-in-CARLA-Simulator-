# ==========================================================
# plot_trajectory.py
# Overlay a recorded eval trajectory (from record_eval CSV)
# against its reference route, to SEE where the agent drove,
# where it left the route, and where it terminated.
#
# Usage:
#   python3 plot_trajectory.py results/dqle_empty/route_3_straight_ep01.csv
#   (optionally pass the route key to draw the reference too)
#   python3 plot_trajectory.py <csv> route_3_straight
#
# Saves: <csv_without_ext>_traj.png
# Needs CARLA only if drawing the reference (2nd arg given).
# CARLA 0.9.16 | Python 3.10
# ==========================================================

import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("usage: python3 plot_trajectory.py <csv> [route_key]")
    sys.exit(1)

csv_path  = sys.argv[1]
route_key = sys.argv[2] if len(sys.argv) > 2 else None

# --- load agent trajectory from CSV ------------------------
xs, ys, steps, coll, off = [], [], [], [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        xs.append(float(row["x"]))
        ys.append(float(row["y"]))
        steps.append(int(row["step"]))
        coll.append(int(row.get("collision", 0)))
        off.append(int(row.get("offroad", 0)))
xs = np.array(xs); ys = np.array(ys); steps = np.array(steps)

# --- optional reference route (needs CARLA up) -------------
ref_xy = None
if route_key:
    try:
        import route_utils as ru
        import carla
        client = carla.Client("localhost", 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        cmap = world.get_map()
        spec = ru.ROUTES[route_key]
        ref_xy, _ = ru.generate_reference_route(
            world, cmap, spec["spawn"], spec["dest"])
        print(f"Reference loaded: {len(ref_xy)} pts")
    except Exception as e:
        print(f"(reference not drawn: {e})")

fig, ax = plt.subplots(figsize=(11, 11))

if ref_xy is not None:
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "-", color="#33aa33",
            linewidth=4, alpha=0.5, label="Reference route")
    ax.scatter(ref_xy[-1, 0], ref_xy[-1, 1], s=300,
               marker="*", color="#33aa33", zorder=6,
               label="Route DEST")

# colour the agent path by step (early=blue -> late=red)
sc = ax.scatter(xs, ys, c=steps, cmap="coolwarm", s=12,
                zorder=4)
ax.plot(xs, ys, "-", color="gray", linewidth=0.6, alpha=0.5)
plt.colorbar(sc, label="step (blue=start, red=end)")

ax.scatter(xs[0], ys[0], s=180, marker="o", color="lime",
           edgecolors="black", zorder=7, label="Agent START")
ax.scatter(xs[-1], ys[-1], s=220, marker="X", color="red",
           edgecolors="black", zorder=7,
           label="Agent END (terminated)")

# mark collision / offroad steps if any
ci = [i for i, c in enumerate(coll) if c]
if ci:
    ax.scatter(xs[ci], ys[ci], s=260, facecolors="none",
               edgecolors="red", linewidths=2.5, zorder=8,
               label="collision step")

ax.set_xlabel("CARLA x (m)"); ax.set_ylabel("CARLA y (m)")
ax.set_title(f"Trajectory: {os.path.basename(csv_path)}\n"
             f"{len(xs)} steps, ended at step {steps[-1]}")
ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
out = os.path.splitext(csv_path)[0] + "_traj.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}")