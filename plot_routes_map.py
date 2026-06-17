# ==========================================================
# plot_routes_map.py
# Render all three evaluation routes (R1 roundabout, R2 curve,
# R3 straight) as a single top-down figure over the Town03
# road network, for reference in the Section B paper doc.
#
# Uses route_utils.generate_reference_route() (same planner the
# eval uses) to get each route's reference waypoints, and draws
# the Town03 road centrelines as background via the CARLA map
# topology. No live spectator/pygame needed -> clean figure.
#
# Output: paper_figures/eval_routes_town03.png
#
# Routes (from route_utils.ROUTES):
#   R1 roundabout : 123 -> 41
#   R2 curve      : 170 -> 45   (sweeping arc, updated)
#   R3 straight   : 249 -> 21
#
# Usage:  python3 plot_routes_map.py    (CARLA server up)
# ==========================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import carla
import route_utils

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
cmap = world.get_map()
SAMPLE_RES = 2.0

ROUTES = route_utils.ROUTES  # {name: {spawn, dest}}

# colours + display names
STYLE = {
    "route_1_roundabout": ("#1f77b4", "R1: Roundabout (123\u219241)"),
    "route_2_curve":      ("#2ca02c", "R2: Sweeping Curve (170\u219245)"),
    "route_3_straight":   ("#d62728", "R3: Straight Arterial (249\u219221)"),
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.size": 11,
})

fig, ax = plt.subplots(figsize=(8.5, 8.5))

# --- background: Town03 road network from topology ---
topology = cmap.get_topology()  # list of (wp_start, wp_end) tuples
for seg_start, seg_end in topology:
    # densely sample each segment for a smooth road centreline
    wp = seg_start
    xs, ys = [wp.transform.location.x], [wp.transform.location.y]
    for _ in range(200):
        nxts = wp.next(2.0)
        if not nxts:
            break
        wp = nxts[0]
        xs.append(wp.transform.location.x)
        ys.append(wp.transform.location.y)
        if wp.transform.location.distance(seg_end.transform.location) < 2.0:
            break
    ax.plot(xs, ys, color="#cccccc", linewidth=1.0, zorder=1)

# --- foreground: the three eval routes ---
sp = cmap.get_spawn_points()
for name, meta in ROUTES.items():
    color, label = STYLE.get(name, ("#000000", name))
    ref_xy, start_tf = route_utils.generate_reference_route(
        world, cmap, meta["spawn"], meta["dest"], SAMPLE_RES
    )
    xs = [p[0] for p in ref_xy]
    ys = [p[1] for p in ref_xy]
    ax.plot(xs, ys, color=color, linewidth=3.2, label=label,
            zorder=3, solid_capstyle="round")
    # start marker (circle) + end marker (square)
    ax.scatter([xs[0]], [ys[0]], color=color, s=90, marker="o",
               edgecolor="white", linewidth=1.5, zorder=4)
    ax.scatter([xs[-1]], [ys[-1]], color=color, s=110, marker="s",
               edgecolor="white", linewidth=1.5, zorder=4)

# CARLA y-axis points "down" in world coords; invert so the
# figure matches the top-down spectator view orientation.
ax.invert_yaxis()
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("Map X (m)")
ax.set_ylabel("Map Y (m)")
ax.set_title("CARLA Town03 \u2014 Evaluation Routes\n"
             "(\u25cf start, \u25a0 destination)")
ax.legend(loc="upper right", framealpha=0.95)
ax.grid(True, alpha=0.2, linestyle="--")

fig.tight_layout()
out = "paper_figures/eval_routes_town03.png"
import os
os.makedirs("paper_figures", exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")