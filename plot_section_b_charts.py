# ==========================================================
# plot_section_b_charts.py
# Generate all Section B data-driven charts from the eval
# aggregates (results/section_b/*.csv). No CARLA needed.
#
# Reads:
#   results/section_b/per_route_breakdown.csv
#   results/section_b/table_iv.csv
#
# Produces (paper_figures/):
#   fig9_collision_vs_density.png   - R1 collision rate, grouped bars
#   fig10_saferate_vs_density.png   - pooled safe% line chart
#   fig11_completion_by_route.png   - completion grouped bars per route
#   fig12_reward_vs_density.png     - mean reward grouped bars
#   fig13_safety_heatmap.png        - agent x density safe% heatmap
#
# All values are REAL, read from the CSVs. n=3 per cell.
#
# Usage:  python3 plot_section_b_charts.py   (from ~/Carla/RLCarla)
# ==========================================================

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper_figures"
os.makedirs(OUT, exist_ok=True)
SB = "results/section_b"

DENS = ["empty", "light", "medium", "heavy"]
AGENTS = ["DQL-E", "SAC", "PPO"]
COLORS = {"DQL-E": "#1f77b4", "SAC": "#2ca02c", "PPO": "#d62728"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})

# ---- load per-route breakdown ----
rows = []
with open(os.path.join(SB, "per_route_breakdown.csv")) as f:
    for r in csv.DictReader(f):
        rows.append(r)

def frac(s):  # "1/3" -> 0.3333
    a, b = s.split("/"); return float(a)/float(b)

def get(agent, route, traffic, col):
    for r in rows:
        if r["agent"]==agent and r["route"]==route and r["traffic"]==traffic:
            return r[col]
    return None

# ---- load table_iv for pooled safe% ----
tiv = []
with open(os.path.join(SB, "table_iv.csv")) as f:
    for r in csv.DictReader(f):
        tiv.append(r)
def safe_pct(agent, traffic):
    for r in tiv:
        if r["Agent"]==agent and r["Density"]==traffic:
            return float(r["Safe%"].replace("%",""))
    return None

x = np.arange(len(DENS))
w = 0.25

# ===== Fig 9: R1 collision rate vs density (grouped bars) =====
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for i, ag in enumerate(AGENTS):
    vals = [frac(get(ag, "route_1_roundabout", d, "collision"))*100 for d in DENS]
    ax.bar(x + (i-1)*w, vals, w, label=ag, color=COLORS[ag])
ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in DENS])
ax.set_ylabel("Roundabout Collision Rate (%)")
ax.set_xlabel("Traffic Density")
ax.set_title("R1 Roundabout: Collision Rate vs Traffic Density (n=3)")
ax.legend()
ax.set_ylim(0, 100)
fig.tight_layout(); fig.savefig(f"{OUT}/fig9_collision_vs_density.png", bbox_inches="tight"); plt.close()
print("saved fig9_collision_vs_density.png")

# ===== Fig 10: pooled safe% vs density (line) =====
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for ag in AGENTS:
    vals = [safe_pct(ag, d) for d in DENS]
    ax.plot(x, vals, marker="o", linewidth=2.4, label=ag, color=COLORS[ag])
ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in DENS])
ax.set_ylabel("Safe Rate (%)  (no collision & no off-road)")
ax.set_xlabel("Traffic Density")
ax.set_title("Pooled Safe Rate vs Traffic Density (all routes, n=3)")
ax.legend(); ax.set_ylim(-5, 105)
fig.tight_layout(); fig.savefig(f"{OUT}/fig10_saferate_vs_density.png", bbox_inches="tight"); plt.close()
print("saved fig10_saferate_vs_density.png")

# ===== Fig 11: completion by route (grouped bars, averaged over density) =====
ROUTES = [("route_1_roundabout","R1 Roundabout"),
          ("route_2_curve","R2 Curve"),
          ("route_3_straight","R3 Straight")]
fig, ax = plt.subplots(figsize=(7.5, 4.4))
rx = np.arange(len(ROUTES))
for i, ag in enumerate(AGENTS):
    vals = []
    for rk,_ in ROUTES:
        comps = [float(get(ag, rk, d, "mean_completion")) for d in DENS]
        vals.append(np.mean(comps)*100)
    ax.bar(rx + (i-1)*w, vals, w, label=ag, color=COLORS[ag])
ax.set_xticks(rx); ax.set_xticklabels([n for _,n in ROUTES])
ax.set_ylabel("Mean Route Completion (%)")
ax.set_title("Route Completion by Route (avg over densities, n=3)")
ax.legend(); ax.set_ylim(0, 105)
fig.tight_layout(); fig.savefig(f"{OUT}/fig11_completion_by_route.png", bbox_inches="tight"); plt.close()
print("saved fig11_completion_by_route.png")

# ===== Fig 12: mean reward vs density (grouped bars, avg over routes) =====
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for i, ag in enumerate(AGENTS):
    vals = []
    for d in DENS:
        rs = [float(get(ag, rk, d, "mean_reward")) for rk,_ in ROUTES]
        vals.append(np.mean(rs))
    ax.bar(x + (i-1)*w, vals, w, label=ag, color=COLORS[ag])
ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in DENS])
ax.set_ylabel("Mean Episode Reward")
ax.set_xlabel("Traffic Density")
ax.set_title("Mean Reward vs Traffic Density (avg over routes, n=3)")
ax.axhline(0, color="#999", linewidth=0.8)
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig12_reward_vs_density.png", bbox_inches="tight"); plt.close()
print("saved fig12_reward_vs_density.png")

# ===== Fig 13: safe% heatmap (agent x density) =====
fig, ax = plt.subplots(figsize=(6.5, 3.6))
mat = np.array([[safe_pct(ag, d) for d in DENS] for ag in AGENTS])
im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(DENS))); ax.set_xticklabels([d.capitalize() for d in DENS])
ax.set_yticks(range(len(AGENTS))); ax.set_yticklabels(AGENTS)
for i in range(len(AGENTS)):
    for j in range(len(DENS)):
        ax.text(j, i, f"{mat[i,j]:.0f}%", ha="center", va="center",
                color="black", fontsize=11, fontweight="bold")
ax.set_title("Safe Rate (%) by Agent and Traffic Density (n=3)")
fig.colorbar(im, ax=ax, label="Safe %")
ax.grid(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig13_safety_heatmap.png", bbox_inches="tight"); plt.close()
print("saved fig13_safety_heatmap.png")

print("\nAll Section B charts generated in paper_figures/")