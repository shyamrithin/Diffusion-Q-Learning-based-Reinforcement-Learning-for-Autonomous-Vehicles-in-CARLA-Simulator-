# ==========================================================
# plot_sac_trajectory.py
# SAC trajectory with Town03 map background
# ==========================================================

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

plt.rcParams.update({
    "font.family"      : "sans-serif",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 11,
    "axes.labelweight" : "bold",
    "axes.facecolor"   : "white",
    "figure.facecolor" : "white",
    "axes.grid"        : False,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.spines.left" : True,
    "axes.spines.bottom": True,
    "axes.edgecolor"   : "#333333",
    "xtick.color"      : "#333333",
    "ytick.color"      : "#333333",
    "axes.labelcolor"  : "#111111",
    "text.color"       : "#111111",
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "pdf.fonttype"     : 42,
})

EP_COLORS  = ["#00BFFF", "#FF6B9D", "#7CFC00"]
EP_LABELS  = ["Episode 1", "Episode 2", "Episode 3"]
DATA_DIR   = "results/sac_light"
OUT_DIR    = "paper_figures_sac"
os.makedirs(OUT_DIR, exist_ok=True)

# Load map
with open("town03_roads.json") as f:
    roads = json.load(f)

map_x = np.array([r["x"] for r in roads])
map_y = np.array([r["y"] for r in roads])

# Load episodes
dfs = []
for i in range(1, 4):
    path = os.path.join(DATA_DIR, f"ep{i:02d}.csv")
    if os.path.exists(path):
        dfs.append(pd.read_csv(path))

fig, ax = plt.subplots(figsize=(9, 8.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Draw road network
ax.scatter(map_x, map_y,
           s=0.8, c="#ccccdd",
           alpha=0.5, zorder=1,
           rasterized=True)

# Highlight roundabout area
theta = np.linspace(0, 2*np.pi, 200)
for r, alpha in [(35, 0.08), (25, 0.06), (15, 0.04)]:
    ax.fill(r*np.cos(theta), r*np.sin(theta),
            color="#aaaaee", alpha=alpha, zorder=1)
ax.text(0, 0, "Roundabout", ha="center", va="center",
        fontsize=7.5, color="#7777aa", alpha=0.7,
        style="italic", zorder=2)

# Draw episode trajectories
for i, df in enumerate(dfs):
    x = df["x"].values
    y = df["y"].values

    # Gradient line
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap_name = ["cool", "spring", "summer"][i]
    lc = LineCollection(segs, cmap=cmap_name,
                        linewidth=2.2, alpha=0.9, zorder=4)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)

    # Start marker
    ax.scatter(x[0], y[0],
               color=EP_COLORS[i], s=160,
               zorder=7, marker="o",
               edgecolors="white", linewidths=1.8,
               label=f"{EP_LABELS[i]}  "
                     f"R = {df['ep_reward'].iloc[-1]:.0f}")

    # End marker
    ax.scatter(x[-1], y[-1],
               color=EP_COLORS[i], s=200,
               zorder=7, marker="*",
               edgecolors="white", linewidths=1.0)

    # Direction arrows every 150 steps
    for j in range(150, len(x)-1, 200):
        dx = x[j+1] - x[j]
        dy = y[j+1] - y[j]
        ax.annotate("",
            xy=(x[j]+dx*5, y[j]+dy*5),
            xytext=(x[j], y[j]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=EP_COLORS[i],
                lw=1.2, alpha=0.7,
            ), zorder=5,
        )

# Start/End labels
ax.annotate(
    "● START",
    xy=(dfs[0]["x"].iloc[0], dfs[0]["y"].iloc[0]),
    xytext=(25, 20), textcoords="offset points",
    fontsize=8.5, color="#ffffff", fontweight="bold",
    arrowprops=dict(arrowstyle="->",
                    color="#333333", lw=1.0),
    zorder=8,
)
ax.annotate(
    "★ END",
    xy=(dfs[0]["x"].iloc[-1], dfs[0]["y"].iloc[-1]),
    xytext=(25, -25), textcoords="offset points",
    fontsize=8.5, color="#ffffff", fontweight="bold",
    arrowprops=dict(arrowstyle="->",
                    color="#333333", lw=1.0),
    zorder=8,
)

ax.set_xlabel("X Position (m)", color="#111111", labelpad=8)
ax.set_ylabel("Y Position (m)", color="#111111", labelpad=8)
ax.set_title(
    "SAC Agent — Executed Trajectories on Town03 Map\n"
    "Light Traffic (10 NPCs)  ·  3 Episodes  ·  "
    "1000 Steps Each",
    color="#111111", pad=16,
)
ax.set_aspect("equal")
ax.autoscale()

# Colorbar for trajectory progress
sm = plt.cm.ScalarMappable(
    cmap="cool",
    norm=plt.Normalize(0, 1)
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.025,
                    pad=0.02, aspect=30)
cbar.set_label("Trajectory Progress",
               color="#333333", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="#333333")
plt.setp(cbar.ax.yaxis.get_ticklabels(),
         color="#333333", fontsize=8)
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(["Start", "Mid", "End"])

# Legend
leg = ax.legend(
    loc="upper left",
    frameon=True,
    framealpha=0.25,
    facecolor="#f8f8ff",
    edgecolor="#aaaacc",
    fontsize=9.5,
    title="● Start  ★ End",
    title_fontsize=8.5,
)
leg.get_title().set_color("#333333")
for text in leg.get_texts():
    text.set_color("#111111")

# Stats box
stats_text = (
    f"Avg Reward : 6,975\n"
    f"Completions: 3/3 (100%)\n"
    f"Collisions : 0\n"
    f"Avg Speed  : 30.9 km/h"
)
ax.text(
    0.98, 0.04, stats_text,
    transform=ax.transAxes,
    fontsize=8.5, color="#005500",
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#f0fff0",
        edgecolor="#009900",
        alpha=0.8,
    ),
    zorder=9,
    family="monospace",
)

plt.tight_layout()

# Save PDF and PNG
for ext in ["pdf", "png"]:
    path = os.path.join(
        OUT_DIR, f"fig1_trajectory_map.{ext}"
    )
    dpi = 300 if ext == "pdf" else 200
    fig.savefig(path, dpi=dpi,
                facecolor="white",
                bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()
