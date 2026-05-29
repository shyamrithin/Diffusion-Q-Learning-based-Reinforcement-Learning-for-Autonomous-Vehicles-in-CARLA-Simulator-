# ==========================================================
# plot_sac_results.py
# Publication-quality graphs for SAC light traffic results
#
# Generates:
#   fig1_trajectory.pdf      - Path trajectory (overhead)
#   fig2_control_signals.pdf - Throttle/Steer/Brake
#   fig3_speed_profile.pdf   - Speed over time
#   fig4_reward_curve.pdf    - Cumulative reward
#   fig5_combined.pdf        - All 4 in one figure (best for paper)
#
# Run: python3 plot_sac_results.py
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import seaborn as sns
from scipy.ndimage import uniform_filter1d

# ==========================================================
# STYLE — PhD-level publication quality
# ==========================================================
plt.rcParams.update({
    "font.family"        : "serif",
    "font.serif"         : ["DejaVu Serif"],
    "font.size"          : 11,
    "axes.titlesize"     : 12,
    "axes.titleweight"   : "bold",
    "axes.labelsize"     : 11,
    "axes.labelweight"   : "bold",
    "xtick.labelsize"    : 9.5,
    "ytick.labelsize"    : 9.5,
    "legend.fontsize"    : 9.5,
    "legend.framealpha"  : 0.92,
    "legend.edgecolor"   : "#cccccc",
    "figure.dpi"         : 300,
    "figure.facecolor"   : "white",
    "axes.facecolor"     : "#FAFAFA",
    "axes.grid"          : True,
    "axes.grid.which"    : "both",
    "grid.alpha"         : 0.25,
    "grid.linewidth"     : 0.6,
    "grid.linestyle"     : "--",
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.linewidth"     : 1.1,
    "lines.linewidth"    : 2.0,
    "savefig.dpi"        : 300,
    "savefig.bbox"       : "tight",
    "savefig.facecolor"  : "white",
    "pdf.fonttype"       : 42,
    "ps.fonttype"        : 42,
})

# Colour palette — elegant and distinct
EP_COLORS = ["#2196F3", "#E91E63", "#4CAF50"]
EP_LABELS = ["Episode 1", "Episode 2", "Episode 3"]
ALPHA_LINE = 0.85
SMOOTH_W   = 30    # smoothing window

OUT_DIR = "paper_figures_sac"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = "results/sac_light"

# ==========================================================
# LOAD DATA
# ==========================================================
def load_episodes():
    dfs = []
    for i in range(1, 4):
        path = os.path.join(DATA_DIR, f"ep{i:02d}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Loaded ep{i:02d}: {len(df)} steps")
        else:
            print(f"Missing: {path}")
    return dfs

def smooth(y, w=SMOOTH_W):
    return uniform_filter1d(y.astype(float), size=w)

# ==========================================================
# FIGURE 1 — Path Trajectory
# ==========================================================
def plot_trajectory(dfs):
    fig, ax = plt.subplots(figsize=(7, 6.5))

    for i, df in enumerate(dfs):
        x = df["x"].values
        y = df["y"].values

        # Gradient line showing progress
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        lc     = LineCollection(
            segs, cmap="Blues" if i == 0 else
                       "RdPu" if i == 1 else "Greens",
            linewidth=2.0, alpha=0.85, zorder=3,
        )
        lc.set_array(np.linspace(0.3, 1.0, len(segs)))
        ax.add_collection(lc)

        # Start and end markers
        ax.scatter(x[0], y[0], color=EP_COLORS[i],
                   s=120, zorder=6, marker="o",
                   edgecolors="white", linewidths=1.5,
                   label=f"{EP_LABELS[i]} (R={df['ep_reward'].iloc[-1]:.0f})")
        ax.scatter(x[-1], y[-1], color=EP_COLORS[i],
                   s=140, zorder=6, marker="*",
                   edgecolors="white", linewidths=1.0)

    # Annotations
    ax.annotate("Start", xy=(dfs[0]["x"].iloc[0],
                              dfs[0]["y"].iloc[0]),
                xytext=(20, 20), textcoords="offset points",
                fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->",
                                color="#555555", lw=1.2))
    ax.annotate("End ★", xy=(dfs[0]["x"].iloc[-1],
                               dfs[0]["y"].iloc[-1]),
                xytext=(20, -25), textcoords="offset points",
                fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->",
                                color="#555555", lw=1.2))

    ax.set_xlabel("X Position (m)", labelpad=8)
    ax.set_ylabel("Y Position (m)", labelpad=8)
    ax.set_title(
        "SAC Agent — Executed Trajectories\n"
        "Town03, Light Traffic (10 NPCs), 3 Episodes",
        pad=14,
    )
    ax.set_aspect("equal")
    ax.autoscale()
    leg = ax.legend(loc="upper right", frameon=True,
                    title="Episodes  ● Start  ★ End",
                    title_fontsize=8.5)
    leg.get_title().set_fontweight("bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_trajectory.pdf")
    fig.savefig(path)
    plt.close()
    print(f"[Fig 1] Saved: {path}")


# ==========================================================
# FIGURE 2 — Control Signals
# ==========================================================
def plot_control_signals(dfs):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7),
                              sharex=True)
    fig.subplots_adjust(hspace=0.08)

    signals = ["throttle", "steer", "brake"]
    ylabels = ["Throttle", "Steering", "Brake"]
    ylims   = [(0, 1.05), (-1.05, 1.05), (0, 1.05)]
    fills   = [True, False, True]
    fill_colors = ["#2196F3", "#E91E63", "#FF5722"]

    for si, (sig, ylabel, ylim, fill, fc) in enumerate(
        zip(signals, ylabels, ylims, fills, fill_colors)
    ):
        ax = axes[si]

        for i, df in enumerate(dfs):
            y  = smooth(df[sig].values)
            xs = np.arange(len(y))
            ax.plot(xs, y, color=EP_COLORS[i],
                    alpha=ALPHA_LINE, lw=1.8,
                    label=EP_LABELS[i] if si == 0 else "")
            if fill and i == 1:
                ax.fill_between(xs, 0, y,
                                alpha=0.08, color=fc)

        if sig == "steer":
            ax.axhline(0, color="#999999", lw=0.9,
                       linestyle="--", alpha=0.7)
            ax.fill_between(
                np.arange(len(dfs[0])),
                -0.15, 0.15,
                alpha=0.06, color="#999999",
                label="Stable zone"
            )

        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(
            plt.MaxNLocator(5, prune="both")
        )

        # Episode boundary lines
        for df in dfs[1:]:
            pass  # all same length here

        if si == 0:
            ax.legend(loc="upper right", ncol=3,
                      frameon=True, fontsize=9)
            ax.set_title(
                "SAC Agent — Control Signal Profiles\n"
                "Light Traffic · Smoothing Window = 30 steps",
                pad=12,
            )

    axes[-1].set_xlabel("Timestep", labelpad=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_control_signals.pdf")
    fig.savefig(path)
    plt.close()
    print(f"[Fig 2] Saved: {path}")


# ==========================================================
# FIGURE 3 — Speed Profile
# ==========================================================
def plot_speed_profile(dfs):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    speed_limit = 50.0  # km/h
    ax.axhline(speed_limit, color="#F44336",
               lw=1.2, linestyle="--", alpha=0.7,
               label=f"Speed limit ({speed_limit:.0f} km/h)")

    for i, df in enumerate(dfs):
        spd = smooth(df["speed_kmh"].values, w=20)
        xs  = np.arange(len(spd))
        ax.plot(xs, spd, color=EP_COLORS[i],
                alpha=ALPHA_LINE, lw=1.8,
                label=f"{EP_LABELS[i]} "
                      f"(avg={df['speed_kmh'].mean():.1f} km/h)")
        ax.fill_between(xs, 0, spd,
                        alpha=0.06, color=EP_COLORS[i])

    ax.set_xlabel("Timestep", labelpad=8)
    ax.set_ylabel("Speed (km/h)", labelpad=8)
    ax.set_title(
        "SAC Agent — Speed Profile\n"
        "Light Traffic · Town03",
        pad=12,
    )
    ax.set_ylim(0, 60)
    ax.legend(loc="lower right", frameon=True)

    # Annotate avg speed region
    avg_spd = np.mean([df["speed_kmh"].mean() for df in dfs])
    ax.axhline(avg_spd, color="#4CAF50", lw=1.2,
               linestyle=":", alpha=0.8,
               label=f"Mean speed ({avg_spd:.1f} km/h)")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_speed_profile.pdf")
    fig.savefig(path)
    plt.close()
    print(f"[Fig 3] Saved: {path}")


# ==========================================================
# FIGURE 4 — Cumulative Reward
# ==========================================================
def plot_reward_curve(dfs):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, df in enumerate(dfs):
        r  = df["ep_reward"].values
        xs = np.arange(len(r))
        final = df["ep_reward"].iloc[-1]
        ax.plot(xs, r, color=EP_COLORS[i],
                alpha=ALPHA_LINE, lw=2.0,
                label=f"{EP_LABELS[i]} "
                      f"(final R = {final:.0f})")
        ax.fill_between(xs, 0, r,
                        alpha=0.07, color=EP_COLORS[i])

        # Mark final reward
        ax.annotate(
            f"{final:.0f}",
            xy=(xs[-1], r[-1]),
            xytext=(-45, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=EP_COLORS[i],
            fontweight="bold",
        )

    ax.set_xlabel("Timestep", labelpad=8)
    ax.set_ylabel("Cumulative Episode Reward", labelpad=8)
    ax.set_title(
        "SAC Agent — Cumulative Reward Progression\n"
        "Light Traffic · Town03",
        pad=12,
    )
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_reward_curve.pdf")
    fig.savefig(path)
    plt.close()
    print(f"[Fig 4] Saved: {path}")


# ==========================================================
# FIGURE 5 — Combined (2×2 panel) — BEST FOR PAPER
# ==========================================================
def plot_combined(dfs):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.38,
        wspace=0.28,
    )

    ax_traj  = fig.add_subplot(gs[0, 0])
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_ctrl  = fig.add_subplot(gs[1, 0])
    ax_rew   = fig.add_subplot(gs[1, 1])

    # ── Trajectory ──────────────────────────────────────
    for i, df in enumerate(dfs):
        x = df["x"].values
        y = df["y"].values
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segs   = np.concatenate(
            [points[:-1], points[1:]], axis=1
        )
        lc = LineCollection(
            segs,
            cmap="Blues" if i==0 else
                 "RdPu"  if i==1 else "Greens",
            linewidth=1.8, alpha=0.85, zorder=3,
        )
        lc.set_array(np.linspace(0.3, 1.0, len(segs)))
        ax_traj.add_collection(lc)
        ax_traj.scatter(
            x[0], y[0], color=EP_COLORS[i],
            s=90, zorder=6, marker="o",
            edgecolors="white", linewidths=1.2,
            label=f"Ep{i+1} (R={df['ep_reward'].iloc[-1]:.0f})"
        )
        ax_traj.scatter(
            x[-1], y[-1], color=EP_COLORS[i],
            s=110, zorder=6, marker="*",
            edgecolors="white", linewidths=0.8,
        )

    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.set_title("(a) Executed Trajectories",
                      fontweight="bold")
    ax_traj.set_aspect("equal")
    ax_traj.autoscale()
    ax_traj.legend(fontsize=8.5, loc="upper right",
                   title="● Start  ★ End",
                   title_fontsize=7.5)

    # ── Speed ────────────────────────────────────────────
    ax_speed.axhline(50, color="#F44336", lw=1.0,
                     linestyle="--", alpha=0.6,
                     label="Speed limit")
    avg_spd = np.mean([df["speed_kmh"].mean() for df in dfs])
    ax_speed.axhline(avg_spd, color="#4CAF50", lw=1.0,
                     linestyle=":", alpha=0.8,
                     label=f"Mean ({avg_spd:.1f} km/h)")

    for i, df in enumerate(dfs):
        spd = smooth(df["speed_kmh"].values, w=20)
        ax_speed.plot(spd, color=EP_COLORS[i],
                      alpha=ALPHA_LINE, lw=1.6,
                      label=f"Ep{i+1}")
        ax_speed.fill_between(
            np.arange(len(spd)), 0, spd,
            alpha=0.05, color=EP_COLORS[i]
        )

    ax_speed.set_xlabel("Timestep")
    ax_speed.set_ylabel("Speed (km/h)")
    ax_speed.set_title("(b) Speed Profile",
                       fontweight="bold")
    ax_speed.set_ylim(0, 60)
    ax_speed.legend(fontsize=8, loc="lower right",
                    ncol=2)

    # ── Control signals (throttle + steer) ───────────────
    ax_ctrl2 = ax_ctrl.twinx()

    df = dfs[1]  # best episode
    thr = smooth(df["throttle"].values, w=25)
    ste = smooth(df["steer"].values,    w=25)
    brk = smooth(df["brake"].values,    w=25)
    xs  = np.arange(len(thr))

    l1, = ax_ctrl.plot(xs, thr, color="#2196F3",
                       lw=1.8, alpha=0.9,
                       label="Throttle")
    ax_ctrl.fill_between(xs, 0, thr,
                         alpha=0.07, color="#2196F3")
    l2, = ax_ctrl.plot(xs, brk, color="#FF5722",
                       lw=1.5, alpha=0.8,
                       label="Brake")
    l3, = ax_ctrl2.plot(xs, ste, color="#9C27B0",
                        lw=1.5, alpha=0.8,
                        linestyle="--",
                        label="Steering")
    ax_ctrl2.axhline(0, color="#999999", lw=0.7,
                     linestyle=":", alpha=0.5)

    ax_ctrl.set_xlabel("Timestep")
    ax_ctrl.set_ylabel("Throttle / Brake [0,1]")
    ax_ctrl2.set_ylabel("Steering [-1,1]",
                         color="#9C27B0")
    ax_ctrl2.tick_params(colors="#9C27B0")
    ax_ctrl.set_title("(c) Control Signals — Best Episode",
                      fontweight="bold")
    ax_ctrl.set_ylim(-0.05, 1.1)
    ax_ctrl2.set_ylim(-1.1, 1.1)
    lines = [l1, l2, l3]
    ax_ctrl.legend(lines, [l.get_label() for l in lines],
                   fontsize=8.5, loc="upper right")

    # ── Reward ───────────────────────────────────────────
    for i, df in enumerate(dfs):
        r  = df["ep_reward"].values
        xs = np.arange(len(r))
        final = df["ep_reward"].iloc[-1]
        ax_rew.plot(xs, r, color=EP_COLORS[i],
                    alpha=ALPHA_LINE, lw=1.8,
                    label=f"Ep{i+1}  R={final:.0f}")
        ax_rew.fill_between(xs, 0, r,
                             alpha=0.06,
                             color=EP_COLORS[i])

    ax_rew.set_xlabel("Timestep")
    ax_rew.set_ylabel("Cumulative Reward")
    ax_rew.set_title("(d) Cumulative Reward",
                     fontweight="bold")
    ax_rew.legend(fontsize=8.5, loc="upper left")

    # Main title
    fig.suptitle(
        "SAC Agent Evaluation — Town03, Light Traffic "
        "(10 NPCs, 3 Episodes)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    path = os.path.join(OUT_DIR, "fig5_combined.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Fig 5] Saved: {path}")

    # Also save PNG for quick viewing
    fig2 = plt.figure(figsize=(14, 10))
    gs2  = gridspec.GridSpec(2, 2, figure=fig2,
                              hspace=0.38, wspace=0.28)
    # (rerun same code for PNG)
    ax_t = fig2.add_subplot(gs2[0, 0])
    ax_s = fig2.add_subplot(gs2[0, 1])
    ax_c = fig2.add_subplot(gs2[1, 0])
    ax_r = fig2.add_subplot(gs2[1, 1])

    for i, df in enumerate(dfs):
        x = df["x"].values; y = df["y"].values
        pts = np.array([x,y]).T.reshape(-1,1,2)
        segs = np.concatenate([pts[:-1],pts[1:]],axis=1)
        lc = LineCollection(segs,
            cmap="Blues" if i==0 else "RdPu" if i==1 else "Greens",
            linewidth=1.8, alpha=0.85, zorder=3)
        lc.set_array(np.linspace(0.3,1.0,len(segs)))
        ax_t.add_collection(lc)
        ax_t.scatter(x[0],y[0],color=EP_COLORS[i],s=90,
            zorder=6,marker="o",edgecolors="white",lw=1.2,
            label=f"Ep{i+1} (R={df['ep_reward'].iloc[-1]:.0f})")
        ax_t.scatter(x[-1],y[-1],color=EP_COLORS[i],
            s=110,zorder=6,marker="*",edgecolors="white",lw=0.8)
    ax_t.set_xlabel("X (m)"); ax_t.set_ylabel("Y (m)")
    ax_t.set_title("(a) Trajectories",fontweight="bold")
    ax_t.set_aspect("equal"); ax_t.autoscale()
    ax_t.legend(fontsize=8,loc="upper right",
        title="● Start  ★ End",title_fontsize=7)

    ax_s.axhline(50,color="#F44336",lw=1,ls="--",alpha=0.6,label="Limit")
    ax_s.axhline(avg_spd,color="#4CAF50",lw=1,ls=":",alpha=0.8,
        label=f"Mean {avg_spd:.1f}")
    for i,df in enumerate(dfs):
        spd=smooth(df["speed_kmh"].values,w=20)
        ax_s.plot(spd,color=EP_COLORS[i],alpha=ALPHA_LINE,lw=1.6,
            label=f"Ep{i+1}")
        ax_s.fill_between(np.arange(len(spd)),0,spd,alpha=0.05,
            color=EP_COLORS[i])
    ax_s.set_xlabel("Timestep"); ax_s.set_ylabel("Speed (km/h)")
    ax_s.set_title("(b) Speed Profile",fontweight="bold")
    ax_s.set_ylim(0,60); ax_s.legend(fontsize=8,loc="lower right",ncol=2)

    df=dfs[1]
    thr=smooth(df["throttle"].values,w=25)
    ste=smooth(df["steer"].values,w=25)
    brk=smooth(df["brake"].values,w=25)
    xs=np.arange(len(thr))
    ax_c2=ax_c.twinx()
    l1,=ax_c.plot(xs,thr,color="#2196F3",lw=1.8,alpha=0.9,label="Throttle")
    ax_c.fill_between(xs,0,thr,alpha=0.07,color="#2196F3")
    l2,=ax_c.plot(xs,brk,color="#FF5722",lw=1.5,alpha=0.8,label="Brake")
    l3,=ax_c2.plot(xs,ste,color="#9C27B0",lw=1.5,alpha=0.8,ls="--",
        label="Steer")
    ax_c2.axhline(0,color="#999",lw=0.7,ls=":",alpha=0.5)
    ax_c.set_xlabel("Timestep")
    ax_c.set_ylabel("Throttle/Brake")
    ax_c2.set_ylabel("Steering",color="#9C27B0")
    ax_c2.tick_params(colors="#9C27B0")
    ax_c.set_title("(c) Control Signals",fontweight="bold")
    ax_c.set_ylim(-0.05,1.1); ax_c2.set_ylim(-1.1,1.1)
    ax_c.legend([l1,l2,l3],[l.get_label() for l in [l1,l2,l3]],
        fontsize=8,loc="upper right")

    for i,df in enumerate(dfs):
        r=df["ep_reward"].values; xs=np.arange(len(r))
        final=df["ep_reward"].iloc[-1]
        ax_r.plot(xs,r,color=EP_COLORS[i],alpha=ALPHA_LINE,lw=1.8,
            label=f"Ep{i+1}  R={final:.0f}")
        ax_r.fill_between(xs,0,r,alpha=0.06,color=EP_COLORS[i])
    ax_r.set_xlabel("Timestep"); ax_r.set_ylabel("Cumulative Reward")
    ax_r.set_title("(d) Cumulative Reward",fontweight="bold")
    ax_r.legend(fontsize=8.5,loc="upper left")

    fig2.suptitle(
        "SAC Agent Evaluation — Town03, Light Traffic (10 NPCs, 3 Episodes)",
        fontsize=13,fontweight="bold",y=1.01)

    png_path = os.path.join(OUT_DIR, "fig5_combined.png")
    fig2.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[Fig 5] PNG: {png_path}")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("Loading data...")
    dfs = load_episodes()
    if not dfs:
        print("ERROR: No CSV files found in", DATA_DIR)
        print("Make sure results/sac_light/ep01.csv etc. exist")
        exit(1)

    print(f"\nGenerating graphs from {len(dfs)} episodes...")
    plot_trajectory(dfs)
    plot_control_signals(dfs)
    plot_speed_profile(dfs)
    plot_reward_curve(dfs)
    plot_combined(dfs)

    print(f"\n✅ All figures saved to: {OUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(
            os.path.join(OUT_DIR, f)
        ) / 1024
        print(f"  {f}  ({size:.0f} KB)")