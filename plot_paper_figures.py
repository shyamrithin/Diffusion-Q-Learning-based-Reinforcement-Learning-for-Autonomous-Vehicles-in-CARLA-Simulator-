# ==========================================================
# plot_paper_figures.py
# Generate all paper figures from evaluation CSVs
#
# Figures generated:
#   Figure 1: Path comparison (3×3 grid)
#   Figure 2: Control signals (heavy traffic)
#   Figure 3: Noise robustness line plot
#   Figure 4: Slow traffic scenario
#
# Usage:
#   python3 plot_paper_figures.py
#
# Requires eval_results/ directory with all CSVs
# ==========================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ==========================================================
# STYLE — IEEE paper quality
# ==========================================================
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 10,
    "axes.titlesize"   : 10,
    "axes.labelsize"   : 9,
    "xtick.labelsize"  : 8,
    "ytick.labelsize"  : 8,
    "legend.fontsize"  : 8,
    "figure.dpi"       : 300,
    "lines.linewidth"  : 1.5,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# Algorithm colors — consistent across all figures
COLORS = {
    "DQL-E" : "#2196F3",   # Blue
    "SAC"   : "#4CAF50",   # Green
    "PPO"   : "#F44336",   # Red
}

OUTPUT_DIR = "paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# DATA LOADERS
# ==========================================================
def load_csv(algo, scenario, run_id=1):
    """Load per-step CSV for given algo/scenario."""
    algo_dir = algo.lower().replace("-", "")
    path = os.path.join(
        "eval_results", algo_dir, scenario,
        f"{scenario}_run{run_id}.csv"
    )
    if not os.path.exists(path):
        print(f"[Warning] Missing: {path}")
        return None
    return pd.read_csv(path)


def load_summary(algo, scenario):
    """Load summary JSON for given algo/scenario."""
    algo_dir = algo.lower().replace("-", "")
    path = os.path.join(
        "eval_results", algo_dir, scenario,
        f"{scenario}_summary.json"
    )
    if not os.path.exists(path):
        print(f"[Warning] Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_noise_summary(algo, sigma):
    """Load noise evaluation summary."""
    algo_dir = algo.lower().replace("-", "")
    tag  = f"medium_noise{sigma}"
    path = os.path.join(
        "eval_results", algo_dir, "noise",
        f"{tag}_summary.json"
    )
    if not os.path.exists(path):
        tag  = f"noise_{sigma}"
        path = os.path.join(
            "eval_results", algo_dir, "noise",
            f"{tag}_summary.json"
        )
    if not os.path.exists(path):
        print(f"[Warning] Missing noise: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def average_runs(algo, scenario, n_runs=3):
    """Average position data across N runs."""
    dfs = []
    for run_id in range(1, n_runs + 1):
        df = load_csv(algo, scenario, run_id)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    min_len = min(len(df) for df in dfs)
    arr_x   = np.array([
        df["x"].values[:min_len] for df in dfs
    ])
    arr_y   = np.array([
        df["y"].values[:min_len] for df in dfs
    ])
    gt_x    = dfs[0]["gt_x"].values[:min_len]
    gt_y    = dfs[0]["gt_y"].values[:min_len]
    return {
        "x"    : arr_x.mean(axis=0),
        "y"    : arr_y.mean(axis=0),
        "gt_x" : gt_x,
        "gt_y" : gt_y,
        "dfs"  : dfs,
    }


# ==========================================================
# FIGURE 1 — Path Comparison (3×3 grid)
# ==========================================================
def plot_path_comparison():
    """
    3×3 grid showing actual path vs GT waypoints.
    Rows: light, medium, heavy traffic
    Cols: DQL-E, SAC, PPO
    """
    fig, axes = plt.subplots(
        3, 3, figsize=(7.0, 7.0),
        sharex=True, sharey=True
    )
    fig.suptitle(
        "Executed Path vs Ground Truth Waypoints",
        fontsize=11, fontweight="bold", y=1.01
    )

    scenarios = ["light", "medium", "heavy"]
    algos     = ["DQL-E", "SAC", "PPO"]
    row_labels= [
        "Light Traffic\n(10 vehicles)",
        "Medium Traffic\n(30 vehicles)",
        "Heavy Traffic\n(60 vehicles)",
    ]

    for row, scenario in enumerate(scenarios):
        for col, algo in enumerate(algos):
            ax   = axes[row][col]
            data = average_runs(algo, scenario)

            if data is not None:
                # Ground truth path
                ax.plot(
                    data["gt_x"], data["gt_y"],
                    color     = "black",
                    linewidth = 1.0,
                    linestyle = "--",
                    alpha     = 0.6,
                    label     = "GT Waypoints",
                    zorder    = 2,
                )
                # Actual path
                ax.plot(
                    data["x"], data["y"],
                    color     = COLORS[algo],
                    linewidth = 1.8,
                    alpha     = 0.85,
                    label     = algo,
                    zorder    = 3,
                )
                # Start marker
                ax.scatter(
                    data["gt_x"][0],
                    data["gt_y"][0],
                    color  = "green",
                    marker = "o",
                    s      = 60,
                    zorder = 5,
                    label  = "Start",
                )
                # End marker
                ax.scatter(
                    data["gt_x"][-1],
                    data["gt_y"][-1],
                    color  = "red",
                    marker = "x",
                    s      = 60,
                    zorder = 5,
                    label  = "End",
                )
            else:
                ax.text(
                    0.5, 0.5, "No Data",
                    transform = ax.transAxes,
                    ha        = "center",
                    va        = "center",
                    color     = "gray",
                )

            # Column headers (top row only)
            if row == 0:
                ax.set_title(
                    algo,
                    fontsize   = 10,
                    fontweight = "bold",
                    color      = COLORS[algo],
                )

            # Row labels (left col only)
            if col == 0:
                ax.set_ylabel(
                    row_labels[row],
                    fontsize = 8,
                )

            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)

    # Shared legend
    handles = [
        mpatches.Patch(
            color="black", label="GT Waypoints",
            linestyle="--", fill=False
        ),
        plt.Line2D(
            [0], [0], color=COLORS["DQL-E"],
            linewidth=2, label="DQL-E"
        ),
        plt.Line2D(
            [0], [0], color=COLORS["SAC"],
            linewidth=2, label="SAC"
        ),
        plt.Line2D(
            [0], [0], color=COLORS["PPO"],
            linewidth=2, label="PPO"
        ),
        plt.Line2D(
            [0], [0], marker="o", color="green",
            linestyle="None", markersize=6,
            label="Start"
        ),
        plt.Line2D(
            [0], [0], marker="x", color="red",
            linestyle="None", markersize=6,
            label="End"
        ),
    ]
    fig.legend(
        handles    = handles,
        loc        = "lower center",
        ncol       = 6,
        fontsize   = 8,
        frameon    = True,
        bbox_to_anchor = (0.5, -0.04),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(OUTPUT_DIR, "fig1_paths.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Figure 1] Saved: {path}")


# ==========================================================
# FIGURE 2 — Control Signals (heavy traffic)
# ==========================================================
def plot_control_signals():
    """
    Control signals over time for heavy traffic.
    3 subplots: throttle, steer, brake
    3 lines per subplot: DQL-E, SAC, PPO
    Uses run 1 only (representative run).
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(7.0, 5.5), sharex=True
    )
    fig.suptitle(
        "Control Signals — Heavy Traffic Scenario",
        fontsize=11, fontweight="bold"
    )

    signals   = ["throttle", "steer", "brake"]
    ylabels   = ["Throttle", "Steering", "Brake"]
    ylims     = [(0, 1.05), (-1.05, 1.05), (0, 1.05)]
    algos     = ["DQL-E", "SAC", "PPO"]

    # Smooth window for readability
    def smooth(y, w=15):
        return np.convolve(
            y, np.ones(w)/w, mode="valid"
        )

    for sig_idx, (sig, ylabel, ylim) in enumerate(
        zip(signals, ylabels, ylims)
    ):
        ax = axes[sig_idx]

        for algo in algos:
            df = load_csv(algo, "heavy", run_id=1)
            if df is None:
                continue
            y = smooth(df[sig].values)
            x = np.arange(len(y))
            ax.plot(
                x, y,
                color     = COLORS[algo],
                label     = algo,
                linewidth = 1.5,
                alpha     = 0.85,
            )

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(ylim)

        if sig == "steer":
            ax.axhline(
                0, color="gray",
                linewidth=0.8, linestyle="--",
                alpha=0.5
            )

        if sig_idx == 0:
            ax.legend(
                loc="upper right", fontsize=8
            )

    axes[-1].set_xlabel("Timestep", fontsize=9)

    plt.tight_layout()
    path = os.path.join(
        OUTPUT_DIR, "fig2_control_signals.pdf"
    )
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Figure 2] Saved: {path}")


# ==========================================================
# FIGURE 3 — Noise Robustness
# ==========================================================
def plot_noise_robustness():
    """
    Waypoint deviation vs noise sigma.
    3 lines: DQL-E, SAC, PPO
    X: noise sigma
    Y: avg waypoint deviation (metres)
    """
    from noise_eval import NOISE_SIGMAS

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    algos = ["DQL-E", "SAC", "PPO"]

    for algo in algos:
        deviations = []
        for sigma in NOISE_SIGMAS:
            summary = load_noise_summary(algo, sigma)
            if summary:
                deviations.append(
                    summary["avg_wp_deviation_m"]
                )
            else:
                deviations.append(np.nan)

        ax.plot(
            NOISE_SIGMAS,
            deviations,
            color     = COLORS[algo],
            marker    = "o",
            markersize= 5,
            linewidth = 1.8,
            label     = algo,
        )

    ax.set_xlabel(
        "Gaussian Noise σ (LiDAR + IMU)",
        fontsize=9
    )
    ax.set_ylabel(
        "Mean Waypoint Deviation (m)",
        fontsize=9
    )
    ax.set_title(
        "Sensor Noise Robustness",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=8)
    ax.set_xticks(NOISE_SIGMAS)

    plt.tight_layout()
    path = os.path.join(
        OUTPUT_DIR, "fig3_noise_robustness.pdf"
    )
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Figure 3] Saved: {path}")


# ==========================================================
# FIGURE 4 — Slow Traffic Scenario
# ==========================================================
def plot_slow_traffic():
    """
    Path comparison in slow surrounding traffic.
    Side by side: DQL-E, SAC, PPO
    Shows NPC positions schematically.
    """
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 3.0),
        sharex=True, sharey=True
    )
    fig.suptitle(
        "Special Case: Ego Surrounded by"
        " Slow-Moving Traffic (20 km/h)",
        fontsize=10, fontweight="bold"
    )

    algos = ["DQL-E", "SAC", "PPO"]

    for col, algo in enumerate(algos):
        ax      = axes[col]
        algo_dir= algo.lower().replace("-", "")
        path    = os.path.join(
            "eval_results", algo_dir, "slow",
            "slow_traffic_run1.csv"
        )

        if os.path.exists(path):
            df = pd.read_csv(path)
            ax.plot(
                df["gt_x"], df["gt_y"],
                color     = "black",
                linestyle = "--",
                linewidth = 1.0,
                alpha     = 0.5,
                label     = "GT",
            )
            ax.plot(
                df["x"], df["y"],
                color     = COLORS[algo],
                linewidth = 2.0,
                label     = algo,
            )

            # Mark collision events
            collisions = df[df["collision"] == 1]
            if len(collisions) > 0:
                ax.scatter(
                    collisions["x"],
                    collisions["y"],
                    color  = "red",
                    marker = "x",
                    s      = 50,
                    zorder = 5,
                    label  = "Collision",
                )

            # Start/end
            ax.scatter(
                df["gt_x"].iloc[0],
                df["gt_y"].iloc[0],
                color="green", marker="o",
                s=60, zorder=6
            )
            ax.scatter(
                df["gt_x"].iloc[-1],
                df["gt_y"].iloc[-1],
                color="red", marker="s",
                s=60, zorder=6
            )
        else:
            ax.text(
                0.5, 0.5, "No Data",
                transform=ax.transAxes,
                ha="center", va="center",
                color="gray"
            )

        ax.set_title(
            algo,
            fontsize   = 10,
            fontweight = "bold",
            color      = COLORS[algo],
        )
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(
        OUTPUT_DIR, "fig4_slow_traffic.pdf"
    )
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Figure 4] Saved: {path}")


# ==========================================================
# RESULT TABLES
# ==========================================================
def generate_result_tables():
    """
    Print LaTeX-ready result tables
    for copy-pasting into paper.
    """
    algos     = ["DQL-E", "SAC", "PPO"]
    scenarios = ["light", "medium", "heavy"]

    print("\n" + "="*60)
    print("TABLE 1 — Main Quantitative Results")
    print("="*60)

    header = (
        f"{'Scenario':<10} {'Algorithm':<8} "
        f"{'Route%':>7} {'WPDev(m)':>9} "
        f"{'Speed':>6} {'Coll':>5} "
        f"{'|Steer|':>8} {'Brake':>6}"
    )
    print(header)
    print("-" * len(header))

    for scenario in scenarios:
        for algo in algos:
            s = load_summary(algo, scenario)
            if s:
                print(
                    f"{scenario:<10} {algo:<8} "
                    f"{s['route_complete_pct']:>6.1f}% "
                    f"{s['avg_wp_deviation_m']:>9.3f} "
                    f"{s['avg_speed_kmh']:>6.1f} "
                    f"{s['total_collisions']:>5} "
                    f"{s['avg_abs_steer']:>8.3f} "
                    f"{s['avg_brake']:>6.3f}"
                )
            else:
                print(
                    f"{scenario:<10} {algo:<8} "
                    f"{'N/A':>6} {'N/A':>9} "
                    f"{'N/A':>6} {'N/A':>5} "
                    f"{'N/A':>8} {'N/A':>6}"
                )
        print()

    print("\n" + "="*60)
    print("TABLE 2 — Noise Robustness")
    print("="*60)

    sigmas = [0.0, 0.01, 0.05, 0.1]
    header2 = (
        f"{'Algorithm':<8} "
        + " ".join([
            f"σ={s:>5}" for s in sigmas
        ])
    )
    print(header2)
    print("-" * len(header2))

    for algo in algos:
        row = f"{algo:<8} "
        for sigma in sigmas:
            s = load_noise_summary(algo, sigma)
            if s:
                row += (
                    f"{s['avg_wp_deviation_m']:>7.3f} "
                )
            else:
                row += f"{'N/A':>7} "
        print(row)

    print("\n" + "="*60)
    print("TABLE 3 — Slow Traffic Special Case")
    print("="*60)

    header3 = (
        f"{'Algorithm':<8} {'Coll':>6} "
        f"{'Route%':>7} {'Speed':>6} "
        f"{'WPDev(m)':>9}"
    )
    print(header3)
    print("-" * len(header3))

    for algo in algos:
        algo_dir = algo.lower().replace("-", "")
        path = os.path.join(
            "eval_results", algo_dir, "slow",
            "slow_traffic_summary.json"
        )
        if os.path.exists(path):
            with open(path) as f:
                s = json.load(f)
            print(
                f"{algo:<8} "
                f"{s['total_collisions']:>6} "
                f"{s['route_complete_pct']:>6.1f}% "
                f"{s['avg_speed_kmh']:>6.1f} "
                f"{s['avg_wp_deviation_m']:>9.3f}"
            )
        else:
            print(
                f"{algo:<8} "
                f"{'N/A':>6} {'N/A':>7} "
                f"{'N/A':>6} {'N/A':>9}"
            )


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("Generating paper figures...")
    print(f"Output directory: {OUTPUT_DIR}/\n")

    plot_path_comparison()
    plot_control_signals()

    try:
        plot_noise_robustness()
    except Exception as e:
        print(f"[Figure 3] Skipped: {e}")

    plot_slow_traffic()
    generate_result_tables()

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")