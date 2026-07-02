# ============================================================================
# make_ablation_figs.py
# ----------------------------------------------------------------------------
# Description:
#   Regenerates all Section D (Ablation Study) figures for the DQL-E paper
#   directly from the TensorBoard-exported CSV curves (produced by
#   tb_to_csv.py). Every number plotted is real training data; nothing is
#   synthesised. Three figures are produced, one per ablated mechanism:
#
#     figD_eta_reward.png      — denoising-guidance (eta) schedule ablation.
#                                Episode reward over the full 2000-episode
#                                horizon: full DQL-E (v14) vs. eta held flat
#                                (no warmup ramp). Annotates the crossover
#                                where the ramp overtakes the flat variant.
#
#     figD_entropy_floor.png   — entropy-floor ablation (two panels):
#                                (a) policy entropy, (b) temperature alpha,
#                                floor ON (v15b) vs floor OFF (v15). Shows
#                                alpha decaying through the 0.25 floor and
#                                entropy collapsing when the floor is removed.
#
#     figD_critic_pretrain.png — critic pre-training ablation (two panels):
#                                (a) behaviour-cloning loss onset (actor
#                                training begins at step ~3 without pretrain
#                                vs ~439 with it — the removed mechanism made
#                                visible); (b) episode reward, showing the
#                                catastrophic late collapse when the critic
#                                is not warmed up.
#
# STYLE
#   Times New Roman (Liberation Serif is the metric-identical substitute on
#   Linux; the actual font is used automatically where installed). Plain,
#   uncoloured backgrounds; smoothed lines over faint raw traces; honest
#   framing (no inflated contrasts).
#
# INPUT
#   A directory of CSVs named "<run>__<tag>.csv" with columns
#   (step, wall_time, value). Run names are configured in RUNS below; edit
#   them if your ablation tags differ.
#
# USAGE
#   python3 make_ablation_figs.py                       # dir=ablation_csv, out=figs_D
#   python3 make_ablation_figs.py --csv_dir DIR --out_dir OUT
#
# Requires: numpy, matplotlib. Python 3.10.
# ============================================================================

import os
import csv
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Run tags (edit if your exported CSV prefixes differ)
# ----------------------------------------------------------------------------
RUNS = {
    "full":        "rlcarla_v14",          # hero model / reference baseline
    "no_eta":      "ablate_no_eta_long",   # eta held flat, 2000-ep re-run
    "no_pretrain": "ablate_no_pretrain",   # critic pre-training disabled
    "floor_on":    "rlcarla_v15b",         # entropy floor active
    "floor_off":   "rlcarla_v15",          # entropy floor removed (collapse)
}

# Palette — colour-blind-safe, consistent across the section
C_FULL   = "#1a1a1a"   # full DQL-E (black)
C_ETA    = "#c1642d"   # eta ablation (orange)
C_PRE    = "#2166ac"   # pretrain ablation (blue)
C_ON     = "#1b7837"   # floor on (green)
C_OFF    = "#b2182b"   # floor off (red)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def set_style():
    plt.rcParams["font.family"]     = "serif"
    plt.rcParams["font.serif"]      = ["Times New Roman", "Liberation Serif",
                                       "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.titlesize"]  = 13
    plt.rcParams["axes.labelsize"]  = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["axes.linewidth"]  = 0.8


def load(csv_dir, run, tag):
    """Load one (step, value) curve; returns (np.array, np.array)."""
    path = os.path.join(csv_dir, f"{run}__{tag}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    s, v = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            s.append(int(float(row["step"])))
            v.append(float(row["value"]))
    return np.array(s), np.array(v)


def smooth(v, k):
    """Trailing moving average; returns array of len(v)-k+1."""
    if len(v) < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="valid")


def bare(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ----------------------------------------------------------------------------
# Figure 1 — eta schedule ablation (reward, full horizon)
# ----------------------------------------------------------------------------
def fig_eta(csv_dir, out_dir):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    W = 40  # display smoothing window
    curves = {}
    finals = {}
    for key, col, lab in [("full", C_FULL, "Full DQL-E (eta ramp)"),
                          ("no_eta", C_ETA, "Without eta ramp (held flat)")]:
        s, v = load(csv_dir, RUNS[key], "reward_episode")
        curves[key] = (s, v)
        finals[key] = (col, float(np.mean(v[-100:])))
        ax.plot(s, v, color=col, lw=0.5, alpha=0.10)      # faint raw
        sm = smooth(v, W)
        ax.plot(s[:len(sm)], sm, color=col, lw=2.1, label=lab)

    # Crossover: last episode where no_eta(smoothed) >= full(smoothed)
    sf, vf = curves["full"]; se, ve = curves["no_eta"]
    n = min(len(vf), len(ve))
    fs = smooth(vf[:n], W); es = smooth(ve[:n], W)
    m = min(len(fs), len(es))
    ahead = es[:m] >= fs[:m]
    cross = next((i for i in range(m - 1, 0, -1)
                  if ahead[i - 1] and not ahead[i]), None)
    if cross is not None:
        xc = sf[cross]
        ax.axvline(xc, color="0.6", lw=0.9, ls="--")
        ax.text(xc - 30, 6600, f"ramp overtakes (ep ~{xc})",
                fontsize=9, color="0.35", ha="right")

    # Final-window value labels (the headline gap)
    for key in ("full", "no_eta"):
        col, val = finals[key]
        s, _ = curves[key]
        ax.annotate(f"{val:.0f}", xy=(s[-1], val),
                    xytext=(s[-1] + 25, val),
                    fontsize=10, color=col, va="center", weight="bold")

    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_ylim(-3200, 7200)
    ax.set_xlim(0, 2120)
    ax.set_title("Denoising-guidance (eta) schedule ablation")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.legend(loc="upper left", frameon=False)
    bare(ax)
    plt.tight_layout()
    p = os.path.join(out_dir, "figD_eta_reward.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    return p


# ----------------------------------------------------------------------------
# Figure 2 — entropy-floor ablation (entropy + alpha)
# ----------------------------------------------------------------------------
def fig_floor(csv_dir, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for key, col, lab in [("floor_on", C_ON, "With entropy floor"),
                          ("floor_off", C_OFF, "Without entropy floor")]:
        s, v = load(csv_dir, RUNS[key], "debug_entropy")
        ax[0].plot(s, v, color=col, lw=0.7, alpha=0.30)
        sm = smooth(v, 15)
        ax[0].plot(s[:len(sm)], sm, color=col, lw=1.9, label=lab)
        sa, va = load(csv_dir, RUNS[key], "debug_alpha")
        ax[1].plot(sa, va, color=col, lw=1.7, label=lab)

    ax[0].set_title("(a) Policy entropy")
    ax[0].set_xlabel("Update step"); ax[0].set_ylabel("Entropy")
    ax[0].legend(loc="lower left", frameon=False)

    ax[1].axhline(0.25, color="0.5", lw=0.9, ls="--")
    ax[1].text(60, 0.275, "floor = 0.25", fontsize=9, color="0.4")
    ax[1].set_title(r"(b) Temperature $\alpha$")
    ax[1].set_xlabel("Update step"); ax[1].set_ylabel(r"$\alpha$")
    ax[1].legend(loc="upper right", frameon=False)
    for a in ax:
        bare(a)
    plt.tight_layout()
    p = os.path.join(out_dir, "figD_entropy_floor.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    return p


# ----------------------------------------------------------------------------
# Figure 3 — critic pre-training ablation (bc-loss onset + reward collapse)
# ----------------------------------------------------------------------------
def fig_pretrain(csv_dir, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # (a) bc-loss onset — the removed mechanism made visible
    for key, col, lab in [("full", C_FULL, "With critic pre-training"),
                          ("no_pretrain", C_PRE, "Without critic pre-training")]:
        s, v = load(csv_dir, RUNS[key], "loss_bc")
        ax[0].plot(s, v, color=col, lw=1.4, label=lab)
        onset = s[0]
        ax[0].scatter([onset], [v[0]], color=col, s=28, zorder=5)
        ax[0].annotate(f"onset ep {onset}", xy=(onset, v[0]),
                       xytext=(onset + 60, v[0] + (0.06 if key == "full" else -0.10)),
                       fontsize=9, color=col)
    ax[0].set_xlim(-20, 700)
    ax[0].set_title("(a) Actor (BC-loss) onset")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("BC loss")
    ax[0].legend(loc="center right", frameon=False)

    # (b) reward collapse
    for key, col, lab in [("full", C_FULL, "With critic pre-training"),
                          ("no_pretrain", C_PRE, "Without critic pre-training")]:
        s, v = load(csv_dir, RUNS[key], "reward_episode")
        m = s <= 699; s, v = s[m], v[m]
        ax[1].plot(s, v, color=col, lw=0.5, alpha=0.25)
        sm = smooth(v, 15)
        ax[1].plot(s[:len(sm)], sm, color=col, lw=1.9, label=lab)
        if key == "no_pretrain":
            imin = int(np.argmin(v))
            ax[1].annotate(f"min {v[imin]:.0f}", xy=(s[imin], v[imin]),
                           xytext=(s[imin] - 300, v[imin] + 2500),
                           fontsize=9, color=C_PRE,
                           arrowprops=dict(arrowstyle="->", color=C_PRE, lw=0.8))
    ax[1].axhline(0, color="0.7", lw=0.6)
    ax[1].set_title("(b) Episode reward")
    ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Episode reward")
    ax[1].legend(loc="lower left", frameon=False)
    for a in ax:
        bare(a)
    plt.tight_layout()
    p = os.path.join(out_dir, "figD_critic_pretrain.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    return p


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="ablation_csv")
    ap.add_argument("--out_dir", default="figs_D")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_style()

    outputs = [
        fig_eta(args.csv_dir, args.out_dir),
        fig_floor(args.csv_dir, args.out_dir),
        fig_pretrain(args.csv_dir, args.out_dir),
    ]
    print("Wrote:")
    for p in outputs:
        print("  ", p)


if __name__ == "__main__":
    main()
