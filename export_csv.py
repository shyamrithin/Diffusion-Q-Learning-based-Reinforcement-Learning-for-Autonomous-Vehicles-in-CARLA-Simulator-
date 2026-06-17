# ==========================================================
# export_csv.py
# Export raw per-episode training metrics (and the Table III
# summary) from results/training_curves.json to CSV.
#
# Layout: ONE CSV PER AGENT, metric columns.
#   results/csv/dqle.csv  results/csv/sac.csv  results/csv/ppo.csv
#   results/csv/table3_summary.csv
#
# Per-agent file columns (whichever the agent logged):
#   step, reward_episode, critic_loss, episode_length,
#   entropy, alpha
# Rows are aligned on the union of all logged steps for that
# agent; a metric that was not logged at a given step is left
# BLANK (e.g. DQL-E entropy is blank before ~ep450; SAC/PPO
# have no entropy/alpha at all). This is honest about exactly
# what each run recorded -- nothing is interpolated or invented.
#
# table3_summary.csv columns:
#   method, final_reward_mean, critic_loss_mean_raw,
#   critic_loss_std_raw, critic_loss_mean_excl,
#   critic_loss_std_excl, n_transients_excl,
#   episode_length_mean, reward_std
#   (means over the final 200 episodes; *_excl excludes the
#    transient critic-loss episodes > 10x median, disclosed.)
#
# Usage:  python3 export_csv.py     (run from ~/Carla/RLCarla)
# ==========================================================

import os
import csv
import json
import numpy as np

TAG_TO_COL = {
    "reward/episode":     "reward_episode",
    "loss/critic":        "critic_loss",
    "env/episode_length": "episode_length",
    "debug/entropy":      "entropy",
    "debug/alpha":        "alpha",
}
COL_ORDER = ["step", "reward_episode", "critic_loss",
             "episode_length", "entropy", "alpha"]

OUTLIER_FACTOR = 10.0
TAIL_N = 200

OUTDIR = "results/csv"
os.makedirs(OUTDIR, exist_ok=True)

with open("results/training_curves.json") as f:
    data = json.load(f)


def per_agent_csv(agent, adata):
    # Build step -> {col: value} from every available tag
    by_step = {}
    cols_present = ["step"]
    for tag, col in TAG_TO_COL.items():
        if tag in adata and adata[tag]:
            if col not in cols_present:
                cols_present.append(col)
            for step, val in adata[tag]:
                by_step.setdefault(int(step), {})[col] = val

    if not by_step:
        print(f"  {agent}: no data, skipped")
        return

    # Column set in canonical order, restricted to present ones
    cols = [c for c in COL_ORDER if c in cols_present]
    steps_sorted = sorted(by_step.keys())

    path = os.path.join(OUTDIR, f"{agent}.csv")
    with open(path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(cols)
        for step in steps_sorted:
            row = []
            for c in cols:
                if c == "step":
                    row.append(step)
                else:
                    v = by_step[step].get(c, "")  # blank if not logged
                    row.append(f"{v:.6f}" if v != "" else "")
            w.writerow(row)
    print(f"  saved {path}  ({len(steps_sorted)} rows, "
          f"cols: {', '.join(cols[1:])})")


def tail(adata, tag, n=TAIL_N):
    if tag not in adata or not adata[tag]:
        return None
    v = np.array([p[1] for p in adata[tag]], dtype=float)
    return v[-n:] if len(v) >= n else v


def summary_row(method, adata):
    r = tail(adata, "reward/episode")
    c = tail(adata, "loss/critic")
    el = tail(adata, "env/episode_length")

    final_reward = f"{np.mean(r):.4f}" if r is not None else ""
    reward_std = f"{np.std(r):.4f}" if r is not None else ""
    ep_len = f"{np.mean(el):.4f}" if el is not None else ""

    if c is not None:
        c_mean_raw = np.mean(c)
        c_std_raw = np.std(c)
        med = np.median(c)
        keep = c[c < OUTLIER_FACTOR * med]
        n_excl = int(len(c) - len(keep))
        c_mean_ex = np.mean(keep)
        c_std_ex = np.std(keep)
        return [method, final_reward,
                f"{c_mean_raw:.4f}", f"{c_std_raw:.4f}",
                f"{c_mean_ex:.4f}", f"{c_std_ex:.4f}",
                n_excl, ep_len, reward_std]
    else:
        return [method, final_reward, "", "", "", "", "", ep_len, reward_std]


def table3_csv():
    path = os.path.join(OUTDIR, "table3_summary.csv")
    header = ["method", "final_reward_mean",
              "critic_loss_mean_raw", "critic_loss_std_raw",
              "critic_loss_mean_excl", "critic_loss_std_excl",
              "n_transients_excl",
              "episode_length_mean", "reward_std"]
    name = {"dqle": "DQL-E", "sac": "SAC", "ppo": "PPO"}
    with open(path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(header)
        for agent in ["ppo", "sac", "dqle"]:
            if agent in data:
                w.writerow(summary_row(name.get(agent, agent),
                                       data[agent]))
    print(f"  saved {path}")


def main():
    print("Exporting raw per-episode CSVs (one per agent) ...")
    for agent in ["dqle", "sac", "ppo"]:
        if agent in data:
            per_agent_csv(agent, data[agent])
    print("Exporting Table III summary CSV ...")
    table3_csv()
    print(f"\nDone. CSVs in '{OUTDIR}/'.")


if __name__ == "__main__":
    main()