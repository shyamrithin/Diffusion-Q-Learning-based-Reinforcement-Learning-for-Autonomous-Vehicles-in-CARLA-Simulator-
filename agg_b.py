# ==========================================================
# agg_b.py
# Aggregate the clean Section B (traffic-density) eval batch
# into the paper's Table IV + the honest per-route breakdown.
#
# Reads results/<agent>_<traffic>/ALL_summary.csv (n=3 per cell,
# headless deterministic batch).
#
# Honest scoring split (decided with the data, documented here):
#   R1 roundabout : SAFETY metric (collision/offroad rate).
#                   Agents legally diverge to different exits,
#                   so completion is NOT comparable; safety is.
#   R2 curve      : COMPLETION + reward. Single sweeping arc
#                   (spawn 170 -> dest 45, ~90deg, no junctions),
#                   fair direct comparison of curve-following.
#   R3 straight   : COMPLETION. Single forced path, fair.
#   Pooled SAFE%  : episodes with no collision AND no offroad,
#                   across all routes, by density.
#
# n=3 per cell: rates are reported as small-sample (e.g. "0/3").
# Outputs:
#   results/section_b/table_iv.csv          (paper table)
#   results/section_b/per_route_breakdown.csv
#   prints both to stdout
#
# Usage:  python3 agg_b.py    (from ~/Carla/RLCarla)
# ==========================================================

import os, csv, glob
from collections import defaultdict

AGENTS = ["dqle", "sac", "ppo"]
DENS = ["empty", "light", "medium", "heavy"]
ROUTES = ["route_1_roundabout", "route_2_curve", "route_3_straight"]
NAME = {"dqle": "DQL-E", "sac": "SAC", "ppo": "PPO"}

OUTDIR = "results/section_b"
os.makedirs(OUTDIR, exist_ok=True)


def truthy(v):
    return str(v).strip() in ("1", "1.0", "True", "true")


def load(agent, traffic):
    p = f"results/{agent}_{traffic}/ALL_summary.csv"
    if not os.path.isfile(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def rows_for(agent, traffic, route):
    return [r for r in load(agent, traffic) if r.get("route") == route]


def rate(rows, pred):
    n = len(rows)
    k = sum(1 for r in rows if pred(r))
    return k, n


# ---------- per-route breakdown ----------
print("PER-ROUTE BREAKDOWN (n=3 per cell)\n" + "="*70)
breakdown = []
for agent in AGENTS:
    for route in ROUTES:
        for traffic in DENS:
            rows = rows_for(agent, traffic, route)
            if not rows:
                continue
            n = len(rows)
            coll_k, _ = rate(rows, lambda r: truthy(r.get("collision")))
            off_k, _ = rate(rows, lambda r: truthy(r.get("offroad")))
            dest_k, _ = rate(rows, lambda r: r.get("term_reason") == "reached_dest")
            comp = sum(float(r.get("route_completion", 0) or 0) for r in rows)/n
            rew = sum(float(r.get("reward", 0) or 0) for r in rows)/n
            breakdown.append([NAME[agent], route, traffic, n,
                              f"{coll_k}/{n}", f"{off_k}/{n}",
                              f"{dest_k}/{n}", f"{comp:.2f}", f"{rew:.1f}"])

hdr = ["agent", "route", "traffic", "n", "collision", "offroad",
       "reached_dest", "mean_completion", "mean_reward"]
print("{:6} {:18} {:7} {:>2} {:>9} {:>8} {:>12} {:>8} {:>9}".format(*hdr))
for r in breakdown:
    print("{:6} {:18} {:7} {:>2} {:>9} {:>8} {:>12} {:>8} {:>9}".format(*r))

with open(os.path.join(OUTDIR, "per_route_breakdown.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(breakdown)


# ---------- Table IV: the paper table ----------
# Per agent x density: R1 collision-rate, R2 completion, R3 completion,
# pooled safe-rate.
print("\n\nTABLE IV — Driving Performance by Traffic Density (n=3/cell)\n" + "="*70)

def pooled_safe(agent, traffic):
    rows = load(agent, traffic)
    n = len(rows)
    k = sum(1 for r in rows
            if not truthy(r.get("collision")) and not truthy(r.get("offroad")))
    return k, n

table = []
for agent in AGENTS:
    for traffic in DENS:
        r1 = rows_for(agent, traffic, "route_1_roundabout")
        r2 = rows_for(agent, traffic, "route_2_curve")
        r3 = rows_for(agent, traffic, "route_3_straight")
        r1_coll_k, r1_n = rate(r1, lambda r: truthy(r.get("collision")))
        r2_comp = (sum(float(r.get("route_completion",0) or 0) for r in r2)/len(r2)) if r2 else 0
        r3_comp = (sum(float(r.get("route_completion",0) or 0) for r in r3)/len(r3)) if r3 else 0
        safe_k, safe_n = pooled_safe(agent, traffic)
        table.append([NAME[agent], traffic,
                      f"{r1_coll_k}/{r1_n}",
                      f"{r2_comp*100:.0f}%",
                      f"{r3_comp*100:.0f}%",
                      f"{safe_k}/{safe_n}",
                      f"{100*safe_k/safe_n:.0f}%" if safe_n else "-"])

thdr = ["Agent", "Density", "R1 collisions", "R2 compl.",
        "R3 compl.", "Safe (n)", "Safe%"]
print("{:6} {:8} {:>13} {:>9} {:>9} {:>9} {:>6}".format(*thdr))
for r in table:
    print("{:6} {:8} {:>13} {:>9} {:>9} {:>9} {:>6}".format(*r))

with open(os.path.join(OUTDIR, "table_iv.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(thdr); w.writerows(table)

print(f"\nSaved: {OUTDIR}/table_iv.csv  and  {OUTDIR}/per_route_breakdown.csv")