# ==========================================================
# check_metrics.py
# Quick honest metrics summary for the current eval batch,
# per agent x route x traffic. Computes the rates that matter
# for Section B BEFORE deciding whether a re-run is needed.
#
# Reads results/<agent>_<traffic>/<route>_summary.csv (per-ep).
# Columns (0-idx): 1=route 7=term_reason 8=success 9=completion
#   10=collision 11=offroad ... (matches ALL_summary header)
#
# Reports per (agent, route, traffic):
#   n, reached_dest%, collision%, offroad%, stuck%,
#   mean_completion, mean_reward
# And a SAFE% = episodes with no collision AND no offroad.
#
# Usage:  python3 check_metrics.py   (from ~/Carla/RLCarla)
# ==========================================================

import os, csv, glob
from collections import defaultdict

AGENTS = ["dqle", "sac", "ppo"]
DENS = ["empty", "light", "medium", "heavy"]
ROUTES = ["route_1_roundabout", "route_2_curve", "route_3_straight"]

def load_rows(agent, traffic):
    # prefer ALL_summary; fall back to per-route summaries
    rows = []
    allp = f"results/{agent}_{traffic}/ALL_summary.csv"
    if os.path.isfile(allp):
        with open(allp) as f:
            r = list(csv.DictReader(f))
        if r:
            return r
    # fallback: stitch per-route
    for rt in ROUTES:
        p = f"results/{agent}_{traffic}/{rt}_summary.csv"
        if os.path.isfile(p):
            with open(p) as f:
                rows.extend(list(csv.DictReader(f)))
    return rows

def pct(num, den):
    return f"{100.0*num/den:4.0f}%" if den else "  - "

print(f"{'agent':5} {'route':18} {'traf':7} {'n':>2} "
      f"{'dest%':>5} {'coll%':>5} {'off%':>5} {'stuck%':>6} "
      f"{'comp':>5} {'reward':>8}")
print("-"*78)

for agent in AGENTS:
    for rt in ROUTES:
        for traffic in DENS:
            rows = [r for r in load_rows(agent, traffic)
                    if r.get("route") == rt]
            n = len(rows)
            if n == 0:
                continue
            dest = sum(1 for r in rows if r.get("term_reason")=="reached_dest")
            coll = sum(1 for r in rows if r.get("collision","0") in ("1","1.0","True"))
            off  = sum(1 for r in rows if r.get("offroad","0") in ("1","1.0","True"))
            stuck= sum(1 for r in rows if r.get("term_reason")=="stuck")
            comp = sum(float(r.get("route_completion",0) or 0) for r in rows)/n
            rew  = sum(float(r.get("reward",0) or 0) for r in rows)/n
            print(f"{agent:5} {rt:18} {traffic:7} {n:>2} "
                  f"{pct(dest,n):>5} {pct(coll,n):>5} {pct(off,n):>5} "
                  f"{pct(stuck,n):>6} {comp:5.2f} {rew:8.1f}")
        print()

# --- aggregate SAFE% by density (collision-free AND offroad-free) ---
print("="*78)
print("SAFE% by density (no collision AND no offroad), all routes pooled:")
print(f"{'agent':5} " + "".join(f"{d:>9}" for d in DENS))
for agent in AGENTS:
    cells = []
    for traffic in DENS:
        rows = load_rows(agent, traffic)
        n = len(rows)
        safe = sum(1 for r in rows
                   if r.get("collision","0") in ("0","0.0","False")
                   and r.get("offroad","0") in ("0","0.0","False"))
        cells.append(pct(safe, n))
    print(f"{agent:5} " + "".join(f"{c:>9}" for c in cells))