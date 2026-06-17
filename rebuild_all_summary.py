# ==========================================================
# rebuild_all_summary.py
# Rebuild each cell's ALL_summary.csv by concatenating the
# per-route *_summary.csv files (route_1_roundabout_summary.csv,
# route_2_curve_summary.csv, route_3_straight_summary.csv).
#
# WHY: re-running a single route with record_eval.py rebuilds
# ALL_summary.csv from only that route, dropping the others.
# The per-route summaries are untouched, so we recombine them
# here instead of re-running R1/R3 (which are already correct).
#
# Scans results/<agent>_<traffic>/ for all agent x traffic cells,
# concatenates the three per-route summaries (header once), and
# writes a fresh ALL_summary.csv per cell.
#
# Safe to run repeatedly. Only touches ALL_summary.csv files.
#
# Usage:  python3 rebuild_all_summary.py   (from ~/Carla/RLCarla)
# ==========================================================

import os
import glob

RESULTS = "results"
ROUTE_FILES = [
    "route_1_roundabout_summary.csv",
    "route_2_curve_summary.csv",
    "route_3_straight_summary.csv",
]

cells = []
for d in sorted(glob.glob(os.path.join(RESULTS, "*"))):
    if not os.path.isdir(d):
        continue
    # only agent_traffic cells (skip csv/, logs/, etc.)
    base = os.path.basename(d)
    if "_" not in base:
        continue
    if base.split("_")[0] not in ("dqle", "sac", "ppo"):
        continue
    cells.append(d)

print(f"Found {len(cells)} cell dirs")

for d in cells:
    parts = []
    header = None
    present = []
    for rf in ROUTE_FILES:
        p = os.path.join(d, rf)
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            lines = f.read().splitlines()
        if not lines:
            continue
        if header is None:
            header = lines[0]
        # append data rows (skip each file's header)
        parts.extend(lines[1:])
        present.append(rf.split("_summary")[0])

    if header is None:
        print(f"  {os.path.basename(d)}: no per-route summaries, skipped")
        continue

    out = os.path.join(d, "ALL_summary.csv")
    with open(out, "w") as f:
        f.write(header + "\n")
        for row in parts:
            if row.strip():
                f.write(row + "\n")
    print(f"  {os.path.basename(d):16s} -> ALL_summary.csv "
          f"({len(parts)} rows; routes: {', '.join(present)})")

print("\nDone. ALL_summary.csv rebuilt for all cells.")