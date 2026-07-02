# ============================================================================
# tb_to_csv.py
# ----------------------------------------------------------------------------
# Description:
#   Extracts all scalar curves from TensorBoard event files into per-tag CSVs.
#   Built for the DQL-E Section D ablation handoff: pulls reward, critic-loss,
#   entropy (and any other logged scalars) from the ablation runs plus the
#   v14 / v15b reference runs, so the curves can be moved between machines/chats
#   as plain CSV instead of raw event blobs.
#
#   For each run directory, writes one CSV per scalar tag:
#       <out_dir>/<run_name>__<sanitized_tag>.csv   (columns: step,wall_time,value)
#   and prints the list of available tags per run.
#
# Usage:
#   python tb_to_csv.py                      # uses the default RUN_DIRS below
#   python tb_to_csv.py runs/foo runs/bar    # or pass run dirs explicitly
#
# Env: conda diffusioncarla, Python 3.10. Requires tensorboard (already
#      installed since training logs to it).
# ============================================================================

import os
import sys
import csv
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- Default runs to extract. Edit if your v14/v15b dirs are named differently.
RUN_DIRS = [
    "runs/ablate_no_eta",
    "runs/ablate_no_pretrain",
    # The two reference/free-row runs — adjust these names to match your disk:
    # "runs/v14",
    # "runs/v15b",
]

OUT_DIR = "ablation_csv"


def sanitize(tag: str) -> str:
    """Make a TB tag safe for a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_")


def extract_run(run_dir: str, out_dir: str) -> None:
    if not os.path.isdir(run_dir):
        print(f"  [SKIP] not a directory: {run_dir}")
        return

    run_name = sanitize(os.path.basename(run_dir.rstrip("/")))
    print(f"\n=== {run_dir}  ->  run_name='{run_name}' ===")

    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})  # 0 = load all
    acc.Reload()

    tags = acc.Tags().get("scalars", [])
    if not tags:
        print("  [WARN] no scalar tags found in this run.")
        return

    print(f"  scalar tags ({len(tags)}):")
    for t in tags:
        print(f"    - {t}")

    for tag in tags:
        events = acc.Scalars(tag)
        csv_path = os.path.join(out_dir, f"{run_name}__{sanitize(tag)}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "wall_time", "value"])
            for e in events:
                w.writerow([e.step, e.wall_time, e.value])
        print(f"      wrote {len(events):5d} rows -> {csv_path}")


def main() -> None:
    run_dirs = sys.argv[1:] if len(sys.argv) > 1 else RUN_DIRS
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Output dir: {OUT_DIR}")
    print(f"Runs to process: {run_dirs}")

    for run_dir in run_dirs:
        extract_run(run_dir, OUT_DIR)

    print(f"\nDone. CSVs in ./{OUT_DIR}/")
    print("Zip them with:  zip -r ablation_csv.zip ablation_csv/")


if __name__ == "__main__":
    main()
