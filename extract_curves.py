# ==========================================================
# extract_curves.py
# Extract Section A training curves from the three TensorBoard
# runs into results/training_curves.json, which plot_section_a.py
# (and fig_section_ae.py) then consume.
#
# Reads scalar tags from:
#   runs/rlcarla_v14   -> dqle   (DQL-E, ours)
#   runs/rlcarla_sac   -> sac
#   runs/rlcarla_ppo   -> ppo
#
# Tags pulled (whichever exist per run):
#   reward/episode, loss/critic, env/episode_length,
#   debug/entropy, debug/alpha
#
# Output JSON shape:
#   { "dqle": { "reward/episode": [[step,val],...], ... },
#     "sac":  {...}, "ppo": {...} }
#
# Usage:  python3 extract_curves.py
#         (run from ~/Carla/RLCarla)
# Requires: tensorboard (EventAccumulator).
#
# ----------------------------------------------------------
# CODE DESCRIPTION
#   A training run that was stopped and resumed leaves SEVERAL
#   `events.out.tfevents.*` files in one run folder. There are
#   two distinct cases, and they need opposite handling:
#
#     (1) SEQUENTIAL resume  (e.g. v14: file A = ep 0..1802,
#         file B = ep 1803..2000). Here we must CONCATENATE the
#         files to recover the full-length curve. Picking only
#         one file truncates the run (this is what wrongly cut
#         DQL-E's late climb toward SAC).
#
#     (2) DUPLICATE restart  (e.g. ppo: two files BOTH covering
#         ep 0..1299). Here concatenation alone would double
#         every step and plot overlapping lines.
#
#   This script handles BOTH by merging every event file in the
#   directory and then de-duplicating by step, keeping the LAST
#   value written for any repeated step (later writes win, which
#   is correct for a resumed/overwritten step). Sequential files
#   keep their full combined range; duplicate files collapse to a
#   single clean series. No agent is truncated, and no curve is
#   doubled.
# ==========================================================
import os
import json
import glob
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator

RUN_MAP = {
    "dqle": "runs/rlcarla_v14",
    "sac":  "runs/rlcarla_sac",
    "ppo":  "runs/rlcarla_ppo",
}
TAGS = [
    "reward/episode",
    "loss/critic",
    "env/episode_length",
    "debug/entropy",
    "debug/alpha",
]


def event_files(run_dir):
    """All TensorBoard event files directly in run_dir (sorted, oldest
    first by mtime so concatenation is in chronological/resume order).
    Anything inside an _archived/ subfolder is ignored."""
    fs = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    fs = [f for f in fs if os.path.isfile(f)]
    fs.sort(key=os.path.getmtime)        # chronological resume order
    return fs


def merged_scalars(ev_files, tag):
    """Merge `tag` across all event files, de-duplicating by step.

    Files are read oldest-first; for any repeated step the LAST value
    seen (i.e. from the newer file) wins. Returns [[step, value], ...]
    sorted by step. Sequential resumes keep their full range; duplicate
    restarts collapse to one clean series.
    """
    by_step = {}
    present = False
    for ev in ev_files:
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            continue
        present = True
        for e in acc.Scalars(tag):
            by_step[int(e.step)] = float(e.value)
    if not present:
        return None
    return [[s, by_step[s]] for s in sorted(by_step)]


def all_tags(ev_files):
    tags = set()
    for ev in ev_files:
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        tags |= set(acc.Tags().get("scalars", []))
    return tags


os.makedirs("results", exist_ok=True)
out = {}
for agent, run_dir in RUN_MAP.items():
    if not os.path.isdir(run_dir):
        print(f"  !! {agent}: '{run_dir}' not found — skipping")
        continue

    ev_files = event_files(run_dir)
    if not ev_files:
        print(f"  !! {agent}: no event files in '{run_dir}' — skipping")
        continue

    print(f"\n{agent} ({run_dir})")
    if len(ev_files) > 1:
        print(f"  merging {len(ev_files)} event files "
              f"(concatenate + dedup by step):")
        for f in ev_files:
            print(f"    {os.path.basename(f)}")
    else:
        print(f"  event file: {os.path.basename(ev_files[0])}")

    available = all_tags(ev_files)
    out[agent] = {}
    for tag in TAGS:
        series = merged_scalars(ev_files, tag)
        if series is None:
            print(f"  (missing {tag})")
            continue
        steps = [p[0] for p in series]
        dup = len(steps) != len(set(steps))      # should be impossible now
        out[agent][tag] = series
        print(f"  extracted {tag}: {len(series)} points "
              f"(steps {steps[0]}..{steps[-1]})"
              + ("  !! DUPLICATES" if dup else ""))

with open("results/training_curves.json", "w") as f:
    json.dump(out, f)
print("\nWrote results/training_curves.json")
print("Agents:", list(out.keys()))
