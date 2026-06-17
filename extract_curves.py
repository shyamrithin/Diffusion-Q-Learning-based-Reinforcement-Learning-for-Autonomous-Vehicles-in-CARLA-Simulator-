# ==========================================================
# extract_curves.py
# Extract Section A training curves from the three TensorBoard
# runs into results/training_curves.json, which plot_section_a.py
# then consumes.
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
# Requires: tensorboard (EventAccumulator). Already present in
# the training env since training wrote these logs.
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

os.makedirs("results", exist_ok=True)
out = {}

for agent, run_dir in RUN_MAP.items():
    if not os.path.isdir(run_dir):
        print(f"  !! {agent}: '{run_dir}' not found — skipping")
        continue

    # EventAccumulator loads the largest in a dir; load with
    # generous size guidance so nothing is downsampled away.
    acc = EventAccumulator(
        run_dir,
        size_guidance={"scalars": 0},  # 0 = load all
    )
    acc.Reload()

    available = set(acc.Tags().get("scalars", []))
    print(f"\n{agent} ({run_dir})")
    print(f"  available scalar tags ({len(available)}):")
    for t in sorted(available):
        print(f"    {t}")

    out[agent] = {}
    for tag in TAGS:
        if tag in available:
            events = acc.Scalars(tag)
            out[agent][tag] = [[int(e.step), float(e.value)]
                               for e in events]
            print(f"  extracted {tag}: {len(events)} points")
        else:
            print(f"  (missing {tag})")

with open("results/training_curves.json", "w") as f:
    json.dump(out, f)

print("\nWrote results/training_curves.json")
print("Agents:", list(out.keys()))