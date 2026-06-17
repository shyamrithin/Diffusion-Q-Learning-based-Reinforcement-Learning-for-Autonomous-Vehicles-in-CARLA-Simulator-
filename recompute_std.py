# ==========================================================
# recompute_std.py
# Recompute DQL-E critic-loss statistics over the final 200
# episodes with the single-episode ep1800 transient excluded,
# so Table III can report a disclosed, outlier-robust std
# alongside the as-is std. Also reports SAC for comparison.
#
# Usage:  python3 recompute_std.py   (run from ~/Carla/RLCarla)
# ==========================================================

import numpy as np
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator

def critic_tail(run, n=200):
    acc = EventAccumulator(run, size_guidance={"scalars": 0})
    acc.Reload()
    ev = acc.Scalars("loss/critic")
    steps = np.array([e.step for e in ev])
    vals  = np.array([e.value for e in ev])
    return steps[-n:], vals[-n:]

# DQL-E
ds, dv = critic_tail("runs/rlcarla_v14")
print("DQL-E final-200 critic loss:")
print(f"  mean       = {dv.mean():.1f}")
print(f"  std (as-is)= {dv.std():.1f}")
print(f"  max        = {dv.max():.1f} at step {int(ds[np.argmax(dv)])}")

# Exclude the single transient (anything > 10x the median)
med = np.median(dv)
keep = dv[dv < 10*med]
removed = len(dv) - len(keep)
print(f"  median     = {med:.1f}")
print(f"  excluded {removed} outlier point(s) > 10x median:")
print(f"  mean (excl)= {keep.mean():.1f}")
print(f"  std  (excl)= {keep.std():.1f}")

# SAC for comparison
ss, sv = critic_tail("runs/rlcarla_sac")
print("\nSAC final-200 critic loss:")
print(f"  mean       = {sv.mean():.1f}")
print(f"  std        = {sv.std():.1f}")

print("\n--- Table III stability line (disclosed-excluded) ---")
print(f"DQL-E: mean {keep.mean():.1f}, std {keep.std():.1f} "
      f"(excl. 1 transient ep)")
print(f"SAC  : mean {sv.mean():.1f}, std {sv.std():.1f}")