# ==========================================================
# diagnose_spike.py
# Diagnose the DQL-E critic-loss transient near ep ~1800.
#
# Pulls the DQL-E run's diagnostic scalars directly from the
# tensorboard log and reports what happened in the window
# around the spike, so we can attribute it to a real training
# mechanism (target reset / eta change / ql or bc loss) rather
# than guess.
#
# Reads:  runs/rlcarla_v14
# Tags :  loss/critic, debug/critic_reset, debug/eta,
#         loss/ql, loss/bc, debug/alpha, debug/entropy
#
# Usage:  python3 diagnose_spike.py     (run from ~/Carla/RLCarla)
# ==========================================================

import numpy as np
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator

RUN = "runs/rlcarla_v14"
TAGS = ["loss/critic", "debug/critic_reset", "debug/eta",
        "loss/ql", "loss/bc", "debug/alpha", "debug/entropy"]

acc = EventAccumulator(RUN, size_guidance={"scalars": 0})
acc.Reload()
avail = set(acc.Tags().get("scalars", []))

def get(tag):
    if tag not in avail:
        return None, None
    ev = acc.Scalars(tag)
    return (np.array([e.step for e in ev]),
            np.array([e.value for e in ev]))

# 1) Locate the critic-loss spike step
cs, cv = get("loss/critic")
peak_i = int(np.argmax(cv))
peak_step = int(cs[peak_i])
peak_val = float(cv[peak_i])
print(f"Critic-loss PEAK: value={peak_val:.1f} at step={peak_step}")
print(f"  (mean critic loss overall = {np.mean(cv):.1f})")

# Window around the peak
lo, hi = peak_step - 60, peak_step + 60
print(f"\nWindow [{lo}, {hi}] around the spike:")

# 2) Did a critic reset fire near the spike?
rs, rv = get("debug/critic_reset")
if rs is not None:
    # reset events: report any nonzero / changed values in window
    mask = (rs >= lo) & (rs <= hi)
    near = list(zip(rs[mask].tolist(), rv[mask].tolist()))
    print("\n[debug/critic_reset] values in window:")
    print("  ", near if near else "none logged in window")
    # also: all steps where critic_reset changed/were nonzero
    nz = [(int(s), float(v)) for s, v in zip(rs, rv) if v != 0]
    print(f"  all nonzero critic_reset events: {nz[:20]}"
          f"{' ...' if len(nz)>20 else ''}")
else:
    print("\n[debug/critic_reset] tag not present")

# 3) ETA value around the spike (schedule change?)
es, ev = get("debug/eta")
if es is not None:
    mask = (es >= lo) & (es <= hi)
    if mask.any():
        print(f"\n[debug/eta] in window: "
              f"min={ev[mask].min():.5f} max={ev[mask].max():.5f}")
    # detect a step change in eta anywhere
    deta = np.abs(np.diff(ev))
    jumps = np.where(deta > deta.mean() + 5*deta.std())[0]
    print(f"  eta step-changes at steps: "
          f"{[int(es[j]) for j in jumps][:20]}")

# 4) ql / bc loss behavior at the spike
for t in ["loss/ql", "loss/bc"]:
    s, v = get(t)
    if s is not None:
        mask = (s >= lo) & (s <= hi)
        if mask.any():
            seg = v[mask]
            print(f"\n[{t}] in window: "
                  f"min={seg.min():.2f} max={seg.max():.2f} "
                  f"mean={seg.mean():.2f}")

# 5) Recovery: how fast does critic loss return to baseline?
base = np.median(cv[(cs > peak_step-300) & (cs < peak_step-50)])
after = cv[cs > peak_step]
after_steps = cs[cs > peak_step]
recovered = after_steps[after < base*1.5]
if len(recovered):
    print(f"\nRecovery: baseline~{base:.1f}, "
          f"critic loss back under 1.5x baseline by "
          f"step={int(recovered[0])} "
          f"({int(recovered[0])-peak_step} eps after peak)")