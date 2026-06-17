# ==========================================================
# find_curve.py  (v2 — reuses route_utils, no import hacks)
# Evaluate candidate spawn->dest pairs for R2 and report which
# trace a genuine LONG SWEEPING CURVE rather than a straight
# road or an intersection / sharp-turn path.
#
# Uses route_utils.generate_reference_route(), which already
# fixes the CARLA `agents` shadowing problem and plans routes
# the SAME way record_eval.py does. So these measurements match
# what the eval will actually drive.
#
# For each candidate, reports over the planned reference path:
#   len(m)   total path length
#   wpts     number of 2m-sampled waypoints
#   meanC    mean heading-change per step (deg) -- curvature
#   maxC     max single-step heading change (deg) -- sharpness
#   totTurn  total accumulated heading change (deg) -- does it
#            actually curve overall
#   sharp    # of steps with >25 deg change (intersection/hard
#            turn proxy)
#   verdict  SWEEPING CURVE / too straight / sharp-junction-heavy
#
# Good sweeping curve = decent length, meaningful total turn
# (>45 deg), but FEW sharp points (smooth, not hard turns).
#
# Usage:  python3 find_curve.py    (CARLA server must be up)
# ==========================================================

import math
import carla
import route_utils  # provides generate_reference_route + planner fix

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
cmap = world.get_map()

SAMPLE_RES = 2.0

# Candidate (spawn, dest) pairs to test as the new R2.
# Mix of outer-ring corners + current R2 baseline for reference.
CANDIDATES = [
    # Outer-ring sweeping-corner candidates (read off the map).
    # SW rounded corner (left edge -> bottom):
    (170, 45), (169, 45), (4, 116), (3, 115), (170, 116),
    # NW rounded corner (left -> top):
    (159, 34), (154, 166), (156, 33),
    # bottom edge sweep:
    (45, 199), (116, 117), (45, 117),
    # left edge long sweep:
    (170, 4), (169, 3),
    (107, 170),  # CURRENT R2 baseline
]


def analyse(spawn_i, dest_i):
    try:
        ref_xy, _ = route_utils.generate_reference_route(
            world, cmap, spawn_i, dest_i, SAMPLE_RES
        )
    except Exception as e:
        return None, f"plan failed: {e}"
    if ref_xy is None or len(ref_xy) < 3:
        return None, "route too short"

    xs = [p[0] for p in ref_xy]
    ys = [p[1] for p in ref_xy]

    # length
    length = 0.0
    for i in range(1, len(ref_xy)):
        length += math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])

    # heading per segment, then change-of-heading per step
    headings = [math.atan2(ys[i]-ys[i-1], xs[i]-xs[i-1])
                for i in range(1, len(ref_xy))]
    dthe = []
    for i in range(1, len(headings)):
        dh = headings[i] - headings[i-1]
        while dh > math.pi: dh -= 2*math.pi
        while dh < -math.pi: dh += 2*math.pi
        dthe.append(abs(dh))
    if not dthe:
        return None, "no heading data"

    mean_curv = sum(dthe)/len(dthe)
    max_curv = max(dthe)
    total_turn = sum(dthe)
    sharp = sum(1 for x in dthe if x > math.radians(25))

    return {
        "len": length, "n": len(ref_xy),
        "mean_curv_deg": math.degrees(mean_curv),
        "max_curv_deg": math.degrees(max_curv),
        "total_turn_deg": math.degrees(total_turn),
        "sharp": sharp,
    }, None


print(f"{'pair':>12} | {'len(m)':>7} | {'wpts':>5} | {'meanC':>6} | "
      f"{'maxC':>6} | {'totTurn':>7} | {'sharp':>5} | verdict")
print("-"*92)
for s_i, d_i in CANDIDATES:
    r, err = analyse(s_i, d_i)
    tag = f"{s_i}->{d_i}"
    if err:
        print(f"{tag:>12} | {err}")
        continue
    # TRUE sweeping curve: bends a lot OVERALL (totTurn high) but
    # SMOOTHLY (maxC low -> no hard 90deg corners; few sharp pts).
    smooth = r["max_curv_deg"] <= 35 and r["sharp"] <= 2
    curves = r["total_turn_deg"] > 45
    long_enough = r["len"] > 60
    if not curves:
        verdict = "too straight"
    elif not smooth:
        verdict = "sharp/junction-heavy"
    elif long_enough:
        verdict = "SWEEPING CURVE <<<"
    else:
        verdict = "smooth but short"
    print(f"{tag:>12} | {r['len']:7.1f} | {r['n']:5d} | "
          f"{r['mean_curv_deg']:6.2f} | {r['max_curv_deg']:6.1f} | "
          f"{r['total_turn_deg']:7.1f} | {r['sharp']:5d} | {verdict}")