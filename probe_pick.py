# Probe v4: at every junction with >=2 lights ahead, show ALL
# candidates with rel_y AND rel_yaw, and mark which one the current
# filter logic would PICK. Lets us see the mispick directly.
import math, sys
import gymnasium as gym, rlcarla, numpy as np, carla

SPAWN = int(sys.argv[1]) if len(sys.argv) > 1 else 237
FACING_MIN = 135.0   # current threshold
LAT_MAX    = 12.0    # current (too wide) threshold

env = gym.make("RLCarla-v0")
env.unwrapped._traffic_preset = "empty"
env.unwrapped.cfg.TRAFFIC_LIGHTS = "normal"
env.unwrapped.cfg.STUCK_LIMIT = 100000
env.unwrapped.cfg.MAX_STEPS = 100000
sps = env.unwrapped.carla_map.get_spawn_points()
env.unwrapped.set_eval_spawn(sps[SPAWN])
env.reset()
veh = env.unwrapped.vehicle
world = env.unwrapped.world
lights = list(world.get_actors().filter("traffic.traffic_light*"))
print(f"spawn {SPAWN}  (FACING_MIN={FACING_MIN}, LAT_MAX={LAT_MAX})\n")

def candidates():
    tf = veh.get_transform(); loc = tf.location
    ey = tf.rotation.yaw; yaw = math.radians(ey)
    fx, fy = math.cos(yaw), math.sin(yaw)
    out = []
    for tl in lights:
        tloc = tl.get_transform().location
        tyaw = tl.get_transform().rotation.yaw
        dx, dy = tloc.x-loc.x, tloc.y-loc.y
        dist = math.hypot(dx,dy)
        if dist > 45: continue
        ahead = dx*fx + dy*fy
        if ahead <= 0: continue
        rel_y = -fy*dx + fx*dy
        rel_yaw = ((tyaw - ey + 180) % 360) - 180
        passes = (abs(rel_yaw) >= FACING_MIN and abs(rel_y) <= LAT_MAX)
        out.append((tl.id, dist, rel_y, rel_yaw, str(tl.get_state()), passes))
    return out

reported = 0
for step in range(2500):
    env.step(np.array([0.45,0.0,0.0],dtype=np.float32))
    c = candidates()
    if len(c) >= 2 and reported < 6:
        # which would we PICK? nearest among those that pass
        passing = [x for x in c if x[5]]
        pick = min(passing, key=lambda r:r[1])[0] if passing else None
        print(f"--- step {step}: pick={pick} ---")
        print(f"{'id':>4} {'dist':>5} {'rel_y':>7} {'rel_yaw':>7} {'state':>7} {'pass':>5}")
        for tid,dist,rel_y,rel_yaw,st,p in sorted(c,key=lambda r:r[1]):
            mark = " <-PICK" if tid==pick else ""
            print(f"{tid:>4} {dist:>5.1f} {rel_y:>7.1f} {rel_yaw:>7.0f} {st:>7} {str(p):>5}{mark}")
        print()
        reported += 1
    if reported >= 6:
        break
print("Find a step where PICK is a RED side light but a GREEN light")
print("is the real lane one. Compare their rel_yaw to set the fix.")
env.close()