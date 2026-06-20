# Probe v3: drive forward and, at each junction with lights ahead,
# print rel_y (lateral offset) and rel_yaw so we can set the corridor
# to keep OUR lane light but drop side/opposite lights.
import math, sys
import gymnasium as gym, rlcarla, numpy as np, carla

SPAWN = int(sys.argv[1]) if len(sys.argv) > 1 else 237

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
print(f"spawn {SPAWN}\n")

def lights_ahead():
    tf = veh.get_transform(); loc = tf.location
    ey = tf.rotation.yaw
    yaw = math.radians(ey)
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
        out.append((tl.id, dist, ahead, rel_y, rel_yaw, str(tl.get_state())))
    return out, ey

reported = 0
for step in range(500):
    env.step(np.array([0.45,0.0,0.0],dtype=np.float32))
    ls, ey = lights_ahead()
    if len(ls) >= 1 and reported < 5:
        print(f"--- step {step}, ego_yaw={ey:.0f}, {len(ls)} lights ---")
        print(f"{'id':>4} {'dist':>5} {'ahead':>6} {'rel_y':>7} {'rel_yaw':>7} {'state':>7}")
        for tid,dist,ahead,rel_y,rel_yaw,st in sorted(ls,key=lambda r:r[1]):
            print(f"{tid:>4} {dist:>5.1f} {ahead:>6.1f} {rel_y:>7.1f} {rel_yaw:>7.0f} {st:>7}")
        print()
        reported += 1
    if reported >= 5:
        break
print("OUR lane light = the one ahead in-window. Note its rel_y and")
print("rel_yaw vs the side lights, so we set the corridor between them.")
env.close()