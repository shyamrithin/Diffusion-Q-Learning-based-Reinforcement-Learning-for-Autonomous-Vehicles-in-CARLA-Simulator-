# Probe: how to manually associate a traffic light with the ego.
# Checks what geometry CARLA exposes per light so we can find the
# one the ego is approaching (ahead + aligned) and read its state.
import math
import gymnasium as gym, rlcarla, numpy as np, carla

env = gym.make("RLCarla-v0")
env.unwrapped._traffic_preset = "empty"
env.unwrapped.cfg.TRAFFIC_LIGHTS = "normal"
env.unwrapped.cfg.STUCK_LIMIT = 100000
sps = env.unwrapped.carla_map.get_spawn_points()
env.unwrapped.set_eval_spawn(sps[257])   # near light 84
env.reset()
veh = env.unwrapped.world.get_actors().filter("vehicle.*")
veh = env.unwrapped.vehicle
world = env.unwrapped.world

lights = world.get_actors().filter("traffic.traffic_light*")
print("num lights:", len(lights))

# inspect one light's available geometry
tl = lights[0]
print("\n--- one light's API ---")
print("id:", tl.id, "state:", tl.get_state())
print("transform loc:", tl.get_transform().location)
print("has get_stop_waypoints:", hasattr(tl, "get_stop_waypoints"))
print("has get_affected_lane_waypoints:",
      hasattr(tl, "get_affected_lane_waypoints"))
print("has trigger_volume:", hasattr(tl, "trigger_volume"))
try:
    tv = tl.trigger_volume
    print("trigger_volume location:", tv.location, "extent:", tv.extent)
except Exception as e:
    print("trigger_volume err:", e)

# drive forward and, each step, find nearest light AHEAD of ego
print("\n--- nearest light ahead while driving ---")
for i in range(60):
    env.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))
    tf = veh.get_transform()
    loc = tf.location
    yaw = math.radians(tf.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)   # ego forward unit vec
    best = None
    for tl in lights:
        tloc = tl.get_transform().location
        dx, dy = tloc.x - loc.x, tloc.y - loc.y
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            continue
        # projection onto forward direction (>0 = ahead)
        ahead = (dx * fx + dy * fy)
        if ahead > 0 and dist < 40:
            if best is None or dist < best[1]:
                best = (tl, dist, ahead, str(tl.get_state()))
    if i % 10 == 0 or best:
        if best:
            print(f"{i} pos=({loc.x:.1f},{loc.y:.1f}) "
                  f"nearest_ahead: light {best[0].id} dist={best[1]:.1f} "
                  f"ahead={best[2]:.1f} state={best[3]}")
        else:
            print(f"{i} pos=({loc.x:.1f},{loc.y:.1f}) no light ahead<40m")
env.close()