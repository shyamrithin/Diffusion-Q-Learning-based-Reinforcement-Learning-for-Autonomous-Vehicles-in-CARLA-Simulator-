# Probe: for the lane-associated light, show the stop-waypoint's
# 'ahead' distance and the heading alignment between ego and the
# stop-waypoint. Goal: find a rule that keeps a real head-on approach
# but rejects a light we've just turned into (stop-line beside/behind,
# or facing crosswise).
import math, sys
import gymnasium as gym, rlcarla, numpy as np, carla

SPAWN = int(sys.argv[1]) if len(sys.argv) > 1 else 237
env = gym.make("RLCarla-v0")
env.unwrapped._traffic_preset = "empty"
env.unwrapped.cfg.TRAFFIC_LIGHTS = "normal"
env.unwrapped.cfg.STUCK_LIMIT = 100000
env.unwrapped.cfg.MAX_STEPS = 100000
sps = env.unwrapped.carla_map.get_spawn_points()
env.unwrapped.set_eval_spawn(sps[SPAWN]); env.reset()
veh = env.unwrapped.vehicle
world = env.unwrapped.world
cmap = world.get_map()
lights = list(world.get_actors().filter("traffic.traffic_light*"))

def assoc():
    ewp = cmap.get_waypoint(veh.get_transform().location,
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    if ewp is None: return None
    tf = veh.get_transform(); loc = tf.location
    yaw = tf.rotation.yaw; yr = math.radians(yaw)
    fx, fy = math.cos(yr), math.sin(yr)
    best = None
    for tl in lights:
        try: wps = tl.get_affected_lane_waypoints()
        except Exception: wps = []
        if not wps:
            try: wps = tl.get_stop_waypoints()
            except Exception: wps = []
        for wp in wps:
            if wp.road_id != ewp.road_id or wp.lane_id != ewp.lane_id:
                continue
            wl = wp.transform.location
            dx, dy = wl.x-loc.x, wl.y-loc.y
            ahead = dx*fx + dy*fy
            dist = math.hypot(dx,dy)
            # heading of stop wp vs ego
            swyaw = wp.transform.rotation.yaw
            rel = ((swyaw - yaw + 180) % 360) - 180
            if best is None or dist < best[2]:
                best = (tl.id, str(tl.get_state()), dist, ahead, rel)
    return best

last=None
for step in range(400):
    env.step(np.array([0.45,0.0,0.0],dtype=np.float32))
    a = assoc()
    if a is None: continue
    tid,st,dist,ahead,rel = a
    key=(tid,round(ahead/3))
    if key!=last:
        print(f"step {step:3d} light {tid} {st:6} dist={dist:5.1f} "
              f"ahead={ahead:+6.1f} wp_rel_yaw={rel:+4.0f}")
        last=key
print("\nReal approach: ahead>0 large, wp_rel_yaw ~0 (aligned).")
print("Just-turned-into: ahead small/neg or wp_rel_yaw far from 0.")
env.close()