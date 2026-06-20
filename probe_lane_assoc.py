# Probe: test CARLA's lane-based light association.
# For the ego's current lane, find which traffic light governs it via
# get_affected_lane_waypoints() / get_stop_waypoints(), and compare to
# the naive nearest-ahead pick. This validates whether Path B works in
# this CARLA build before we wire it into the wrapper.
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
veh = env.unwrapped.world.get_actors().filter("vehicle.*")
veh = env.unwrapped.vehicle
world = env.unwrapped.world
cmap = world.get_map()
lights = list(world.get_actors().filter("traffic.traffic_light*"))
print(f"spawn {SPAWN}\n")

# 1) does the API exist?
tl0 = lights[0]
print("=== API availability ===")
for meth in ["get_affected_lane_waypoints", "get_stop_waypoints",
             "get_opendrive_id", "get_pole_index", "get_group_traffic_lights"]:
    print(f"  {meth}: {'yes' if hasattr(tl0, meth) else 'NO'}")
print()

def ego_lane():
    wp = cmap.get_waypoint(veh.get_transform().location,
                           project_to_road=True,
                           lane_type=carla.LaneType.Driving)
    return wp

def lane_assoc_pick():
    """Return the light whose affected/stop waypoints lie on the ego's
    lane, nearest ahead."""
    ewp = ego_lane()
    if ewp is None:
        return None, None, float("inf")
    e_road, e_lane = ewp.road_id, ewp.lane_id
    tf = veh.get_transform(); loc = tf.location
    yaw = math.radians(tf.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    best = (None, None, float("inf"))
    for tl in lights:
        try:
            wps = tl.get_affected_lane_waypoints()
        except Exception:
            wps = []
        if not wps:
            try:
                wps = tl.get_stop_waypoints()
            except Exception:
                wps = []
        for wp in wps:
            # is this governed lane the ego's lane?
            same = (wp.road_id == e_road and wp.lane_id == e_lane)
            if not same:
                continue
            wl = wp.transform.location
            dx, dy = wl.x-loc.x, wl.y-loc.y
            ahead = dx*fx + dy*fy
            dist = math.hypot(dx,dy)
            if ahead > 0 and dist < best[2]:
                best = (tl.id, str(tl.get_state()), dist)
    return best

# drive forward, report lane-assoc pick vs ego lane each junction
print("=== driving; lane-assoc light pick per step ===")
last = None
for step in range(400):
    env.step(np.array([0.45,0.0,0.0],dtype=np.float32))
    ewp = ego_lane()
    pid, pstate, pdist = lane_assoc_pick()
    key = (ewp.road_id, ewp.lane_id, pid)
    if key != last and pid is not None:
        print(f"step {step:3d} ego_lane(road={ewp.road_id},lane={ewp.lane_id}) "
              f"-> light {pid} {pstate} @ {pdist:.0f}m")
        last = key
print("\nIf each junction maps the ego's lane to ONE sensible light")
print("matching what you'd see ahead, Path B works -> wire it in.")
env.close()