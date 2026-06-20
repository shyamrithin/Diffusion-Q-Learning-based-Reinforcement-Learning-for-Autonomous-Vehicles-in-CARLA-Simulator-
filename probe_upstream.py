# Probe: for the lane-associated light, measure where the EGO sits
# relative to the stop line, IN THE STOP LINE'S OWN FRAME.
#  s = +ve  -> ego is UPSTREAM of the stop line (legitimately
#              approaching it; should obey)
#  s = -ve  -> ego is AT/PAST the stop line (entered from the side /
#              turning in; should NOT obey)
# This is heading-independent, so DQL-E steer noise is irrelevant.
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
    loc = veh.get_transform().location
    best=None
    for tl in lights:
        try: wps = tl.get_affected_lane_waypoints()
        except Exception: wps=[]
        if not wps:
            try: wps=tl.get_stop_waypoints()
            except Exception: wps=[]
        for wp in wps:
            if wp.road_id!=ewp.road_id or wp.lane_id!=ewp.lane_id: continue
            # stop line frame: forward = stop wp's yaw direction
            sl = wp.transform.location
            syaw = math.radians(wp.transform.rotation.yaw)
            sfx, sfy = math.cos(syaw), math.sin(syaw)
            # vector from stop line to ego, projected on stop fwd dir.
            # ego upstream means ego is BEHIND the line along travel,
            # i.e. (ego-stop) . fwd < 0  -> so upstream s = -that
            ex, ey = loc.x-sl.x, loc.y-sl.y
            along = ex*sfx + ey*sfy
            s = -along   # +ve = upstream (approaching)
            dist = math.hypot(ex,ey)
            if best is None or dist<best[3]:
                best=(tl.id,str(tl.get_state()),s,dist)
    return best, ewp.road_id, ewp.lane_id

last=None
for step in range(400):
    env.step(np.array([0.45,0.0,0.0],dtype=np.float32))
    r=assoc()
    if r is None: continue
    a,road,lane=r
    if a is None: continue
    tid,st,s,dist=a
    key=(tid, round(s/2))
    if key!=last:
        tag = "UPSTREAM(obey)" if s>1.0 else "AT/PAST(ignore)"
        print(f"step {step:3d} road={road} lane={lane} light {tid} {st:6} "
              f"s={s:+5.1f} dist={dist:4.1f}  {tag}")
        last=key
print("\nLegit approach: s large +ve, shrinking as you near line.")
print("Turned-into: s near 0 or -ve (you're at/past the line).")
env.close()