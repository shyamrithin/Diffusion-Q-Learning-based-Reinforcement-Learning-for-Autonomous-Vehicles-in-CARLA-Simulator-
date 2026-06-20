# Verify: does ego-LANE-direction alignment reject the turned-into
# light? Print, per step, ego lane dir vs the associated light's
# stop-wp dir, and whether the new rule would accept it.
import math, sys
import gymnasium as gym, rlcarla, numpy as np, carla

SPAWN = int(sys.argv[1]) if len(sys.argv) > 1 else 237
ALIGN_MAX = 45.0
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
    lane_yaw = ewp.transform.rotation.yaw
    lr = math.radians(lane_yaw); lfx,lfy=math.cos(lr),math.sin(lr)
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
            swyaw=wp.transform.rotation.yaw
            rel=((swyaw-lane_yaw+180)%360)-180
            wl=wp.transform.location
            dx,dy=wl.x-loc.x,wl.y-loc.y
            ahead=dx*lfx+dy*lfy
            dist=math.hypot(dx,dy)
            accept = (abs(rel)<=ALIGN_MAX and ahead>-3.0)
            if best is None or dist<best[3]:
                best=(tl.id,str(tl.get_state()),rel,dist,ahead,accept)
    return best, ewp.road_id, ewp.lane_id

last=None
for step in range(400):
    env.step(np.array([0.45,0.0,0.0],dtype=np.float32))
    r=assoc()
    if r is None: continue
    a,road,lane=r
    if a is None: continue
    tid,st,rel,dist,ahead,acc=a
    key=(tid,acc)
    if key!=last:
        print(f"step {step:3d} road={road} lane={lane} light {tid} {st:6} "
              f"lane_rel={rel:+4.0f} ahead={ahead:+5.1f} ACCEPT={acc}")
        last=key
print("\nWant: legit approaches ACCEPT=True, turned-into ACCEPT=False")
env.close()