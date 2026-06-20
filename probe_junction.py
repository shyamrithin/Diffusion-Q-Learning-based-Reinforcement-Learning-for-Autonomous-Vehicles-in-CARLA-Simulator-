# Probe: confirm waypoint.is_junction flips True inside intersections,
# so we can suppress light-gating while the ego is in the junction box
# (fixes the right-turn "obeys cross-street red mid-turn" bug).
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
cmap = env.unwrapped.world.get_map()

last = None
for step in range(400):
    env.step(np.array([0.45,0.0,0.0],dtype=np.float32))
    wp = cmap.get_waypoint(veh.get_transform().location,
                           project_to_road=True,
                           lane_type=carla.LaneType.Driving)
    key = (wp.is_junction, wp.road_id, wp.lane_id)
    if key != last:
        print(f"step {step:3d} is_junction={wp.is_junction} "
              f"road={wp.road_id} lane={wp.lane_id}")
        last = key
print("\nWatch is_junction flip True as the car enters an intersection.")
print("If it does, we suppress light gating while True.")
env.close()