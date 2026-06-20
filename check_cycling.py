# Check whether traffic lights actually CYCLE in "normal" mode,
# and find a spawn with a light ~25-40m ahead for a clean approach.
import math, time
import gymnasium as gym, rlcarla, numpy as np, carla

env = gym.make("RLCarla-v0")
env.unwrapped._traffic_preset = "empty"
env.unwrapped.cfg.TRAFFIC_LIGHTS = "normal"
env.unwrapped.cfg.STUCK_LIMIT = 100000
sps = env.unwrapped.carla_map.get_spawn_points()
env.unwrapped.set_eval_spawn(sps[257])
env.reset()
world = env.unwrapped.world
lights = list(world.get_actors().filter("traffic.traffic_light*"))

# 1) do lights cycle? watch one light's state over ~150 ticks
print("=== cycling check: watching light states over time ===")
watch = lights[:3]
states = {tl.id: [] for tl in watch}
for t in range(150):
    env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))  # sit
    for tl in watch:
        states[tl.id].append(str(tl.get_state())[0])  # R/G/Y
for tid, seq in states.items():
    uniq = sorted(set(seq))
    changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    print(f"light {tid}: states seen={uniq} changes={changes} "
          f"seq={''.join(seq[:60])}...")

print("\nIf changes==0 for all, lights are FROZEN (not cycling).")
print("If you see R->G->Y transitions, cycling works.\n")

# 2) try forcing a light green via API (workaround if frozen)
print("=== can we control a light directly? ===")
tl = lights[0]
try:
    tl.set_state(carla.TrafficLightState.Green)
    tl.set_green_time(8.0); tl.set_red_time(6.0); tl.set_yellow_time(2.0)
    print("set_state / set_*_time available -> we CAN script cycling")
except Exception as e:
    print("cannot control:", e)

env.close()