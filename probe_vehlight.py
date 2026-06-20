# Probe: does vehicle.get_traffic_light() / is_at_traffic_light() work
# while DRIVING on proper lanes? If CARLA attaches the governing light
# to the ego, that's the authoritative answer (handles turns natively).
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

last=None
hits=0
for step in range(400):
    env.step(np.array([0.45,0.0,0.0],dtype=np.float32))
    at = veh.is_at_traffic_light()
    tl = veh.get_traffic_light()
    tlid = tl.id if tl is not None else None
    st = str(tl.get_state()) if tl is not None else "-"
    if tl is not None: hits+=1
    key=(at, tlid, st)
    if key!=last:
        print(f"step {step:3d} is_at={at} light={tlid} state={st}")
        last=key
print(f"\nSteps with a governing light attached: {hits}/400")
print("If get_traffic_light() returns sensible lights while driving,")
print("we use CARLA's own attachment (handles turns correctly).")
env.close()