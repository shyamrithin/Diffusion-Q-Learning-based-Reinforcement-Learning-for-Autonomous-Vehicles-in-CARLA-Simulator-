# ==========================================================
# find_traffic_lights.py
# Locate all traffic lights in the current CARLA map and the
# nearest spawn point to each, so the demo can be spawned at a
# SIGNALISED intersection (where red/yellow/green behaviour is
# actually visible).
#
# Prints, for each traffic light:
#   light id, location, current state, nearest spawn-point index,
#   and distance from that spawn to the light.
#
# Use the printed spawn index as the demo spawn so the ego
# approaches a real traffic light.
#
# Usage:  python3 find_traffic_lights.py   (CARLA server up)
# ==========================================================

import math
import carla

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
cmap = world.get_map()

lights = world.get_actors().filter("traffic.traffic_light*")
spawns = cmap.get_spawn_points()
print(f"Map has {len(lights)} traffic lights, "
      f"{len(spawns)} spawn points\n")

if len(lights) == 0:
    print("No traffic lights in this map — signal demo not "
          "possible here.")
    raise SystemExit

rows = []
for tl in lights:
    tloc = tl.get_transform().location
    # nearest spawn point to this light
    best_i, best_d = -1, 1e9
    for i, sp in enumerate(spawns):
        d = math.sqrt((sp.location.x - tloc.x) ** 2 +
                      (sp.location.y - tloc.y) ** 2)
        if d < best_d:
            best_d, best_i = d, i
    rows.append((tl.id, tloc, str(tl.get_state()),
                 best_i, best_d))

# sort by nearest-spawn distance (closest approachable first)
rows.sort(key=lambda r: r[4])

print(f"{'light_id':>9} | {'location (x,y)':>20} | {'state':>7} "
      f"| {'near_spawn':>10} | {'dist(m)':>7}")
print("-" * 70)
for tid, loc, state, si, d in rows:
    print(f"{tid:>9} | ({loc.x:7.1f},{loc.y:7.1f})      | "
          f"{state:>7} | {si:>10} | {d:7.1f}")

print("\nPick a spawn index with a SMALL dist to a light, and an "
      "approach that drives TOWARD it. Spawn the demo there with "
      "--spawn <index>. A good candidate is one where the nearest "
      "spawn is ~15-40 m from the light (room to approach + stop).")

# suggest a few good demo spawns: 15-45m from a light
print("\nSuggested demo spawns (15-45 m from a light):")
for tid, loc, state, si, d in rows:
    if 15.0 <= d <= 45.0:
        print(f"  --spawn {si}   (light {tid}, {d:.0f} m away)")