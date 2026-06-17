# ==========================================================
# draw_routes_carla.py
# Draw all three evaluation routes directly in the LIVE CARLA
# window (the glowing-line overlay look), for taking screenshots.
#
# Uses route_utils.generate_reference_route() so the drawn paths
# exactly match what the eval drives. Draws each route as a
# coloured debug line with START/END labels, then holds so you
# can move the spectator and screenshot each route.
#
# Routes (from route_utils.ROUTES):
#   R1 roundabout : 123 -> 41   (blue)
#   R2 curve      : 170 -> 45   (green)
#   R3 straight   : 249 -> 21   (red)
#
# The lines are drawn with a long life_time so they persist
# while you zoom/reposition the spectator to screenshot each.
#
# Usage:  python3 draw_routes_carla.py [--life 600] [--single R2]
#   --life N    : seconds the lines persist (default 600 = 10 min)
#   --single K  : draw only one route (route_1_roundabout /
#                 route_2_curve / route_3_straight) for clean
#                 individual screenshots
#
# Tip for clean individual shots: run with --single route_2_curve
# (etc.) so only one coloured route is on screen at a time.
# ==========================================================

import argparse
import carla
import route_utils

ap = argparse.ArgumentParser()
ap.add_argument("--life", type=float, default=600.0,
                help="line persistence (seconds)")
ap.add_argument("--single", default=None,
                help="draw only one route key")
ap.add_argument("--res", type=float, default=0.5,
                help="reference sampling resolution (m)")
args = ap.parse_args()

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
cmap = world.get_map()
dbg = world.debug

# CARLA debug colours (R,G,B)
STYLE = {
    "route_1_roundabout": (carla.Color(30, 120, 255),  "R1 ROUNDABOUT"),
    "route_2_curve":      (carla.Color(40, 200, 70),   "R2 CURVE"),
    "route_3_straight":   (carla.Color(220, 50, 50),   "R3 STRAIGHT"),
}

routes = route_utils.ROUTES
if args.single:
    if args.single not in routes:
        raise SystemExit(f"--single must be one of {list(routes)}")
    routes = {args.single: routes[args.single]}

for name, meta in routes.items():
    color, label = STYLE.get(name, (carla.Color(255, 255, 255), name))
    ref_xy, start_tf = route_utils.generate_reference_route(
        world, cmap, meta["spawn"], meta["dest"], args.res
    )
    # draw the route as connected line segments. We query the
    # actual road surface height at each point and lift the line
    # a fixed amount ABOVE it, so the line stays visible on
    # banked / elevated curve sections (fixes R2 not showing).
    LIFT = 1.2  # metres above the road surface
    prev = None
    for (x, y) in ref_xy:
        # snap to the road to get true z, then lift
        wp = cmap.get_waypoint(
            carla.Location(x=float(x), y=float(y), z=0.0),
            project_to_road=True)
        z = (wp.transform.location.z if wp else 0.0) + LIFT
        loc = carla.Location(x=float(x), y=float(y), z=z)
        if prev is not None:
            dbg.draw_line(prev, loc, thickness=0.50,
                          color=color, life_time=args.life)
        # also drop a small point at each node for continuity
        dbg.draw_point(loc, size=0.08, color=color,
                       life_time=args.life)
        prev = loc

    # start (big point + label) and end (point + label)
    sx, sy = ref_xy[0]
    ex, ey = ref_xy[-1]
    s_loc = carla.Location(x=float(sx), y=float(sy), z=1.0)
    e_loc = carla.Location(x=float(ex), y=float(ey), z=1.0)
    dbg.draw_point(s_loc, size=0.25, color=color, life_time=args.life)
    dbg.draw_point(e_loc, size=0.25, color=color, life_time=args.life)
    dbg.draw_string(s_loc, f"{label} START", draw_shadow=True,
                    color=color, life_time=args.life)
    dbg.draw_string(e_loc, f"{label} END", draw_shadow=True,
                    color=color, life_time=args.life)
    print(f"drew {name}: {len(ref_xy)} pts  "
          f"spawn={meta['spawn']} dest={meta['dest']}")

print(f"\nRoutes drawn (persist {args.life:.0f}s). "
      f"Move the CARLA spectator and screenshot each route.")
print("Re-run with --single route_2_curve (etc.) for clean "
      "single-route shots.")