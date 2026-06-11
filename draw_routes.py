# ==========================================================
# draw_routes.py
# Visualise the three locked evaluation routes in the CARLA
# window (Town03), the same way visualize_spawn_points drew
# the spawn indices.
#
# Draws, for each route:
#   * planned path (GlobalRoutePlanner) as a coloured line
#   * SPAWN marker (green-ish) with a heading arrow
#   * DEST  marker (red-ish) label
#   * route label at the spawn
#
# Routes (locked):
#   R1 roundabout      : 123 -> 41   (colour: cyan)
#   R2 sweeping curve  : 107 -> 170  (colour: yellow)
#   R3 straight arterial: 249 -> 21  (colour: magenta)
#
# Look at the CARLA spectator window while this runs. It
# redraws for DRAW_SECONDS so you can pan around and confirm:
#   - each path stays on ONE continuous drivable lane
#   - the spawn arrow points INTO the route (forward), so an
#     open-loop agent driving forward will trace the path
#   - R1 sweeps through the roundabout, R2 bends through the
#     bottom-left corner, R3 is straight
#
# Uses the same robust planner import as the other tools
# (bypasses the local agents/ package shadowing).
#
# Usage:  (CARLA up, Town03)  python3 draw_routes.py
# CARLA 0.9.16 | Python 3.10 | diffusioncarla
# ==========================================================

import sys, os, time, math, types

# --- robust GlobalRoutePlanner import ----------------------
def _load_grp():
    roots = [
        "/home/shyam/Carla/PythonAPI/carla",
        "/home/shyam/carla/PythonAPI/carla",
        os.path.expanduser("~/Carla/PythonAPI/carla"),
        os.path.expanduser("~/carla/PythonAPI/carla"),
    ]
    nav = None
    for r in roots:
        c = os.path.join(r, "agents", "navigation")
        if os.path.isdir(c):
            nav = c; break
    if nav is None:
        raise ImportError("CARLA agents/navigation not found.")
    agents_root = os.path.dirname(nav)
    pa = types.ModuleType("agents"); pa.__path__ = [agents_root]
    sys.modules["agents"] = pa
    pn = types.ModuleType("agents.navigation"); pn.__path__ = [nav]
    sys.modules["agents.navigation"] = pn
    from agents.navigation.global_route_planner import (
        GlobalRoutePlanner as G)
    return G

import carla
GlobalRoutePlanner = _load_grp()

# ----------------------------------------------------------
# Locked routes: name -> (spawn_idx, dest_idx, color)
# ----------------------------------------------------------
ROUTES = {
    "R1_roundabout": (123,  41, carla.Color(0, 255, 255)),   # cyan
    "R2_curve":      (107, 170, carla.Color(255, 230, 0)),   # yellow
    "R3_straight":   (249,  21, carla.Color(255, 0, 255)),   # magenta
}

SAMPLING_RES = 2.0
DRAW_SECONDS = 120     # how long to keep the drawing up
LINE_THICK   = 0.30
Z_LIFT       = 0.5     # draw slightly above road


def draw_arrow(world, tf, color, length=6.0):
    """Heading arrow at a transform (shows spawn facing)."""
    loc = tf.location
    yaw = math.radians(tf.rotation.yaw)
    end = carla.Location(
        x=loc.x + length * math.cos(yaw),
        y=loc.y + length * math.sin(yaw),
        z=loc.z + Z_LIFT,
    )
    world.debug.draw_arrow(
        carla.Location(loc.x, loc.y, loc.z + Z_LIFT),
        end, thickness=0.25, arrow_size=0.6,
        color=color, life_time=DRAW_SECONDS,
    )


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap  = world.get_map()
    sps   = cmap.get_spawn_points()
    grp   = GlobalRoutePlanner(cmap, SAMPLING_RES)

    print(f"Town03 spawn points: {len(sps)}")
    print(f"Drawing 3 routes for {DRAW_SECONDS}s. "
          f"Look at the CARLA window.\n")

    for name, (a, b, color) in ROUTES.items():
        if a >= len(sps) or b >= len(sps):
            print(f"  {name}: index OOB ({a}->{b}) — skip")
            continue
        sp_tf = sps[a]
        ds_tf = sps[b]
        try:
            wps = grp.trace_route(
                sp_tf.location, ds_tf.location)
        except Exception as e:
            print(f"  {name}: planner FAIL — {e}")
            continue
        if not wps or len(wps) < 2:
            print(f"  {name}: empty route — skip")
            continue

        # length for the print summary
        L = 0.0
        for i in range(1, len(wps)):
            p = wps[i-1][0].transform.location
            q = wps[i][0].transform.location
            L += math.hypot(p.x-q.x, p.y-q.y)

        # draw the planned path as connected segments
        for i in range(1, len(wps)):
            p = wps[i-1][0].transform.location
            q = wps[i][0].transform.location
            world.debug.draw_line(
                carla.Location(p.x, p.y, p.z + Z_LIFT),
                carla.Location(q.x, q.y, q.z + Z_LIFT),
                thickness=LINE_THICK, color=color,
                life_time=DRAW_SECONDS,
            )

        # spawn marker (big point) + heading arrow + label
        world.debug.draw_point(
            carla.Location(sp_tf.location.x,
                           sp_tf.location.y,
                           sp_tf.location.z + Z_LIFT),
            size=0.25, color=carla.Color(0, 255, 0),
            life_time=DRAW_SECONDS,
        )
        draw_arrow(world, sp_tf, carla.Color(0, 255, 0))
        world.debug.draw_string(
            carla.Location(sp_tf.location.x,
                           sp_tf.location.y,
                           sp_tf.location.z + 2.0),
            f"{name} START [{a}]", draw_shadow=True,
            color=color, life_time=DRAW_SECONDS,
        )

        # dest marker + label
        world.debug.draw_point(
            carla.Location(ds_tf.location.x,
                           ds_tf.location.y,
                           ds_tf.location.z + Z_LIFT),
            size=0.25, color=carla.Color(255, 60, 60),
            life_time=DRAW_SECONDS,
        )
        world.debug.draw_string(
            carla.Location(ds_tf.location.x,
                           ds_tf.location.y,
                           ds_tf.location.z + 2.0),
            f"{name} END [{b}]", draw_shadow=True,
            color=color, life_time=DRAW_SECONDS,
        )

        print(f"  {name:16s} {a:>3} -> {b:>3} | "
              f"len {L:6.1f} m | {len(wps)} waypoints | "
              f"drawn")

    print("\nRoutes drawn. In the CARLA window, confirm each "
          "path stays on one lane and the green arrow points\n"
          "INTO the route. Press Ctrl+C here when done "
          "(drawing persists for the set time).")
    try:
        time.sleep(DRAW_SECONDS)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()