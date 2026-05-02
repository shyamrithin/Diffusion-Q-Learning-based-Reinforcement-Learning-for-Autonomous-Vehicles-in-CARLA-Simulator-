# ==========================================================
# rlcarla/utils/traffic_manager.py
# NPC Traffic Manager for RLCarla
# Customisable density, behaviour, cleanup
# CARLA 0.9.16 compatible
# ==========================================================

import carla
import random
import logging

logger = logging.getLogger(__name__)


# ==========================================================
# TRAFFIC PRESETS
# ==========================================================
TRAFFIC_PRESETS = {
    "empty"    : {"n_vehicles": 0,  "n_walkers": 0 },
    "light"    : {"n_vehicles": 10, "n_walkers": 5 },
    "medium"   : {"n_vehicles": 30, "n_walkers": 15},
    "heavy"    : {"n_vehicles": 60, "n_walkers": 30},
    "chaos"    : {"n_vehicles": 100,"n_walkers": 50},
}

# Vehicle filter — exclude bikes/motorcycles for cleaner training
VEHICLE_FILTERS = [
    "vehicle.audi.*",
    "vehicle.bmw.*",
    "vehicle.chevrolet.*",
    "vehicle.citroen.*",
    "vehicle.dodge.*",
    "vehicle.ford.*",
    "vehicle.jeep.*",
    "vehicle.lincoln.*",
    "vehicle.mercedes.*",
    "vehicle.mini.*",
    "vehicle.mustang.*",
    "vehicle.nissan.*",
    "vehicle.seat.*",
    "vehicle.tesla.*",
    "vehicle.toyota.*",
    "vehicle.volkswagen.*",
]


# ==========================================================
class TrafficManager:
    """
    Spawns and manages NPC vehicles and walkers.

    Usage:
        tm = TrafficManager(client, world)
        tm.spawn(preset="medium")       # or custom n_vehicles/n_walkers
        ...
        tm.destroy()                    # clean up before reset
    """

    def __init__(self, client, world, tm_port=8000):
        self.client   = client
        self.world    = world
        self.tm_port  = tm_port

        self._vehicles = []   # list of carla.Actor
        self._walkers  = []   # list of (walker_actor, walker_controller)
        self._tm       = None

        self._init_traffic_manager()

    # ----------------------------------------------------------
    def _init_traffic_manager(self):
        self._tm = self.client.get_trafficmanager(self.tm_port)
        self._tm.set_synchronous_mode(True)

        # Global NPC behaviour defaults
        self._tm.set_global_distance_to_leading_vehicle(2.5)
        self._tm.global_percentage_speed_difference(10.0)  # 10% slower than limit

    # ----------------------------------------------------------
    def spawn(
        self,
        preset       = None,
        n_vehicles   = 20,
        n_walkers    = 0,
        ego_vehicle  = None,      # keep NPCs away from ego spawn
        safe_radius  = 20.0,      # metres — no NPC within this of ego
        random_seed  = None,
    ):
        """
        Spawn NPC traffic.

        Args:
            preset      : one of TRAFFIC_PRESETS keys, overrides n_vehicles/n_walkers
            n_vehicles  : number of NPC vehicles
            n_walkers   : number of pedestrians
            ego_vehicle : ego carla.Vehicle — used for safe radius check
            safe_radius : NPCs won't spawn within this distance of ego
            random_seed : for reproducibility
        """

        if random_seed is not None:
            random.seed(random_seed)

        if preset is not None:
            if preset not in TRAFFIC_PRESETS:
                logger.warning(f"Unknown preset '{preset}'. Using 'medium'.")
                preset = "medium"
            cfg        = TRAFFIC_PRESETS[preset]
            n_vehicles = cfg["n_vehicles"]
            n_walkers  = cfg["n_walkers"]

        logger.info(f"[TrafficManager] Spawning {n_vehicles} vehicles, {n_walkers} walkers")

        self._spawn_vehicles(n_vehicles, ego_vehicle, safe_radius)
        if n_walkers > 0:
            self._spawn_walkers(n_walkers)

    # ----------------------------------------------------------
    def _spawn_vehicles(self, n, ego_vehicle, safe_radius):

        bp_lib       = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # Collect ego location for distance check
        ego_loc = None
        if ego_vehicle is not None:
            ego_loc = ego_vehicle.get_location()

        # Build blueprint pool
        blueprints = []
        for f in VEHICLE_FILTERS:
            blueprints.extend(bp_lib.filter(f))

        if not blueprints:
            logger.warning("[TrafficManager] No vehicle blueprints found.")
            return

        spawned = 0
        for sp in spawn_points:
            if spawned >= n:
                break

            # Skip spawn points too close to ego
            if ego_loc is not None:
                dist = sp.location.distance(ego_loc)
                if dist < safe_radius:
                    continue

            bp = random.choice(blueprints)

            # Randomise vehicle colour
            if bp.has_attribute("color"):
                color = random.choice(
                    bp.get_attribute("color").recommended_values
                )
                bp.set_attribute("color", color)

            # Disable autopilot initially, set after spawn
            bp.set_attribute("role_name", "autopilot")

            actor = self.world.try_spawn_actor(bp, sp)
            if actor is not None:
                actor.set_autopilot(True, self.tm_port)

                # Per-vehicle behaviour randomisation
                self._tm.vehicle_percentage_speed_difference(
                    actor, random.uniform(-10, 20)
                )
                self._tm.distance_to_leading_vehicle(
                    actor, random.uniform(1.5, 4.0)
                )
                self._tm.auto_lane_change(actor, True)

                # Small chance of ignoring lights (chaos factor)
                if random.random() < 0.05:
                    self._tm.ignore_lights_percentage(actor, 100)

                self._vehicles.append(actor)
                spawned += 1

        logger.info(f"[TrafficManager] Spawned {spawned}/{n} vehicles")

    # ----------------------------------------------------------
    def _spawn_walkers(self, n):
        """Spawn pedestrians with walker AI controllers."""

        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter("walker.pedestrian.*")

        if not walker_bps:
            logger.warning("[TrafficManager] No walker blueprints found.")
            return

        controller_bp = bp_lib.find("controller.ai.walker")

        spawned = 0
        for _ in range(n * 2):   # try 2x to hit target count
            if spawned >= n:
                break

            bp  = random.choice(walker_bps)
            loc = self.world.get_random_location_from_navigation()

            if loc is None:
                continue

            tf    = carla.Transform(loc)
            actor = self.world.try_spawn_actor(bp, tf)

            if actor is None:
                continue

            ctrl = self.world.try_spawn_actor(
                controller_bp,
                carla.Transform(),
                attach_to=actor
            )

            if ctrl is None:
                actor.destroy()
                continue

            self._walkers.append((actor, ctrl))
            spawned += 1

        # Tick once so controllers initialise
        self.world.tick()

        # Start walker controllers
        for actor, ctrl in self._walkers:
            ctrl.start()
            ctrl.go_to_location(
                self.world.get_random_location_from_navigation()
            )
            ctrl.set_max_speed(random.uniform(0.8, 2.0))

        logger.info(f"[TrafficManager] Spawned {spawned}/{n} walkers")

    # ----------------------------------------------------------
    def set_global_speed(self, percentage_difference):
        """
        Adjust all NPC speeds relative to road speed limit.
        Negative = faster, Positive = slower.
        e.g. -20 means NPCs drive 20% faster than limit.
        """
        self._tm.global_percentage_speed_difference(percentage_difference)

    # ----------------------------------------------------------
    def set_ignore_lights(self, percentage=100):
        """Make all NPCs ignore traffic lights (chaos mode)."""
        for v in self._vehicles:
            try:
                self._tm.ignore_lights_percentage(v, percentage)
            except Exception:
                pass

    # ----------------------------------------------------------
    def get_nearby_vehicles(self, ego_vehicle, n=4, max_range=50.0):
        """
        Returns info on the n closest NPC vehicles to the ego.
        Used by obs_builder to fill the nearby_vehicles obs component.

        Returns list of dicts:
            rel_x, rel_y     : position relative to ego (ego frame)
            rel_vx, rel_vy   : velocity relative to ego
            dist             : distance in metres
        """

        ego_tf  = ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = ego_tf.rotation.yaw
        ego_vel = ego_vehicle.get_velocity()

        cos_y = __import__("math").cos(__import__("math").radians(ego_yaw))
        sin_y = __import__("math").sin(__import__("math").radians(ego_yaw))

        nearby = []
        for v in self._vehicles:
            try:
                loc  = v.get_location()
                dist = ego_loc.distance(loc)
                if dist > max_range:
                    continue

                vel  = v.get_velocity()

                # World-frame offsets
                dx = loc.x - ego_loc.x
                dy = loc.y - ego_loc.y

                # Rotate into ego vehicle frame
                rel_x  =  cos_y * dx + sin_y * dy
                rel_y  = -sin_y * dx + cos_y * dy

                # Relative velocity in world frame
                dvx = vel.x - ego_vel.x
                dvy = vel.y - ego_vel.y

                rel_vx =  cos_y * dvx + sin_y * dvy
                rel_vy = -sin_y * dvx + cos_y * dvy

                nearby.append({
                    "rel_x" : rel_x,
                    "rel_y" : rel_y,
                    "rel_vx": rel_vx,
                    "rel_vy": rel_vy,
                    "dist"  : dist,
                })
            except Exception:
                continue

        # Sort by distance, return n closest
        nearby.sort(key=lambda d: d["dist"])
        return nearby[:n]

    # ----------------------------------------------------------
    def vehicle_count(self):
        return len(self._vehicles)

    def walker_count(self):
        return len(self._walkers)

    # ----------------------------------------------------------
    def destroy(self):
        """Clean up all NPCs. Call before env.reset()."""

        # Stop walker controllers first
        for actor, ctrl in self._walkers:
            try:
                ctrl.stop()
            except Exception:
                pass

        actors_to_destroy = []

        for v in self._vehicles:
            actors_to_destroy.append(v)

        for actor, ctrl in self._walkers:
            actors_to_destroy.append(ctrl)
            actors_to_destroy.append(actor)

        # Batch destroy is faster than one-by-one
        if actors_to_destroy:
            self.client.apply_batch([
                carla.command.DestroyActor(a) for a in actors_to_destroy
            ])

        self._vehicles = []
        self._walkers  = []

        logger.info("[TrafficManager] All NPCs destroyed")
