# ==========================================================
# rlcarla/envs/carla_env.py
# RLCarla Main Environment — Full Restructure v3
#
# Changes in this version:
#   - Intersection false-positive offroad fix
#     (CARLA returns None at junctions even on valid road)
#   - Curve-biased spawning (60% near curves)
#   - Same-lane waypoint following
#   - Wrong-lane strict enforcement (3 step limit)
#   - Town03 single map for curve training
#   - Trajectory overlay (white centerline + green kinematic)
#   - LiDAR queue flush on reset (no FPS dip)
#   - Kickstart on reset (vehicle moving from step 1)
#
# CARLA 0.9.16 | Gymnasium 1.3 | Python 3.10
# ==========================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import random
import math
import logging
import time

from rlcarla.sensors.lidar         import LidarSensor
from rlcarla.sensors.camera        import CameraManager
from rlcarla.core.obs_builder      import (
    ObservationBuilder, FrameStack,
    SINGLE_OBS_DIM, OBS_DIM
)
from rlcarla.utils.reward          import RewardCalculator
from rlcarla.utils.traffic_manager import TrafficManager

logger = logging.getLogger(__name__)


def _wrap_angle(angle):
    """Wrap angle to [-180, 180] degrees."""
    while angle >  180: angle -= 360
    while angle < -180: angle += 360
    return angle


# ==========================================================
# ENVIRONMENT CONFIG
# ==========================================================
class EnvConfig:
    """
    Centralised configuration for CarlaEnv.
    All tunable parameters in one place.
    Use from_dict() for EasyCarla-style params dict.
    """

    HOST             = "localhost"
    PORT             = 2000
    TIMEOUT          = 30.0
    FIXED_DT         = 0.05
    MAX_STEPS        = 1000
    CAM_WIDTH        = 800
    CAM_HEIGHT       = 450
    CAM_FOV          = 100
    SPAWN_RETRIES    = 30
    SAFE_SPAWN_DIST  = 20.0
    STUCK_LIMIT      = 80
    STUCK_SPEED      = 0.3
    COLLISION_DONE   = True
    OFFROAD_DONE     = True
    WRONG_LANE_DONE  = True
    WRONG_LANE_LIMIT = 3       # strict — 3 steps = ~0.15s
    WRONG_WAY_DONE   = True
    TRAFFIC_PRESET   = "empty"
    TRAFFIC_LIGHTS   = "off"
    SPECTATOR_MODE   = "follow"
    MAPS             = ["Town03"]   # best curves + roundabouts
    CURVE_SPAWN_BIAS = 0.6          # 60% spawns near curves
    CURVE_MIN_ANGLE  = 5.0          # degrees to qualify as curve

    @classmethod
    def from_dict(cls, d):
        """EasyCarla-style params dict → EnvConfig."""
        cfg     = cls()
        mapping = {
            'town'             : ('MAPS',           lambda v: [v]),
            'port'             : ('PORT',           lambda v: v),
            'dt'               : ('FIXED_DT',       lambda v: v),
            'max_time_episode' : ('MAX_STEPS',      lambda v: v),
            'traffic'          : ('TRAFFIC_LIGHTS', lambda v: v),
        }
        for key, (attr, transform) in mapping.items():
            if key in d:
                setattr(cfg, attr, transform(d[key]))
        return cfg


# ==========================================================
# MAIN ENVIRONMENT
# ==========================================================
class CarlaEnv(gym.Env):
    """
    RLCarla — Full Autonomous Driving Environment.

    Observation : 564D stacked vector (4 × 141D)
    Action      : [throttle ∈ [0,1], steer ∈ [-1,1],
                   brake ∈ [0,1]]

    Key design decisions:
      - Waypoints follow same lane_id (no lane changes)
      - Intersection offroad is correctly handled
      - 60% of spawns are near curves (curve training bias)
      - LiDAR queue flushed on reset (no FPS dip)
      - Vehicle kickstarted on reset (always moving)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None, traffic_preset=None,
                 map_name=None):
        super().__init__()

        self.cfg             = config or EnvConfig()
        self._map_name       = map_name
        self._traffic_preset = (
            traffic_preset or self.cfg.TRAFFIC_PRESET
        )

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array( [0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.vehicle           = None
        self.actor_list        = []
        self._lidar            = None
        self._camera           = None
        self._traffic          = None
        self._obs_builder      = ObservationBuilder()
        self._frame_stack      = FrameStack()
        self._reward_calc      = RewardCalculator()

        self._collision_flag   = False
        self._offroad_flag     = False
        self._wrong_way_flag   = False
        self._stuck_count      = 0
        self._wrong_lane_count = 0
        self._step_count       = 0
        self._episode_reward   = 0.0
        self._episode          = 0
        self._prev_action      = np.zeros(3, dtype=np.float32)
        self._collision_sensor = None

        self._connect()

    # ==========================================================
    # CONNECTION
    # ==========================================================
    def _connect(self):
        """Connect to CARLA server and apply sync settings."""
        logger.info("[Env] Connecting to CARLA...")
        self.client    = carla.Client(self.cfg.HOST, self.cfg.PORT)
        self.client.set_timeout(self.cfg.TIMEOUT)
        self.world     = self.client.get_world()
        self.carla_map = self.world.get_map()
        self._apply_sync_settings()
        logger.info(f"[Env] Connected. Map: {self.carla_map.name}")

    def _apply_sync_settings(self):
        """Enable synchronous mode with fixed timestep."""
        settings                     = self.world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = self.cfg.FIXED_DT
        self.world.apply_settings(settings)
        self._obs_builder._dt        = self.cfg.FIXED_DT

    # ==========================================================
    # MAP LOADING
    # ==========================================================
    def _load_map(self, map_name=None):
        """
        Load map only if different from current.
        Avoids expensive load_world() on every reset.
        """
        current = self.world.get_map().name.split("/")[-1]
        target  = map_name or random.choice(self.cfg.MAPS)

        if target != current:
            logger.info(f"[Env] Loading map: {target}")
            self.world     = self.client.load_world(target)
            self.carla_map = self.world.get_map()
            self._apply_sync_settings()
            time.sleep(3.0)
        else:
            self.carla_map = self.world.get_map()

    # ==========================================================
    # TRAFFIC LIGHTS
    # ==========================================================
    def _configure_traffic_lights(self):
        """
        'off' → freeze all lights green (phase 1 training)
        'on'  → normal traffic light behaviour (phase 2+)
        """
        actors = self.world.get_actors().filter(
            "traffic.traffic_light*"
        )
        if self.cfg.TRAFFIC_LIGHTS == "off":
            for actor in actors:
                actor.set_state(carla.TrafficLightState.Green)
                actor.freeze(True)
            logger.info("[Env] Traffic lights frozen GREEN")
        else:
            for actor in actors:
                actor.freeze(False)
            logger.info("[Env] Traffic lights NORMAL")

    # ==========================================================
    # SPECTATOR CAMERA
    # ==========================================================
    def _update_spectator(self):
        """
        Sync CARLA world camera to ego vehicle.
        Modes: 'follow' (3rd person), 'top' (bird-eye), 'none'
        """
        if self.cfg.SPECTATOR_MODE == "none" or \
                self.vehicle is None:
            return

        spectator = self.world.get_spectator()
        tf        = self.vehicle.get_transform()

        if self.cfg.SPECTATOR_MODE == "follow":
            cam_loc = tf.transform(carla.Location(x=-6.0, z=3.0))
            cam_rot = carla.Rotation(
                pitch=-10, yaw=tf.rotation.yaw, roll=0
            )
            spectator.set_transform(
                carla.Transform(cam_loc, cam_rot)
            )
        elif self.cfg.SPECTATOR_MODE == "top":
            spectator.set_transform(carla.Transform(
                tf.location + carla.Location(z=40),
                carla.Rotation(pitch=-90)
            ))

    # ==========================================================
    # TRAJECTORY OVERLAY (CARLA world debug draw)
    # ==========================================================
    def _draw_trajectory_overlay(self):
        """
        Draws two overlays in CARLA world space:
          White dots — road centerline (ideal path)
          Green line — kinematic prediction (agent intention)
            uses bicycle model: wheelbase + current steer
        Visible in CARLA spectator window.
        """
        if self.vehicle is None:
            return

        tf  = self.vehicle.get_transform()
        loc = tf.location

        # White dots — road centerline
        wp      = self.carla_map.get_waypoint(
            loc, project_to_road=True
        )
        current = wp
        for i in range(20):
            nexts = current.next(2.0)
            if not nexts:
                break
            current   = nexts[0]
            wloc      = current.transform.location
            intensity = max(120, 255 - i * 7)
            self.world.debug.draw_point(
                carla.Location(
                    x=wloc.x, y=wloc.y, z=wloc.z + 0.3
                ),
                size      = 0.10,
                color     = carla.Color(
                    intensity, intensity, intensity
                ),
                life_time = 0.06,
            )

        # Green line — kinematic bicycle model prediction
        vel       = self.vehicle.get_velocity()
        control   = self.vehicle.get_control()
        speed     = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        steer     = control.steer
        x         = loc.x
        y         = loc.y
        z         = loc.z + 0.3
        yaw       = math.radians(tf.rotation.yaw)
        wheelbase = 2.87   # Tesla Model 3
        prev_loc  = carla.Location(x=x, y=y, z=z)

        for i in range(25):
            if speed < 0.1:
                break
            if abs(steer) > 0.01:
                turn_radius = wheelbase / math.tan(
                    abs(steer) * 0.5 + 1e-6
                )
                d_yaw = (speed * 0.1) / turn_radius
                d_yaw = d_yaw if steer > 0 else -d_yaw
            else:
                d_yaw = 0.0

            yaw     += d_yaw
            x       += speed * math.cos(yaw) * 0.1
            y       += speed * math.sin(yaw) * 0.1

            intensity = max(80, 255 - i * 7)
            new_loc   = carla.Location(x=x, y=y, z=z)

            self.world.debug.draw_line(
                prev_loc, new_loc,
                thickness = 0.10,
                color     = carla.Color(0, intensity, 0),
                life_time = 0.06,
            )
            prev_loc = new_loc

    # ==========================================================
    # CURVE-BIASED SPAWNING
    # ==========================================================
    def _get_curve_spawn_points(self):
        """
        Filter spawn points to find ones with curves ahead.
        Curve defined as yaw_diff > CURVE_MIN_ANGLE within 10m.
        Returns (curved_points, all_points).
        """
        spawn_points = self.carla_map.get_spawn_points()
        curved = []

        for sp in spawn_points:
            wp = self.carla_map.get_waypoint(sp.location)
            if wp is None:
                continue
            nexts = wp.next(10.0)
            if not nexts:
                continue
            yaw_diff = abs(_wrap_angle(
                nexts[0].transform.rotation.yaw -
                wp.transform.rotation.yaw
            ))
            if yaw_diff > self.cfg.CURVE_MIN_ANGLE:
                curved.append(sp)

        return curved, spawn_points

    def _spawn_vehicle(self):
        """
        Spawn ego vehicle (blue Tesla Model 3).
        60% chance to spawn near a curve (CURVE_SPAWN_BIAS).
        """
        bp_lib = self.world.get_blueprint_library()
        bp     = random.choice(
            bp_lib.filter("vehicle.tesla.model3")
        )
        if bp.has_attribute("color"):
            bp.set_attribute("color", "0,120,255")

        curved_points, all_points = self._get_curve_spawn_points()

        if (curved_points and
                np.random.rand() < self.cfg.CURVE_SPAWN_BIAS):
            spawn_pool = curved_points
        else:
            spawn_pool = all_points

        random.shuffle(spawn_pool)

        for sp in spawn_pool[:self.cfg.SPAWN_RETRIES]:
            v = self.world.try_spawn_actor(bp, sp)
            if v is not None:
                self.vehicle = v
                self.actor_list.append(v)
                return True

        logger.error("[Env] Failed to spawn ego vehicle!")
        return False

    # ==========================================================
    # SENSORS
    # ==========================================================
    def _attach_sensors(self):
        """Attach LiDAR, camera and collision sensor to ego."""
        self._lidar = LidarSensor(self.vehicle, self.world)
        self.actor_list.append(self._lidar._sensor)

        self._camera = CameraManager(
            self.vehicle, self.world,
            width  = self.cfg.CAM_WIDTH,
            height = self.cfg.CAM_HEIGHT,
            fov    = self.cfg.CAM_FOV,
        )
        self.actor_list.append(self._camera._sensor)

        col_bp = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        self._collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle
        )
        self._collision_sensor.listen(self._on_collision)
        self.actor_list.append(self._collision_sensor)

    def _on_collision(self, event):
        """Collision callback — sets flag for this step."""
        self._collision_flag = True
        logger.debug(
            f"[Env] Collision: {event.other_actor.type_id}"
        )

    # ==========================================================
    # TRAFFIC
    # ==========================================================
    def _spawn_traffic(self):
        """Spawn NPC traffic according to current preset."""
        if self._traffic is not None:
            self._traffic.destroy()
        self._traffic = TrafficManager(self.client, self.world)
        self._traffic.spawn(
            preset      = self._traffic_preset,
            ego_vehicle = self.vehicle,
            safe_radius = self.cfg.SAFE_SPAWN_DIST,
        )

    # ==========================================================
    # RESET
    # ==========================================================
    def reset(self, seed=None, options=None, map_name=None):
        """
        Reset environment for new episode.
        Order: cleanup → map → spawn → sensors →
               traffic → kickstart → flush → obs
        """
        super().reset(seed=seed)
        map_name = (options or {}).get("map_name", map_name)

        self._episode += 1
        logger.info(
            f"[Env] Reset — episode {self._episode} | "
            f"prev reward {self._episode_reward:.1f}"
        )

        self._cleanup()
        self._load_map(map_name or self._map_name)

        if not self._spawn_vehicle():
            raise RuntimeError("Could not spawn ego vehicle.")

        # Warm up physics before sensors
        for _ in range(10):
            self.world.tick()

        self._attach_sensors()
        self._configure_traffic_lights()
        self._spawn_traffic()

        # Kickstart — ensures vehicle is moving at step 0
        # Prevents stuck-at-spawn during random exploration
        for _ in range(15):
            self.vehicle.apply_control(carla.VehicleControl(
                throttle = 0.5,
                steer    = 0.0,
                brake    = 0.0,
            ))
            self.world.tick()

        # Flush LiDAR queue — prevents FPS dip from backlog
        if self._lidar is not None:
            self._lidar.flush()

        # Reset all episode state
        self._collision_flag   = False
        self._offroad_flag     = False
        self._wrong_way_flag   = False
        self._stuck_count      = 0
        self._wrong_lane_count = 0
        self._step_count       = 0
        self._episode_reward   = 0.0
        self._prev_action      = np.zeros(3, dtype=np.float32)

        self._obs_builder.reset(self.vehicle)
        self._reward_calc.reset()

        # Warm up sensors
        for _ in range(5):
            self.world.tick()

        raw_obs = self._obs_builder.build(
            self.vehicle, self.carla_map,
            self._lidar, self._traffic,
        )
        obs = self._frame_stack.reset(raw_obs)

        self._update_spectator()
        self._draw_trajectory_overlay()

        return obs, {}

    # ==========================================================
    # STEP
    # ==========================================================
    def step(self, action):
        """
        Apply action, tick world, build obs, compute reward.
        Returns: (obs, reward, done, truncated, info)
        """
        self._step_count += 1

        throttle = float(np.clip(action[0],  0.0, 1.0))
        steer    = float(np.clip(action[1], -1.0, 1.0))
        brake    = float(np.clip(action[2],  0.0, 1.0))

        self.vehicle.apply_control(carla.VehicleControl(
            throttle = throttle,
            steer    = steer,
            brake    = brake,
        ))

        self.world.tick()
        self._update_spectator()
        self._draw_trajectory_overlay()

        raw_obs = self._obs_builder.build(
            self.vehicle, self.carla_map,
            self._lidar, self._traffic,
        )
        obs = self._frame_stack.step(raw_obs)

        self._update_flags(action)

        reward, reward_info = self._get_reward(obs, action)
        self._episode_reward += reward

        done, term_reason = self._terminal()

        # Reset collision flag after reading (per-step flag)
        self._collision_flag = False
        self._prev_action    = np.array(action, dtype=np.float32)

        if done:
            logger.info(
                f"[Env] Done — {term_reason} | "
                f"steps: {self._step_count} | "
                f"reward: {self._episode_reward:.1f}"
            )

        info = {
            "step"             : self._step_count,
            "speed"            : self._get_speed(),
            "episode_reward"   : self._episode_reward,
            "term_reason"      : term_reason,
            "wrong_lane_count" : self._wrong_lane_count,
            "map"              : self.carla_map.name,
            **reward_info,
        }

        return obs, reward, done, False, info

    # ==========================================================
    # FLAG UPDATES
    # ==========================================================
    def _update_flags(self, action):
        """
        Update all episode state flags each step:
          - stuck_count
          - offroad_flag (with junction false-positive fix)
          - wrong_lane_count
          - wrong_way_flag
        """
        speed = self._get_speed()

        # Stuck counter
        if speed < self.cfg.STUCK_SPEED:
            self._stuck_count += 1
        else:
            self._stuck_count = 0

        loc    = self.vehicle.get_location()
        wp_raw = self.carla_map.get_waypoint(
            loc, project_to_road=False
        )

        # 🔥 Intersection false-positive fix
        # CARLA returns None at junctions even on valid road
        # Check if we're at a junction before flagging offroad
        wp_road = self.carla_map.get_waypoint(
            loc,
            project_to_road = True,
            lane_type       = carla.LaneType.Driving
        )

        if wp_raw is None and wp_road is not None:
            if wp_road.is_junction:
                # Inside junction — not actually offroad
                self._offroad_flag = False
            else:
                # Genuinely offroad
                self._offroad_flag = True
        else:
            self._offroad_flag = (wp_raw is None)

        # Wrong lane detection (lane_id sign check)
        if wp_raw is not None and wp_road is not None:
            in_wrong_lane = (
                wp_raw.lane_id * wp_road.lane_id < 0
            )
        else:
            in_wrong_lane = False

        if in_wrong_lane:
            self._wrong_lane_count += 1
        else:
            self._wrong_lane_count  = 0

        # Wrong way detection (yaw diff > 90°)
        # Skip at junctions — turns are valid there
        self._wrong_way_flag = False
        if wp_road is not None and not wp_road.is_junction:
            ego_yaw  = self.vehicle.get_transform().rotation.yaw
            lane_yaw = wp_road.transform.rotation.yaw
            yaw_diff = math.radians(ego_yaw - lane_yaw)
            yaw_diff = math.atan2(
                math.sin(yaw_diff), math.cos(yaw_diff)
            )
            if abs(yaw_diff) > math.pi / 2:
                self._wrong_way_flag = True

    # ==========================================================
    # TERMINAL CONDITIONS
    # ==========================================================
    def _terminal(self):
        """
        Check all termination conditions.
        Returns (done: bool, reason: str or None).
        """
        if self._stuck_count > self.cfg.STUCK_LIMIT:
            return True, "stuck"
        if self.cfg.COLLISION_DONE and self._collision_flag:
            return True, "collision"
        if self.cfg.OFFROAD_DONE and self._offroad_flag:
            return True, "offroad"
        if (self.cfg.WRONG_LANE_DONE and
                self._wrong_lane_count > self.cfg.WRONG_LANE_LIMIT):
            return True, "wrong_lane"
        if self.cfg.WRONG_WAY_DONE and self._wrong_way_flag:
            return True, "wrong_way"
        if self._step_count >= self.cfg.MAX_STEPS:
            return True, "max_steps"
        return False, None

    # ==========================================================
    # REWARD
    # ==========================================================
    def _get_reward(self, obs, action):
        """Delegate reward computation to RewardCalculator."""
        return self._reward_calc.compute(
            vehicle    = self.vehicle,
            carla_map  = self.carla_map,
            obs        = obs,
            action     = action,
            collision  = self._collision_flag,
            wrong_way  = self._wrong_way_flag,
            wrong_lane = self._wrong_lane_count > 0,
        )

    # ==========================================================
    # HELPERS
    # ==========================================================
    def _get_speed(self):
        """Return ego vehicle speed in m/s."""
        if self.vehicle is None:
            return 0.0
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    # ==========================================================
    # CAMERA CONTROL (runtime switching)
    # ==========================================================
    def set_camera_view(self, view_name):
        if self._camera is not None:
            self._camera.set_view(view_name)
            if self._camera._sensor not in self.actor_list:
                self.actor_list.append(self._camera._sensor)

    def get_camera_frame(self):
        if self._camera is not None:
            return self._camera.get_frame()
        return None

    def get_camera_intrinsic(self):
        if self._camera is not None:
            return self._camera.intrinsic
        return None

    def get_camera_transform(self):
        if self._camera is not None:
            return self._camera.get_transform()
        return None

    def available_views(self):
        if self._camera is not None:
            return self._camera.available_views
        return []

    def set_spectator_mode(self, mode):
        """Switch spectator: 'follow', 'top', 'none'"""
        self.cfg.SPECTATOR_MODE = mode

    # ==========================================================
    # TRAFFIC CONTROL (runtime)
    # ==========================================================
    def set_traffic_preset(self, preset):
        """Hot-swap traffic density without full reset."""
        self._traffic_preset = preset
        if self._traffic is not None:
            self._traffic.destroy()
        self._spawn_traffic()

    def set_traffic_count(self, n_vehicles, n_walkers=0):
        """Set exact NPC counts."""
        self._traffic_preset = None
        if self._traffic is not None:
            self._traffic.destroy()
        self._traffic = TrafficManager(self.client, self.world)
        self._traffic.spawn(
            n_vehicles  = n_vehicles,
            n_walkers   = n_walkers,
            ego_vehicle = self.vehicle,
        )

    def set_traffic_lights(self, mode):
        """Switch traffic lights on/off at runtime."""
        self.cfg.TRAFFIC_LIGHTS = mode
        self._configure_traffic_lights()

    # ==========================================================
    # CLEANUP
    # ==========================================================
    def _cleanup(self):
        """
        Destroy all actors from previous episode.
        Uses apply_batch for fast batch destruction.
        """
        if self._lidar is not None:
            self._lidar.destroy()
            self._lidar = None

        if self._camera is not None:
            self._camera.destroy()
            self._camera = None

        if self._collision_sensor is not None:
            try:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
            except Exception:
                pass
            self._collision_sensor = None

        if self._traffic is not None:
            self._traffic.destroy()
            self._traffic = None

        if self.actor_list:
            self.client.apply_batch([
                carla.command.DestroyActor(a)
                for a in self.actor_list
            ])
            self.actor_list = []

        self.vehicle = None

    def close(self):
        """Clean shutdown — restore async mode."""
        self._cleanup()
        try:
            settings                     = self.world.get_settings()
            settings.synchronous_mode    = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except Exception:
            pass
        logger.info("[Env] Closed.")

    # ==========================================================
    # UTILITIES
    # ==========================================================
    def get_episode_info(self):
        """Return dict of current episode statistics."""
        return {
            "episode"       : self._episode,
            "step"          : self._step_count,
            "episode_reward": self._episode_reward,
            "traffic"       : self._traffic_preset,
            "map"           : self.carla_map.name.split("/")[-1],
            "speed"         : self._get_speed(),
        }