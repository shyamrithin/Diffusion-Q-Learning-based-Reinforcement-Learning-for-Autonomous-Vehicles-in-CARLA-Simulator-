# ==========================================================
# rlcarla/core/obs_builder.py
# 141D Fusion Observation Builder + 4-Frame Stack
# Final OBS_DIM = 141 * 4 = 564
# CARLA 0.9.16 compatible
# ==========================================================

import math
import numpy as np
import carla
from collections import deque


# ==========================================================
# OBSERVATION LAYOUT (single frame = 141D)
# ==========================================================
#
#  Component          Dims   Description
#  ─────────────────────────────────────────────────────────
#  ego_state            9    speed, vx, vy, yaw_rate,
#                            steer, throttle, brake, ax, ay
#  lane_info            4    signed_offset, yaw_error,
#                            lane_width, is_junction
#  traffic_state        2    is_red_light, speed_limit_norm
#  lidar               72    polar histogram (5° bins)
#  nearby_vehicles     20    4 vehicles × 5 features
#  waypoints           34    17 future wps × (rel_x, rel_y)
#  ─────────────────────────────────────────────────────────
#  SINGLE FRAME       141
#  STACKED (×4)       564    ← actual input to network
#
# ==========================================================

SINGLE_OBS_DIM = 141
N_STACK        = 4
OBS_DIM        = SINGLE_OBS_DIM * N_STACK   # 564

# Normalisation constants
MAX_SPEED       = 15.0
MAX_ACCEL       = 5.0
MAX_YAW_RATE    = 90.0
MAX_LANE_OFFSET = 3.0
MAX_LANE_WIDTH  = 5.0
MAX_REL_POS     = 50.0
MAX_REL_VEL     = 15.0
MAX_WP_DIST     = 50.0
MAX_SPEED_LIMIT = 20.0

N_NEARBY        = 4
N_WAYPOINTS     = 17
WP_SPACING      = 3.0


# ==========================================================
# FRAME STACK
# ==========================================================
class FrameStack:
    """
    Maintains a rolling buffer of N_STACK frames.
    Returns them concatenated as a single (OBS_DIM,) vector.

    Why frame stacking?
      A single frame gives the network no sense of motion —
      it can't infer velocity, acceleration, or turning rate
      from one snapshot. Stacking 4 frames (0.2 seconds of
      history at 0.05s/tick) lets the network learn dynamics
      without explicit recurrence (no LSTM needed).
    """

    def __init__(self, n_stack=N_STACK, obs_dim=SINGLE_OBS_DIM):
        self.n_stack = n_stack
        self.obs_dim = obs_dim
        self._frames = deque(maxlen=n_stack)

    def reset(self, first_obs):
        """Fill all frames with the first observation."""
        for _ in range(self.n_stack):
            self._frames.append(first_obs.copy())
        return self._get()

    def step(self, obs):
        """Push new obs, return stacked vector."""
        self._frames.append(obs.copy())
        return self._get()

    def _get(self):
        """Concatenate frames oldest → newest."""
        return np.concatenate(list(self._frames), axis=0).astype(np.float32)


# ==========================================================
# OBSERVATION BUILDER
# ==========================================================
class ObservationBuilder:
    """
    Builds the 141D single-frame observation vector each step.
    Frame stacking is handled by FrameStack above.

    Usage:
        builder = ObservationBuilder()
        stack   = FrameStack()

        # On reset:
        raw_obs = builder.build(vehicle, carla_map, lidar, tm)
        obs     = stack.reset(raw_obs)   # shape (564,)

        # On step:
        raw_obs = builder.build(vehicle, carla_map, lidar, tm)
        obs     = stack.step(raw_obs)    # shape (564,)
    """

    def __init__(self):
        self._prev_vel   = carla.Vector3D(0, 0, 0)
        self._prev_yaw   = 0.0
        self._dt         = 0.05

    def reset(self, vehicle):
        vel              = vehicle.get_velocity()
        self._prev_vel   = vel
        self._prev_yaw   = vehicle.get_transform().rotation.yaw

    def build(
        self,
        vehicle,
        carla_map,
        lidar_sensor,
        traffic_manager = None,
    ):
        obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        ptr = 0

        tf      = vehicle.get_transform()
        vel     = vehicle.get_velocity()
        control = vehicle.get_control()
        loc     = tf.location
        yaw     = tf.rotation.yaw

        # --------------------------------------------------
        # 1. EGO STATE  (9D)
        # --------------------------------------------------
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        cos_y    = math.cos(math.radians(yaw))
        sin_y    = math.sin(math.radians(yaw))
        vx_local =  cos_y * vel.x + sin_y * vel.y
        vy_local = -sin_y * vel.x + cos_y * vel.y

        yaw_rate = _wrap_angle(yaw - self._prev_yaw) / self._dt

        dvx      = vel.x - self._prev_vel.x
        dvy      = vel.y - self._prev_vel.y
        ax_world = dvx / self._dt
        ay_world = dvy / self._dt
        ax_local =  cos_y * ax_world + sin_y * ay_world
        ay_local = -sin_y * ax_world + cos_y * ay_world

        obs[ptr+0] = _clip_norm(speed,      MAX_SPEED)
        obs[ptr+1] = _clip_norm(vx_local,   MAX_SPEED)
        obs[ptr+2] = _clip_norm(vy_local,   MAX_SPEED)
        obs[ptr+3] = _clip_norm(yaw_rate,   MAX_YAW_RATE)
        obs[ptr+4] = float(control.steer)
        obs[ptr+5] = float(control.throttle)
        obs[ptr+6] = float(control.brake)
        obs[ptr+7] = _clip_norm(ax_local,   MAX_ACCEL)
        obs[ptr+8] = _clip_norm(ay_local,   MAX_ACCEL)
        ptr += 9

        self._prev_vel = vel
        self._prev_yaw = yaw

        # --------------------------------------------------
        # 2. LANE INFO  (4D)
        # --------------------------------------------------
        wp = carla_map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if wp is not None:
            lane_center  = wp.transform.location
            lane_yaw     = wp.transform.rotation.yaw
            dx           = loc.x - lane_center.x
            dy           = loc.y - lane_center.y
            lane_right_x = math.cos(math.radians(lane_yaw + 90))
            lane_right_y = math.sin(math.radians(lane_yaw + 90))
            signed_offset= dx * lane_right_x + dy * lane_right_y
            yaw_error    = _wrap_angle(yaw - lane_yaw)
            lane_width   = wp.lane_width
            is_junc      = float(wp.is_junction)
        else:
            signed_offset = MAX_LANE_OFFSET
            yaw_error     = 90.0
            lane_width    = 3.5
            is_junc       = 0.0

        obs[ptr+0] = _clip_norm(signed_offset, MAX_LANE_OFFSET)
        obs[ptr+1] = yaw_error / 90.0
        obs[ptr+2] = _clip_norm(lane_width,    MAX_LANE_WIDTH)
        obs[ptr+3] = is_junc
        ptr += 4

        # --------------------------------------------------
        # 3. TRAFFIC STATE  (2D)
        # --------------------------------------------------
        tl_state         = vehicle.get_traffic_light_state()
        is_red           = float(tl_state == carla.TrafficLightState.Red)
        speed_limit_raw  = vehicle.get_speed_limit()
        speed_limit_ms   = speed_limit_raw / 3.6
        speed_limit_norm = _clip_norm(speed_limit_ms, MAX_SPEED_LIMIT)

        obs[ptr+0] = is_red
        obs[ptr+1] = speed_limit_norm
        ptr += 2

        # --------------------------------------------------
        # 4. LIDAR HISTOGRAM  (72D)
        # --------------------------------------------------
        histogram       = lidar_sensor.get_histogram()
        obs[ptr:ptr+72] = histogram
        ptr += 72

        # --------------------------------------------------
        # 5. NEARBY VEHICLES  (20D = 4 × 5)
        # --------------------------------------------------
        if traffic_manager is not None:
            nearby = traffic_manager.get_nearby_vehicles(
                vehicle, n=N_NEARBY
            )
        else:
            nearby = []

        for i in range(N_NEARBY):
            base = ptr + i * 5
            if i < len(nearby):
                v = nearby[i]
                obs[base+0] = _clip_norm(v["rel_x"],  MAX_REL_POS)
                obs[base+1] = _clip_norm(v["rel_y"],  MAX_REL_POS)
                obs[base+2] = _clip_norm(v["rel_vx"], MAX_REL_VEL)
                obs[base+3] = _clip_norm(v["rel_vy"], MAX_REL_VEL)
                obs[base+4] = _clip_norm(v["dist"],   MAX_REL_POS)
            else:
                obs[base+0] = 0.0
                obs[base+1] = 0.0
                obs[base+2] = 0.0
                obs[base+3] = 0.0
                obs[base+4] = 1.0
        ptr += N_NEARBY * 5

        # --------------------------------------------------
        # 6. FUTURE WAYPOINTS  (34D = 17 × 2)
        # --------------------------------------------------
        future_wps = _get_future_waypoints(
            loc, yaw, wp, N_WAYPOINTS, WP_SPACING
        )

        for i in range(N_WAYPOINTS):
            base = ptr + i * 2
            if i < len(future_wps):
                rx, ry        = future_wps[i]
                obs[base+0]   = _clip_norm(rx, MAX_WP_DIST)
                obs[base+1]   = _clip_norm(ry, MAX_WP_DIST)
            else:
                obs[base+0]   = _clip_norm(
                    (i + 1) * WP_SPACING, MAX_WP_DIST
                )
                obs[base+1]   = 0.0
        ptr += N_WAYPOINTS * 2

        assert ptr == SINGLE_OBS_DIM, \
            f"obs dim mismatch: {ptr} != {SINGLE_OBS_DIM}"

        return obs

    @staticmethod
    def dim():
        return SINGLE_OBS_DIM


# ==========================================================
# HELPERS
# ==========================================================

def _clip_norm(value, max_val):
    return float(np.clip(value / max_val, -1.0, 1.0))


def _wrap_angle(angle):
    while angle >  180: angle -= 360
    while angle < -180: angle += 360
    return angle


def _get_future_waypoints(ego_loc, ego_yaw, start_wp, n, spacing):
    if start_wp is None:
        return []

    cos_y     = math.cos(math.radians(ego_yaw))
    sin_y     = math.sin(math.radians(ego_yaw))
    waypoints = []
    current   = start_wp

    for _ in range(n):
        nexts = current.next(spacing)
        if not nexts:
            break
        current = nexts[0]
        loc     = current.transform.location
        dx      = loc.x - ego_loc.x
        dy      = loc.y - ego_loc.y
        rel_x   =  cos_y * dx + sin_y * dy
        rel_y   = -sin_y * dx + cos_y * dy
        waypoints.append((rel_x, rel_y))

    return waypoints
