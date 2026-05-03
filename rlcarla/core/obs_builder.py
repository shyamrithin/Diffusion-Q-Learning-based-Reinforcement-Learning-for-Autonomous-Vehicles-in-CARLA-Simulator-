import math
import numpy as np
import carla
from collections import deque


SINGLE_OBS_DIM = 141
N_STACK        = 4
OBS_DIM        = SINGLE_OBS_DIM * N_STACK   # 564

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


class FrameStack:
    """Rolling buffer of N_STACK frames concatenated."""

    def __init__(self, n_stack=N_STACK, obs_dim=SINGLE_OBS_DIM):
        self.n_stack = n_stack
        self.obs_dim = obs_dim
        self._frames = deque(maxlen=n_stack)

    def reset(self, first_obs):
        for _ in range(self.n_stack):
            self._frames.append(first_obs.copy())
        return self._get()

    def step(self, obs):
        self._frames.append(obs.copy())
        return self._get()

    def _get(self):
        return np.concatenate(
            list(self._frames), axis=0
        ).astype(np.float32)


class ObservationBuilder:

    def __init__(self):
        self._prev_vel   = carla.Vector3D(0, 0, 0)
        self._prev_yaw   = 0.0
        self._dt         = 0.05

    def reset(self, vehicle):
        self._prev_vel = vehicle.get_velocity()
        self._prev_yaw = vehicle.get_transform().rotation.yaw

    def build(self, vehicle, carla_map, lidar_sensor,
              traffic_manager=None):

        obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        ptr = 0

        tf      = vehicle.get_transform()
        vel     = vehicle.get_velocity()
        control = vehicle.get_control()
        loc     = tf.location
        yaw     = tf.rotation.yaw

        # --------------------------------------------------
        # 1. EGO STATE (9D)
        # --------------------------------------------------
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        cos_y    = math.cos(math.radians(yaw))
        sin_y    = math.sin(math.radians(yaw))
        vx_local =  cos_y * vel.x + sin_y * vel.y
        vy_local = -sin_y * vel.x + cos_y * vel.y

        yaw_rate = _wrap_angle(yaw - self._prev_yaw) / self._dt

        dvx      = vel.x - self._prev_vel.x
        dvy      = vel.y - self._prev_vel.y
        ax_local =  cos_y * (dvx/self._dt) + sin_y * (dvy/self._dt)
        ay_local = -sin_y * (dvx/self._dt) + cos_y * (dvy/self._dt)

        obs[ptr+0] = _clip_norm(speed,    MAX_SPEED)
        obs[ptr+1] = _clip_norm(vx_local, MAX_SPEED)
        obs[ptr+2] = _clip_norm(vy_local, MAX_SPEED)
        obs[ptr+3] = _clip_norm(yaw_rate, MAX_YAW_RATE)
        obs[ptr+4] = float(control.steer)
        obs[ptr+5] = float(control.throttle)
        obs[ptr+6] = float(control.brake)
        obs[ptr+7] = _clip_norm(ax_local, MAX_ACCEL)
        obs[ptr+8] = _clip_norm(ay_local, MAX_ACCEL)
        ptr += 9

        self._prev_vel = vel
        self._prev_yaw = yaw

        # --------------------------------------------------
        # 2. LANE INFO (4D)
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
        # 3. TRAFFIC STATE (2D)
        # --------------------------------------------------
        tl_state         = vehicle.get_traffic_light_state()
        is_red           = float(
            tl_state == carla.TrafficLightState.Red
        )
        speed_limit_ms   = vehicle.get_speed_limit() / 3.6
        speed_limit_norm = _clip_norm(speed_limit_ms, MAX_SPEED_LIMIT)

        obs[ptr+0] = is_red
        obs[ptr+1] = speed_limit_norm
        ptr += 2

        # --------------------------------------------------
        # 4. LIDAR HISTOGRAM (72D)
        # --------------------------------------------------
        histogram       = lidar_sensor.get_histogram()
        obs[ptr:ptr+72] = histogram
        ptr += 72

        # --------------------------------------------------
        # 5. NEARBY VEHICLES (20D)
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
        # 6. FUTURE WAYPOINTS (34D) — SAME LANE ENFORCED
        # --------------------------------------------------
        future_wps = _get_future_waypoints_same_lane(
            loc, yaw, wp, N_WAYPOINTS, WP_SPACING
        )

        for i in range(N_WAYPOINTS):
            base = ptr + i * 2
            if i < len(future_wps):
                rx, ry      = future_wps[i]
                obs[base+0] = _clip_norm(rx, MAX_WP_DIST)
                obs[base+1] = _clip_norm(ry, MAX_WP_DIST)
            else:
                obs[base+0] = _clip_norm(
                    (i+1) * WP_SPACING, MAX_WP_DIST
                )
                obs[base+1] = 0.0
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


def _get_future_waypoints_same_lane(
    ego_loc, ego_yaw, start_wp, n, spacing
):
    """
    Walk n waypoints ahead staying in the SAME lane.
    Never switches lanes at intersections.
    This is the key to single-lane following behaviour.
    """
    if start_wp is None:
        return []

    cos_y     = math.cos(math.radians(ego_yaw))
    sin_y     = math.sin(math.radians(ego_yaw))
    waypoints = []
    current   = start_wp
    target_lane_id = start_wp.lane_id

    for _ in range(n):
        nexts = current.next(spacing)
        if not nexts:
            break

        # 🔥 Enforce same lane — filter by lane_id
        same_lane = [
            wp for wp in nexts
            if wp.lane_id == target_lane_id
        ]

        if same_lane:
            current = same_lane[0]
        else:
            # At intersection or lane end — take first option
            # but update target lane id to follow through
            current = nexts[0]
            target_lane_id = current.lane_id

        loc   = current.transform.location
        dx    = loc.x - ego_loc.x
        dy    = loc.y - ego_loc.y
        rel_x =  cos_y * dx + sin_y * dy
        rel_y = -sin_y * dx + cos_y * dy
        waypoints.append((rel_x, rel_y))

    return waypoints