import math
import numpy as np
import carla

W = {
    # Core locomotion
    "forward_progress"    :  1.0,   # reduced — lane discipline dominant
    "raw_speed"           :  0.2,
    "speed_target"        :  1.0,

    # Lane discipline — PRIMARY signals
    "lane_center"         :  2.5,   # most important
    "lane_offset_penalty" :  6.0,
    "yaw_align"           :  2.0,
    "yaw_penalty"         :  8.0,

    # Curve following
    "curve_steer_penalty" :  8.0,
    "curve_steer_reward"  :  3.0,
    "wrong_steer_penalty" :  6.0,

    # Smoothness
    "steer_jerk"          :  0.3,
    "throttle_jerk"       :  0.1,
    "brake_jerk"          :  0.1,
    "comfort"             :  0.3,

    # Safety
    "collision"           : 50.0,
    "off_road"            : 20.0,
    "wrong_lane"          : 40.0,   # very high — single lane critical
    "wrong_way"           : 30.0,
    "stuck"               :  0.5,

    # Proactive collision avoidance — LiDAR proximity
    "proximity_penalty"   :  8.0,   # scales with closeness
    "proximity_brake"     :  2.0,   # reward for braking near obstacle

    # Traffic compliance
    "red_light_penalty"   : 25.0,
    "speed_limit_excess"  :  2.0,

    # Survival
    "alive"               :  0.1,
}

# Thresholds
SPEED_TARGET_MIN      = 3.0
SPEED_TARGET_MAX      = 9.0
LANE_OFFSET_OK        = 0.2    # very tight — was 0.3
LANE_OFFSET_BAD       = 0.5    # was 0.8
YAW_OK_DEG            = 8.0
YAW_BAD_DEG           = 35.0
CURVE_ANGLE_THRESHOLD = 8.0
STUCK_SPEED_MS        = 0.5

# LiDAR proximity thresholds
# LiDAR range = 30m, normalised 0-1
# 0.3 = 9m, 0.2 = 6m, 0.1 = 3m
PROXIMITY_WARN   = 0.3   # 9m — start penalising
PROXIMITY_DANGER = 0.15  # 4.5m — strong penalty
PROXIMITY_CRITICAL = 0.08  # 2.4m — near collision

# LiDAR obs indices in 141D single frame
LIDAR_START = 15
LIDAR_END   = 87

# Forward-facing bins (±15° = bins 0,1,2 and 70,71)
FORWARD_BINS = [0, 1, 2, 70, 71]
# Side bins for lane change detection
LEFT_BINS    = [17, 18, 19]   # ~90° left
RIGHT_BINS   = [53, 54, 55]   # ~90° right


class RewardCalculator:

    def __init__(self):
        self.prev_steer           = 0.0
        self.prev_throttle        = 0.0
        self.prev_brake           = 0.0
        self.prev_speed           = 0.0
        self.stuck_count          = 0
        self._red_light_penalised = False

    def reset(self):
        self.prev_steer           = 0.0
        self.prev_throttle        = 0.0
        self.prev_brake           = 0.0
        self.prev_speed           = 0.0
        self.stuck_count          = 0
        self._red_light_penalised = False

    def compute(
        self,
        vehicle,
        carla_map,
        obs,
        action,
        collision  = False,
        wrong_way  = False,
        wrong_lane = False,
    ):
        reward = 0.0
        info   = {}

        throttle = float(action[0])
        steer    = float(action[1])
        brake    = float(action[2])

        tf      = vehicle.get_transform()
        vel     = vehicle.get_velocity()
        forward = tf.get_forward_vector()

        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        forward_speed = (
            vel.x * forward.x +
            vel.y * forward.y +
            vel.z * forward.z
        )

        loc = vehicle.get_location()
        wp  = carla_map.get_waypoint(
            loc,
            project_to_road = True,
            lane_type       = carla.LaneType.Driving
        )

        if wp is None:
            reward -= W["off_road"]
            info["off_road"] = -W["off_road"]
            return float(reward), info

        lane_center  = wp.transform.location
        lane_yaw     = wp.transform.rotation.yaw
        vehicle_yaw  = tf.rotation.yaw

        dx            = loc.x - lane_center.x
        dy            = loc.y - lane_center.y
        lane_right_x  = math.cos(math.radians(lane_yaw + 90))
        lane_right_y  = math.sin(math.radians(lane_yaw + 90))
        signed_offset = dx * lane_right_x + dy * lane_right_y
        lane_offset   = abs(signed_offset)
        yaw_error     = _wrap_angle(vehicle_yaw - lane_yaw)

        future_angles = _get_future_angles(
            wp, vehicle_yaw, steps=5, dist=4.0
        )
        max_curve_angle = (
            max(abs(a) for a in future_angles)
            if future_angles else 0.0
        )
        curve_sign = (
            np.sign(future_angles[0])
            if future_angles else 0.0
        )
        is_curve = max_curve_angle > CURVE_ANGLE_THRESHOLD

        # Extract LiDAR from obs
        # obs is 564D stacked — first frame lidar is at
        # LIDAR_START:LIDAR_END
        lidar = obs[LIDAR_START:LIDAR_END]

        # --------------------------------------------------
        # 1. ALIVE
        # --------------------------------------------------
        reward += W["alive"]
        info["alive"] = W["alive"]

        # --------------------------------------------------
        # 2. FORWARD PROGRESS
        # Scaled down on curves
        # --------------------------------------------------
        progress_scale = 0.5 if is_curve else 1.0
        r_progress = (
            forward_speed * W["forward_progress"] * progress_scale
        )
        reward    += r_progress
        info["forward_progress"] = r_progress

        # --------------------------------------------------
        # 3. SPEED REGULATION
        # --------------------------------------------------
        reward += speed * W["raw_speed"]
        info["raw_speed"] = speed * W["raw_speed"]

        if SPEED_TARGET_MIN <= speed <= SPEED_TARGET_MAX:
            reward += W["speed_target"]
            info["speed_target"] = W["speed_target"]
        else:
            info["speed_target"] = 0.0

        speed_limit_ms = vehicle.get_speed_limit() / 3.6
        if speed > speed_limit_ms * 1.1:
            excess  = (speed - speed_limit_ms) * W["speed_limit_excess"]
            reward -= excess
            info["speed_limit_excess"] = -excess
        else:
            info["speed_limit_excess"] = 0.0

        # --------------------------------------------------
        # 4. LANE CENTERING — PRIMARY SIGNAL
        # --------------------------------------------------
        if lane_offset < LANE_OFFSET_OK:
            r_center = W["lane_center"]
        elif lane_offset < LANE_OFFSET_BAD:
            frac     = (
                (lane_offset - LANE_OFFSET_OK) /
                (LANE_OFFSET_BAD - LANE_OFFSET_OK)
            )
            r_center = W["lane_center"] * (1.0 - frac)
        else:
            r_center = -W["lane_offset_penalty"] * (
                lane_offset - LANE_OFFSET_BAD + 1.0
            )

        reward += r_center
        info["lane_center"] = r_center

        # --------------------------------------------------
        # 5. YAW ALIGNMENT
        # Double weight on curves
        # --------------------------------------------------
        abs_yaw    = abs(yaw_error)
        yaw_weight = 2.0 if is_curve else 1.0

        if abs_yaw < YAW_OK_DEG:
            r_yaw = W["yaw_align"] * yaw_weight
        elif abs_yaw < YAW_BAD_DEG:
            frac  = (abs_yaw - YAW_OK_DEG) / (YAW_BAD_DEG - YAW_OK_DEG)
            r_yaw = W["yaw_align"] * (1.0 - frac) * yaw_weight
        else:
            r_yaw = -W["yaw_penalty"] * yaw_weight

        reward += r_yaw
        info["yaw_align"] = r_yaw

        # --------------------------------------------------
        # 6. CURVE FOLLOWING
        # --------------------------------------------------
        if is_curve:
            steer_in_curve_dir = (np.sign(steer) == curve_sign)
            curve_sharpness    = max_curve_angle / 90.0

            if abs(steer) < 0.05:
                r_curve       = -W["curve_steer_penalty"] * (
                    1.0 + curve_sharpness
                )
                info["curve"] = "no_steer"
            elif steer_in_curve_dir:
                r_curve       = W["curve_steer_reward"] * (
                    1.0 + curve_sharpness
                )
                info["curve"] = "correct"
            else:
                r_curve       = -W["wrong_steer_penalty"] * (
                    1.0 + curve_sharpness
                )
                info["curve"] = "wrong_dir"

            reward += r_curve
            info["curve_reward"] = r_curve
        else:
            info["curve_reward"] = 0.0
            info["curve"]        = "straight"

        # --------------------------------------------------
        # 7. SMOOTHNESS
        # --------------------------------------------------
        r_steer_jerk    = -abs(steer - self.prev_steer) * W["steer_jerk"]
        r_throttle_jerk = -abs(throttle - self.prev_throttle) * W["throttle_jerk"]
        r_brake_jerk    = -abs(brake - self.prev_brake) * W["brake_jerk"]

        reward += r_steer_jerk + r_throttle_jerk + r_brake_jerk
        info["steer_jerk"]    = r_steer_jerk
        info["throttle_jerk"] = r_throttle_jerk
        info["brake_jerk"]    = r_brake_jerk

        total_jerk = (
            abs(steer    - self.prev_steer) +
            abs(throttle - self.prev_throttle) +
            abs(brake    - self.prev_brake)
        )
        if total_jerk < 0.1:
            reward += W["comfort"]
            info["comfort"] = W["comfort"]
        else:
            info["comfort"] = 0.0

        # --------------------------------------------------
        # 8. STUCK
        # --------------------------------------------------
        if speed < STUCK_SPEED_MS:
            self.stuck_count += 1
            reward           -= W["stuck"]
            info["stuck"]     = -W["stuck"]
        else:
            self.stuck_count  = 0
            info["stuck"]     = 0.0

        # --------------------------------------------------
        # 9. WRONG LANE
        # --------------------------------------------------
        if wrong_lane:
            reward -= W["wrong_lane"]
            info["wrong_lane"] = -W["wrong_lane"]
        else:
            info["wrong_lane"] = 0.0

        # --------------------------------------------------
        # 10. WRONG WAY
        # --------------------------------------------------
        if wrong_way:
            reward -= W["wrong_way"]
            info["wrong_way"] = -W["wrong_way"]
        else:
            info["wrong_way"] = 0.0

        # --------------------------------------------------
        # 11. OFF ROAD
        # --------------------------------------------------
        wp_raw = carla_map.get_waypoint(
            loc, project_to_road=False
        )
        if wp_raw is None:
            reward -= W["off_road"]
            info["off_road"] = -W["off_road"]
        else:
            info["off_road"] = 0.0

        # --------------------------------------------------
        # 12. COLLISION
        # --------------------------------------------------
        if collision:
            reward -= W["collision"]
            info["collision"] = -W["collision"]
        else:
            info["collision"] = 0.0

        # --------------------------------------------------
        # 13. PROACTIVE COLLISION AVOIDANCE — LiDAR proximity
        # Option C: both proximity penalty + brake reward
        # --------------------------------------------------
        forward_min = float(np.min(lidar[FORWARD_BINS]))

        if forward_min < PROXIMITY_CRITICAL:
            # Very close — heavy penalty
            r_proximity = -W["proximity_penalty"] * (
                (PROXIMITY_WARN - forward_min) / PROXIMITY_WARN
            ) * 3.0
            info["proximity"] = "critical"

        elif forward_min < PROXIMITY_DANGER:
            # Getting close — medium penalty
            r_proximity = -W["proximity_penalty"] * (
                (PROXIMITY_WARN - forward_min) / PROXIMITY_WARN
            ) * 1.5
            info["proximity"] = "danger"

        elif forward_min < PROXIMITY_WARN:
            # Obstacle detected — light penalty
            r_proximity = -W["proximity_penalty"] * (
                (PROXIMITY_WARN - forward_min) / PROXIMITY_WARN
            )
            info["proximity"] = "warn"

        else:
            r_proximity = 0.0
            info["proximity"] = "clear"

        reward += r_proximity
        info["proximity_reward"] = r_proximity

        # Reward for braking when obstacle is close
        # Agent learns to slow down proactively
        if forward_min < PROXIMITY_WARN and brake > 0.2:
            r_brake = W["proximity_brake"] * brake * (
                1.0 - forward_min / PROXIMITY_WARN
            )
            reward += r_brake
            info["proximity_brake"] = r_brake
        else:
            info["proximity_brake"] = 0.0

        # --------------------------------------------------
        # 14. RED LIGHT — CARLA API
        # --------------------------------------------------
        tl_state = vehicle.get_traffic_light_state()
        at_red   = (tl_state == carla.TrafficLightState.Red)

        if at_red and forward_speed > 1.0:
            if not self._red_light_penalised:
                reward                    -= W["red_light_penalty"]
                info["red_light"]          = -W["red_light_penalty"]
                self._red_light_penalised  = True
        else:
            if not at_red:
                self._red_light_penalised = False
            info["red_light"] = 0.0

        # --------------------------------------------------
        # UPDATE PREV
        # --------------------------------------------------
        self.prev_steer    = steer
        self.prev_throttle = throttle
        self.prev_brake    = brake
        self.prev_speed    = speed

        return float(reward), info


def _wrap_angle(angle):
    while angle >  180: angle -= 360
    while angle < -180: angle += 360
    return angle


def _get_future_angles(wp, vehicle_yaw, steps=5, dist=4.0):
    angles  = []
    current = wp
    for _ in range(steps):
        nexts = current.next(dist)
        if not nexts:
            break
        current = nexts[0]
        rel_yaw = _wrap_angle(
            current.transform.rotation.yaw - vehicle_yaw
        )
        angles.append(rel_yaw)
    return angles