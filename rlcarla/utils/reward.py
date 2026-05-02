import math
import numpy as np
import carla

W = {
    "forward_progress"    :  2.0,
    "raw_speed"           :  0.3,
    "speed_target"        :  1.0,
    "lane_center"         :  1.0,
    "lane_offset_penalty" :  3.0,
    "yaw_align"           :  0.8,
    "yaw_penalty"         :  5.0,
    "curve_steer_penalty" :  3.0,
    "curve_steer_reward"  :  1.0,
    "wrong_steer_penalty" :  2.0,
    "steer_jerk"          :  0.3,
    "throttle_jerk"       :  0.1,
    "brake_jerk"          :  0.1,
    "collision"           : 50.0,
    "off_road"            : 15.0,
    "wrong_lane"          : 20.0,
    "wrong_way"           : 30.0,
    "stuck"               :  0.5,
    "red_light_penalty"   : 25.0,
    "speed_limit_excess"  :  2.0,
    "comfort"             :  0.3,
    "alive"               :  0.1,
}

SPEED_TARGET_MIN      = 4.0
SPEED_TARGET_MAX      = 9.0
LANE_OFFSET_OK        = 0.4
LANE_OFFSET_BAD       = 1.0
YAW_OK_DEG            = 10.0
YAW_BAD_DEG           = 45.0
CURVE_ANGLE_THRESHOLD = 10.0
STUCK_SPEED_MS        = 0.5


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

        future_angles   = _get_future_angles(
            wp, vehicle_yaw, steps=3, dist=5.0
        )
        max_curve_angle = (
            max(abs(a) for a in future_angles)
            if future_angles else 0.0
        )
        curve_sign = (
            np.sign(future_angles[0])
            if future_angles else 0.0
        )

        # --------------------------------------------------
        # 1. ALIVE
        # --------------------------------------------------
        reward += W["alive"]
        info["alive"] = W["alive"]

        # --------------------------------------------------
        # 2. FORWARD PROGRESS
        # --------------------------------------------------
        r_progress = forward_speed * W["forward_progress"]
        reward    += r_progress
        info["forward_progress"] = r_progress

        # --------------------------------------------------
        # 3. SPEED
        # --------------------------------------------------
        r_speed = speed * W["raw_speed"]
        reward += r_speed
        info["raw_speed"] = r_speed

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
        # 4. LANE CENTERING
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
        # --------------------------------------------------
        abs_yaw = abs(yaw_error)

        if abs_yaw < YAW_OK_DEG:
            r_yaw = W["yaw_align"]
        elif abs_yaw < YAW_BAD_DEG:
            frac  = (
                (abs_yaw - YAW_OK_DEG) /
                (YAW_BAD_DEG - YAW_OK_DEG)
            )
            r_yaw = W["yaw_align"] * (1.0 - frac)
        else:
            r_yaw = -W["yaw_penalty"]

        reward += r_yaw
        info["yaw_align"] = r_yaw

        # --------------------------------------------------
        # 6. CURVE FOLLOWING
        # --------------------------------------------------
        if max_curve_angle > CURVE_ANGLE_THRESHOLD:
            steer_in_curve_dir = (np.sign(steer) == curve_sign)

            if abs(steer) < 0.05:
                r_curve       = -W["curve_steer_penalty"]
                info["curve"] = "no_steer"
            elif steer_in_curve_dir:
                r_curve       = W["curve_steer_reward"] * (
                    max_curve_angle / 90.0
                )
                info["curve"] = "correct"
            else:
                r_curve       = -W["wrong_steer_penalty"]
                info["curve"] = "wrong_dir"

            reward += r_curve
            info["curve_reward"] = r_curve
        else:
            info["curve_reward"] = 0.0
            info["curve"]        = "straight"

        # --------------------------------------------------
        # 7. SMOOTHNESS + COMFORT
        # --------------------------------------------------
        r_steer_jerk    = -abs(steer    - self.prev_steer)    * W["steer_jerk"]
        r_throttle_jerk = -abs(throttle - self.prev_throttle) * W["throttle_jerk"]
        r_brake_jerk    = -abs(brake    - self.prev_brake)    * W["brake_jerk"]

        reward += r_steer_jerk + r_throttle_jerk + r_brake_jerk
        info["steer_jerk"]    = r_steer_jerk
        info["throttle_jerk"] = r_throttle_jerk
        info["brake_jerk"]    = r_brake_jerk

        total_jerk = (
            abs(steer    - self.prev_steer)    +
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
        # 13. RED LIGHT
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


def _get_future_angles(wp, vehicle_yaw, steps=3, dist=5.0):
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
