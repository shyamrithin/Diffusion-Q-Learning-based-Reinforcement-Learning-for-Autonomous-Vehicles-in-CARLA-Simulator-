# ==========================================================
# signal_aware_wrapper.py
# A rule-based SUPERVISORY layer that wraps the learned DQL-E
# driving policy for the traffic-signal demonstration.
#
# IMPORTANT — HONEST SCOPE:
#   The DRIVING (steering, base throttle, traffic avoidance) is
#   produced by the trained DQL-E policy. This wrapper does NOT
#   drive. It only GATES the policy's throttle for two things the
#   learned agent was never trained to do:
#     1. Traffic-signal compliance (red / yellow / green)
#     2. Stopping behind stopped traffic (forward clearance)
#   Lateral control (steering) is passed through untouched.
#
#   This is a rule-based supervisor, NOT learned signal handling.
#   Any demo using it must be described that way.
#
# GATING MODEL:
#   final_throttle = agent_throttle * light_gate * clearance_gate
#   final_brake    = max(agent_brake, forced_brake)
#   final_steer    = agent_steer                      (untouched)
#
# This is the SIMPLE baseline version: nearest-ahead light
# association, in-lane vehicle clearance, distance-ramped gates.
# (Reverted from the more elaborate speed-aware / facing-filter /
#  locked-light experiments, which over-complicated things.)
# ==========================================================

import math
import numpy as np
import carla


class SignalAwareWrapper:
    # ---- light gating ----
    YELLOW_GATE   = 0.5         # professor's spec: yellow -> 0.5
    LIGHT_SLOW_DIST = 18.0      # m: begin slowing for red
    LIGHT_STOP_DIST = 4.0       # m: fully stopped by here
    LIGHT_BRAKE_DIST = 8.0      # m: also brake within this of red

    # ---- forward-clearance gating (vehicle gap, metres) ----
    CLEAR_SLOW_DIST = 14.0      # m: begin easing off when car ahead
    CLEAR_STOP_DIST = 7.0       # m: fully stop behind stopped car
    CLEAR_BRAKE_DIST = 9.0      # m: apply brake when this close

    # ---- forward-vehicle detection (in-lane gate) ----
    LANE_HALF_WIDTH = 2.5       # m: |rel_y| under this = "in my lane"
    MAX_LOOKAHEAD   = 30.0      # m: ignore vehicles further ahead

    # ---- manual traffic-light association ----
    # (vehicle.get_traffic_light() does not attach lights to the ego
    #  in this env, so we find the nearest light AHEAD ourselves)
    LIGHT_SEARCH_RANGE = 40.0   # m: only consider lights within this
    LIGHT_AHEAD_MIN    = 0.0    # m: forward projection must exceed this
    # the governing light's stop-waypoint must face roughly the way the
    # ego heads (head-on approach). Rejects lights turned into mid-
    # junction whose stop line is crosswise (~90 deg).
    LIGHT_ALIGN_MAX_DEG = 45.0
    # forward lane lookahead: walk this many steps of this length along
    # the ego's path to also consider lights on lanes just ahead, so
    # the gate engages early enough to stop AT the line (~5 x 5 m = 25 m).
    LANE_LOOKAHEAD_STEPS = 5
    LANE_LOOKAHEAD_STEP_M = 5.0

    # ---- misc ----
    EPS = 1e-3

    # genuine turning is measured by real heading change per call (deg),
    # not DQL-E's noisy instantaneous steer. ~3 deg/call sustained is a
    # real turn; straight-line jitter stays well below this.
    TURN_YAWRATE_THRESH = 3.0
    # settle period (calls) after turning ends before lights re-engage
    # on a freshly-picked light (~1.5 s at 20 Hz)
    LIGHT_SWITCH_GRACE = 30

    def __init__(self, vehicle, traffic_manager, world=None,
                 verbose=False):
        self.vehicle = vehicle
        self.traffic = traffic_manager   # rlcarla TrafficManager
        self.world = world or vehicle.get_world()
        self.verbose = verbose
        self._lights_cache = None        # filled lazily
        self._last_light_id = None       # last governing light id
        self._grace_left = 0             # grace-period counter
        self._prev_yaw = None            # for measuring real turn rate
        try:
            self._map = self.world.get_map()
        except Exception:
            self._map = None

    # ------------------------------------------------------------
    def _forward_clearance(self):
        """Return (distance_m, closing_speed) to the nearest NPC the
        ego could hit: ahead of the ego and within a forward corridor
        that widens slightly with distance (so cars on a curve / while
        turning are not missed). closing_speed>0 means approaching.
        Returns (MAX_LOOKAHEAD, 0.0) if the path is clear."""
        if self.traffic is None:
            return self.MAX_LOOKAHEAD, 0.0
        try:
            nearby = self.traffic.get_nearby_vehicles(
                self.vehicle, n=12, max_range=self.MAX_LOOKAHEAD + 10)
        except Exception:
            return self.MAX_LOOKAHEAD, 0.0
        best_d = self.MAX_LOOKAHEAD
        best_clos = 0.0
        for v in nearby:
            rel_x = float(v.get("rel_x", 0.0))   # + = ahead
            rel_y = float(v.get("rel_y", 0.0))   # lateral offset
            if rel_x <= 0.5:                     # not ahead
                continue
            # corridor half-width grows with distance so a car ahead
            # on a bend is still "in path"
            corridor = self.LANE_HALF_WIDTH + 0.15 * rel_x
            if abs(rel_y) > corridor:
                continue
            if rel_x < best_d:
                best_d = rel_x
                rel_vx = float(v.get("rel_vx", 0.0))
                best_clos = max(0.0, -rel_vx)   # closing speed
        return best_d, best_clos

    def _forward_clearance_m(self):
        d, _ = self._forward_clearance()
        return d

    # ------------------------------------------------------------
    def _all_lights(self):
        """Cache the world's traffic-light actors."""
        if getattr(self, "_lights_cache", None) is None:
            try:
                self._lights_cache = list(
                    self.world.get_actors().filter(
                        "traffic.traffic_light*"))
            except Exception:
                self._lights_cache = []
        return self._lights_cache

    def _ego_lane_light(self):
        """Lane-based association (CARLA's authoritative mapping):
        find the traffic light that governs the ego's CURRENT lane,
        nearest ahead. Returns (light_actor, distance_m) or (None, inf).

        Uses light.get_affected_lane_waypoints()/get_stop_waypoints()
        and matches the waypoint whose (road_id, lane_id) equals the
        ego's lane. This is geometry-independent and correct across
        junctions (unlike nearest-ahead heuristics)."""
        if self._map is None:
            return None, float("inf")
        lights = self._all_lights()
        if not lights:
            return None, float("inf")
        try:
            ewp = self._map.get_waypoint(
                self.vehicle.get_transform().location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
        except Exception:
            ewp = None
        if ewp is None:
            return None, float("inf")
        e_road, e_lane = ewp.road_id, ewp.lane_id
        # use the ego's LANE direction (stable) rather than the
        # vehicle's instantaneous heading (which swings through
        # alignment during a turn and causes false matches).
        lane_yaw = ewp.transform.rotation.yaw
        lane_rad = math.radians(lane_yaw)
        lfx, lfy = math.cos(lane_rad), math.sin(lane_rad)

        # LOOKAHEAD: collect the (road_id, lane_id) of the ego's lane
        # AND the lanes ahead along its path (walk forward ~25 m). This
        # lets us see a light governing the lane we are ABOUT to enter
        # from farther back, so the gate engages in time to stop at the
        # line instead of detecting it only once we're on top of it.
        lane_keys = {(e_road, e_lane)}
        try:
            cur = ewp
            for _ in range(self.LANE_LOOKAHEAD_STEPS):
                nxts = cur.next(self.LANE_LOOKAHEAD_STEP_M)
                if not nxts:
                    break
                cur = nxts[0]
                lane_keys.add((cur.road_id, cur.lane_id))
        except Exception:
            pass

        tf = self.vehicle.get_transform()
        loc = tf.location
        best_tl, best_dist = None, float("inf")
        for tl in lights:
            try:
                wps = tl.get_affected_lane_waypoints()
            except Exception:
                wps = []
            if not wps:
                try:
                    wps = tl.get_stop_waypoints()
                except Exception:
                    wps = []
            for wp in wps:
                if (wp.road_id, wp.lane_id) not in lane_keys:
                    continue
                # the stop-waypoint must face roughly the same way the
                # ego's LANE runs (head-on approach). A light turned
                # into mid-junction sits on a lane whose direction is
                # crosswise, so this rejects it even while the car's
                # instantaneous heading swings through alignment.
                swyaw = wp.transform.rotation.yaw
                rel_yaw = ((swyaw - lane_yaw + 180) % 360) - 180
                if abs(rel_yaw) > self.LIGHT_ALIGN_MAX_DEG:
                    continue
                wl = wp.transform.location
                dx, dy = wl.x - loc.x, wl.y - loc.y
                ahead = dx * lfx + dy * lfy   # along lane direction
                dist = math.hypot(dx, dy)
                # accept lights governing our lane (or an upcoming lane)
                # whose stop line is ahead (or only just behind, so we
                # hold at the line once stopped on it)
                if ahead > -3.0 and dist < best_dist:
                    best_dist = dist
                    best_tl = tl
        return best_tl, best_dist

    def nearest_light_ahead(self):
        """Return the traffic-light ACTOR governing the ego's lane
        (lane-based association only). Returns (None, inf) if no light
        governs the ego's current lane ahead — in which case the gate
        stays OPEN. We deliberately do NOT fall back to nearest-ahead
        geometry: that fallback would grab an unrelated nearby red
        (e.g. a cross-street light) once the ego passes its own green
        stop line, freezing the car for no reason."""
        return self._ego_lane_light()

    def debug_light_geom(self):
        """Diagnostic: for the lane-associated light, return a string
        describing ego-vs-stopline geometry during a REAL driven run
        (so we tune against actual turn situations, not a probe driving
        into a wall). Returns '' if no governing light."""
        if self._map is None:
            return ""
        lights = self._all_lights()
        if not lights:
            return ""
        try:
            ewp = self._map.get_waypoint(
                self.vehicle.get_transform().location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
        except Exception:
            ewp = None
        if ewp is None:
            return ""
        loc = self.vehicle.get_transform().location
        veh_yaw = self.vehicle.get_transform().rotation.yaw
        best = None
        for tl in lights:
            try:
                wps = tl.get_affected_lane_waypoints()
            except Exception:
                wps = []
            if not wps:
                try:
                    wps = tl.get_stop_waypoints()
                except Exception:
                    wps = []
            for wp in wps:
                if wp.road_id != ewp.road_id or wp.lane_id != ewp.lane_id:
                    continue
                sl = wp.transform.location
                syaw = math.radians(wp.transform.rotation.yaw)
                sfx, sfy = math.cos(syaw), math.sin(syaw)
                ex, ey = loc.x - sl.x, loc.y - sl.y
                s = -(ex * sfx + ey * sfy)   # +ve = upstream
                dist = math.hypot(ex, ey)
                rel_vy = ((veh_yaw - wp.transform.rotation.yaw + 180)
                          % 360) - 180
                if best is None or dist < best[0]:
                    best = (dist, tl.id,
                            str(tl.get_state()).split(".")[-1],
                            s, rel_vy, ewp.road_id, ewp.lane_id)
        if best is None:
            return ""
        dist, tid, st, s, rel_vy, road, lane = best
        return (f"light={tid} {st} road={road} lane={lane} "
                f"s={s:+.1f} dist={dist:.1f} veh_vs_stop={rel_vy:+.0f}")

    def _light_state_and_distance(self):
        """Return (state, distance_m) of the light governing the ego's
        lane via lane-based association (with geometric fallback), or
        (None, inf) if none applies."""
        tl, dist = self.nearest_light_ahead()
        if tl is None:
            return None, float("inf")
        try:
            return tl.get_state(), dist
        except Exception:
            return None, float("inf")

    # ------------------------------------------------------------
    def _light_gate(self, steer=0.0):
        """Throttle multiplier from the traffic light, with a
        distance-based graceful stop on red. Also returns whether
        a brake should be forced.

        TURN SUPPRESSION: while the car is actually TURNING (measured
        by real heading change, not DQL-E's noisy instantaneous steer),
        a newly-picked light on the lane being turned into would freeze
        the car mid-turn. So while genuinely turning, and for a short
        settle period after, we ignore a light that only just became
        the governing one. A light we were ALREADY obeying before the
        turn is still obeyed."""
        tl, dist = self.nearest_light_ahead()
        tl_id = tl.id if tl is not None else None

        # measure REAL heading change since last call (deg). DQL-E's
        # steer output is noisy, so we use actual yaw delta instead.
        cur_yaw = self.vehicle.get_transform().rotation.yaw
        if self._prev_yaw is None:
            yaw_rate = 0.0
        else:
            yaw_rate = abs(((cur_yaw - self._prev_yaw + 180) % 360) - 180)
        self._prev_yaw = cur_yaw

        is_new = (tl_id is not None and tl_id != self._last_light_id)

        # genuinely turning? (sustained heading change per call)
        if yaw_rate >= self.TURN_YAWRATE_THRESH:
            self._grace_left = self.LIGHT_SWITCH_GRACE
        elif self._grace_left > 0:
            self._grace_left -= 1

        # ignore a NEW light only while turning / settling after a turn
        if is_new and self._grace_left > 0:
            return 1.0, False

        # committed to this light now
        self._last_light_id = tl_id

        if tl is None:
            return 1.0, False  # open road
        try:
            state = tl.get_state()
        except Exception:
            return 1.0, False

        if state == carla.TrafficLightState.Green:
            return 1.0, False
        if state == carla.TrafficLightState.Yellow:
            return self.YELLOW_GATE, False
        # Red (or Off/Unknown treated as caution->stop)
        if dist <= self.LIGHT_STOP_DIST:
            gate = 0.0
        elif dist >= self.LIGHT_SLOW_DIST:
            gate = 1.0
        else:
            span = self.LIGHT_SLOW_DIST - self.LIGHT_STOP_DIST
            gate = (dist - self.LIGHT_STOP_DIST) / max(span, self.EPS)
        force_brake = dist <= self.LIGHT_BRAKE_DIST
        return float(np.clip(gate, 0.0, 1.0)), force_brake

    # ------------------------------------------------------------
    def _clearance_gate(self):
        """Throttle multiplier from forward clearance; brake flag
        when close to the vehicle ahead. Extends slow/brake distances
        when closing fast (need more room to stop)."""
        d, closing = self._forward_clearance()
        margin = 0.7 * closing          # extra room per m/s closing
        slow_d  = self.CLEAR_SLOW_DIST + margin
        stop_d  = self.CLEAR_STOP_DIST + 0.5 * margin
        brake_d = self.CLEAR_BRAKE_DIST + 0.5 * margin
        if d >= slow_d:
            gate = 1.0
        elif d <= stop_d:
            gate = 0.0
        else:
            span = slow_d - stop_d
            gate = (d - stop_d) / max(span, self.EPS)
        force_brake = d <= brake_d
        return float(np.clip(gate, 0.0, 1.0)), force_brake

    # ------------------------------------------------------------
    def apply(self, action):
        """Gate a raw (throttle, steer, brake) action from DQL-E.
        Steering is passed through untouched."""
        throttle = float(action[0])
        steer    = float(action[1])
        brake    = float(action[2])

        lg, lbrake = self._light_gate(steer=steer)
        cg, cbrake = self._clearance_gate()

        gate = lg * cg
        final_throttle = throttle * gate

        if lbrake or cbrake or gate < 0.05:
            # WE want to stop (red light / blocked): brake hard,
            # no throttle.
            final_throttle = 0.0
            final_brake = 0.8
        elif gate < 0.85:
            # ramping down (approaching a car / light): ease off
            # throttle AND apply a gentle, increasing brake so the car
            # actively decelerates instead of coasting in.
            final_throttle = throttle * gate
            final_brake = float(np.clip((0.85 - gate) * 0.7, 0.0, 0.5))
        else:
            # open road (green + clear): the car should MOVE. DQL-E
            # often emits throttle AND brake together; the env applies
            # brake directly, so that spurious brake fights the
            # throttle and can hold the car (especially from a stop on
            # green). Suppress DQL-E's brake when it clearly wants to
            # accelerate; keep it only if it is genuinely braking far
            # more than accelerating.
            if final_throttle > 0.2 and brake < final_throttle + 0.2:
                final_brake = 0.0          # let throttle win
            else:
                final_brake = brake        # genuine brake, keep it

        if self.verbose:
            print(f"[wrap] light_gate={lg:.2f} clear_gate={cg:.2f} "
                  f"thr {throttle:.2f}->{final_throttle:.2f} "
                  f"brake={final_brake:.2f}")

        return final_throttle, steer, final_brake