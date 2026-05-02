import numpy as np
import carla
import queue


class LidarSensor:
    """
    32-channel 3D LiDAR (Velodyne VLP-32 style).

    Outputs:
      - 72D polar histogram  → observation vector
      - (N,4) point array    → pygame 3D visualization

    All 32 vertical channels compressed into 72 azimuth bins
    via min-distance. Richer obstacle detection, OBS_DIM unchanged.
    """

    CHANNELS       = 32
    RANGE          = 30.0
    PTS_PER_SEC    = 56000     # 32ch × 1750 — dense but manageable
    ROTATION_FREQ  = 20.0
    UPPER_FOV      =  15.0
    LOWER_FOV      = -25.0
    N_BINS         = 72
    SENSOR_HEIGHT  = 1.8

    def __init__(self, vehicle, world):
        self.vehicle    = vehicle
        self.world      = world
        self._queue     = queue.Queue()
        self._sensor    = None
        self._histogram = np.ones(self.N_BINS, dtype=np.float32)
        self._points    = np.zeros((0, 4),     dtype=np.float32)
        self._attach()

    def _attach(self):
        bp = self.world.get_blueprint_library().find(
            "sensor.lidar.ray_cast"
        )
        bp.set_attribute("channels",           str(self.CHANNELS))
        bp.set_attribute("range",              str(self.RANGE))
        bp.set_attribute("points_per_second",  str(self.PTS_PER_SEC))
        bp.set_attribute("rotation_frequency", str(self.ROTATION_FREQ))
        bp.set_attribute("upper_fov",          str(self.UPPER_FOV))
        bp.set_attribute("lower_fov",          str(self.LOWER_FOV))

        tf = carla.Transform(
            carla.Location(x=0.0, z=self.SENSOR_HEIGHT)
        )
        self._sensor = self.world.spawn_actor(
            bp, tf, attach_to=self.vehicle
        )
        self._sensor.listen(self._queue.put)

    def _process(self, raw):
        histogram = np.ones(self.N_BINS, dtype=np.float32)

        pts = np.frombuffer(
            raw.raw_data, dtype=np.float32
        ).reshape(-1, 4)

        if len(pts) == 0:
            return histogram, pts

        x = pts[:, 0]
        y = pts[:, 1]

        dist      = np.sqrt(x**2 + y**2)
        azimuth   = np.degrees(np.arctan2(y, x)) % 360.0
        bin_idx   = np.clip(
            (azimuth / (360.0 / self.N_BINS)).astype(np.int32),
            0, self.N_BINS - 1
        )
        norm_dist = np.clip(dist / self.RANGE, 0.0, 1.0)

        for i in range(self.N_BINS):
            mask = bin_idx == i
            if mask.any():
                histogram[i] = norm_dist[mask].min()

        return histogram, pts

    def _update(self):
        """Pull latest scan — discard stale frames if queue backed up."""
        latest = None
        while not self._queue.empty():
            try:
                latest = self._queue.get_nowait()
            except Exception:
                break

        if latest is not None:
            self._histogram, self._points = self._process(latest)

    def flush(self):
        """
        Empty the queue completely.
        Call after reset to prevent backlog causing FPS dips.
        """
        flushed = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                flushed += 1
            except Exception:
                break
        return flushed

    def get_histogram(self):
        """72D normalised histogram for obs vector."""
        self._update()
        return self._histogram.copy()

    def get_points(self):
        """(N,4) raw points for visualization."""
        self._update()
        return self._points.copy()

    def get_histogram_and_points(self):
        """Both in one call — avoids double processing."""
        self._update()
        return self._histogram.copy(), self._points.copy()

    def destroy(self):
        if self._sensor is not None:
            try:
                self._sensor.stop()
                self._sensor.destroy()
            except Exception:
                pass
            self._sensor = None
