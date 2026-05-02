import numpy as np
import carla
import queue

CAMERA_VIEWS = {
    "third_person": carla.Transform(
        carla.Location(x=-6, z=2.5),
        carla.Rotation(pitch=-12)
    ),
    "driver": carla.Transform(
        carla.Location(x=0.4, z=1.3),
        carla.Rotation(pitch=0)
    ),
    "bird_eye": carla.Transform(
        carla.Location(x=0, z=20),
        carla.Rotation(pitch=-90)
    ),
    "front": carla.Transform(
        carla.Location(x=2.0, z=1.4),
        carla.Rotation(pitch=-5)
    ),
}


class CameraManager:

    def __init__(self, vehicle, world, width=800, height=450, fov=100):
        self.vehicle    = vehicle
        self.world      = world
        self.width      = width
        self.height     = height
        self.fov        = fov
        self._sensor    = None
        self._queue     = queue.Queue()
        self._frame     = np.zeros((height, width, 3), dtype=np.uint8)
        self._view      = "third_person"
        self._intrinsic = self._build_intrinsic()
        self.set_view("third_person")

    def _build_intrinsic(self):
        f  = self.width / (2.0 * np.tan(np.radians(self.fov / 2.0)))
        cx = self.width  / 2.0
        cy = self.height / 2.0
        return np.array([
            [f,  0, cx],
            [0,  f, cy],
            [0,  0,  1]
        ], dtype=np.float64)

    @property
    def intrinsic(self):
        return self._intrinsic

    def set_view(self, view_name):
        if view_name not in CAMERA_VIEWS:
            print(f"[Camera] Unknown view '{view_name}'. Options: {list(CAMERA_VIEWS)}")
            return
        self._destroy_sensor()
        self._view   = view_name
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.width))
        bp.set_attribute("image_size_y", str(self.height))
        bp.set_attribute("fov",          str(self.fov))
        tf           = CAMERA_VIEWS[view_name]
        self._sensor = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
        self._queue  = queue.Queue()
        self._sensor.listen(self._queue.put)

    def get_frame(self):
        try:
            raw         = self._queue.get(timeout=0.05)
            arr         = np.frombuffer(raw.raw_data, dtype=np.uint8)
            arr         = arr.reshape((self.height, self.width, 4))[:, :, :3][:, :, ::-1]
            self._frame = arr
        except queue.Empty:
            pass
        return self._frame.copy()

    def get_transform(self):
        if self._sensor:
            return self._sensor.get_transform()
        return None

    @property
    def current_view(self):
        return self._view

    @property
    def available_views(self):
        return list(CAMERA_VIEWS.keys())

    def _destroy_sensor(self):
        if self._sensor:
            try:
                self._sensor.stop()
                self._sensor.destroy()
            except Exception:
                pass
            self._sensor = None

    def destroy(self):
        self._destroy_sensor()
