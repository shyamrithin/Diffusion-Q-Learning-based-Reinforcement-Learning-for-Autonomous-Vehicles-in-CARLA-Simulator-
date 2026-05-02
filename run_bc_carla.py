import sys
sys.path.append('/home/shyam/Carla/PythonAPI/carla')

from agents.navigation.global_route_planner import GlobalRoutePlanner

import carla
import torch
import torch.nn as nn
import numpy as np
import pygame
import random
import math
import queue

# ==========================================================
# CONFIG
# ==========================================================
WIDTH = 1280
HEIGHT = 720
FPS = 20
TARGET_SPEED = 8.0          # m/s
WAYPOINT_LOOKAHEAD = 8      # route index lookahead
STEER_GAIN = 1.6
MAX_STEER = 0.55
REACHED_DIST = 6.0

# ==========================================================
# MODEL (BC throttle/brake helper)
# ==========================================================
class BCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(307, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BCModel().to(device)
model.load_state_dict(torch.load("bc_model_best.pth", map_location=device))
model.eval()

# ==========================================================
# PYGAME
# ==========================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Goal Navigation | Random Start Random Goal")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 22)

# ==========================================================
# CARLA CONNECT
# ==========================================================
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprints = world.get_blueprint_library()
map_obj = world.get_map()

# ==========================================================
# CLEAN OLD VEHICLES / SENSORS
# ==========================================================
for actor in world.get_actors():
    if actor.type_id.startswith("vehicle") or actor.type_id.startswith("sensor"):
        try:
            actor.destroy()
        except:
            pass

# ==========================================================
# RANDOM START / GOAL
# ==========================================================
spawn_points = map_obj.get_spawn_points()

spawn_point = random.choice(spawn_points)
goal_point = random.choice(spawn_points)

while goal_point.location.distance(spawn_point.location) < 40:
    goal_point = random.choice(spawn_points)

# ==========================================================
# SPAWN VEHICLE
# ==========================================================
vehicle_bp = blueprints.filter("model3")[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

print("Spawned vehicle")
print("Start:", spawn_point.location)
print("Goal :", goal_point.location)

# ==========================================================
# CAMERA
# ==========================================================
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(WIDTH))
camera_bp.set_attribute("image_size_y", str(HEIGHT))
camera_bp.set_attribute("fov", "110")

camera_tf = carla.Transform(
    carla.Location(x=-7, z=3.5),
    carla.Rotation(pitch=-15)
)

camera = world.spawn_actor(camera_bp, camera_tf, attach_to=vehicle)

image_queue = queue.Queue()
camera.listen(image_queue.put)

# ==========================================================
# ROUTE PLANNER
# ==========================================================
grp = GlobalRoutePlanner(map_obj, 2.0)

route = grp.trace_route(
    spawn_point.location,
    goal_point.location
)

print("Route length:", len(route))

route_index = 0

# ==========================================================
# HELPERS
# ==========================================================
def get_speed():
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def normalize_angle_deg(a):
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a

def build_obs():
    obs = np.zeros(307, dtype=np.float32)

    tf = vehicle.get_transform()
    loc = tf.location
    yaw = math.radians(tf.rotation.yaw)

    speed = get_speed()

    # ego
    obs[0] = speed / 15.0
    obs[1] = math.cos(yaw)
    obs[2] = math.sin(yaw)

    # future route points relative frame
    for i in range(6):
        idx = min(route_index + i * 2, len(route) - 1)
        wp = route[idx][0].transform.location

        dx = wp.x - loc.x
        dy = wp.y - loc.y

        rel_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        rel_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        base = 271 + i * 6
        obs[base + 0] = rel_x / 25.0
        obs[base + 1] = rel_y / 25.0

    return obs

def compute_steer():
    global route_index

    tf = vehicle.get_transform()
    loc = tf.location
    yaw = tf.rotation.yaw

    # advance route index when close
    while route_index < len(route) - 2:
        wp_loc = route[route_index][0].transform.location
        if loc.distance(wp_loc) < 4.0:
            route_index += 1
        else:
            break

    target_idx = min(route_index + WAYPOINT_LOOKAHEAD, len(route) - 1)
    target_wp = route[target_idx][0]

    target_loc = target_wp.transform.location

    dx = target_loc.x - loc.x
    dy = target_loc.y - loc.y

    desired_yaw = math.degrees(math.atan2(dy, dx))
    yaw_error = normalize_angle_deg(desired_yaw - yaw)

    steer = (yaw_error / 45.0) * STEER_GAIN
    steer = max(-MAX_STEER, min(MAX_STEER, steer))

    return steer, yaw_error

# ==========================================================
# MAIN LOOP
# ==========================================================
running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        running = False

    # ------------------------------------------
    # Goal reached?
    # ------------------------------------------
    if vehicle.get_location().distance(goal_point.location) < REACHED_DIST:
        print("Goal reached!")
        running = False

    # ------------------------------------------
    # Build observation
    # ------------------------------------------
    obs = build_obs()
    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        action = model(state).cpu().numpy()[0]

    raw_throttle = float(action[0])
    raw_brake = float(action[2])

    speed = get_speed()

    # ------------------------------------------
    # Hybrid steering
    # ------------------------------------------
    steer, yaw_error = compute_steer()

    # ------------------------------------------
    # Speed control + BC assist
    # ------------------------------------------
    if speed < TARGET_SPEED:
        throttle = max(0.35, min(0.55, raw_throttle))
        brake = 0.0
    else:
        throttle = min(0.25, max(0.0, raw_throttle * 0.5))
        brake = min(0.25, max(0.0, raw_brake * 0.5))

    # strong turns slow down slightly
    if abs(steer) > 0.35:
        throttle *= 0.75

    vehicle.apply_control(
        carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
    )

    # ------------------------------------------
    # Render
    # ------------------------------------------
    if not image_queue.empty():
        image = image_queue.get()

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((HEIGHT, WIDTH, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        screen.blit(surface, (0, 0))

    # HUD
    speed_txt = font.render(f"Speed: {speed:.1f} m/s", True, (255,255,255))
    steer_txt = font.render(f"Steer: {steer:.2f}", True, (255,255,255))
    route_txt = font.render(f"WP: {route_index}/{len(route)}", True, (255,255,255))
    goal_txt = font.render(f"Dist to Goal: {vehicle.get_location().distance(goal_point.location):.1f} m", True, (255,255,255))

    screen.blit(speed_txt, (15, 15))
    screen.blit(steer_txt, (15, 45))
    screen.blit(route_txt, (15, 75))
    screen.blit(goal_txt, (15, 105))

    pygame.display.flip()
    clock.tick(FPS)

# ==========================================================
# CLEANUP
# ==========================================================
camera.stop()
camera.destroy()
vehicle.destroy()
pygame.quit()

print("Closed cleanly.")
