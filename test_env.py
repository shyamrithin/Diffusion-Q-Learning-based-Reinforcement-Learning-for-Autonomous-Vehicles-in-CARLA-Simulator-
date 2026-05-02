# manual_drive.py
# Run with: python3 manual_drive.py

import gymnasium as gym
import rlcarla
import numpy as np
import pygame
import carla
import queue
import math

WIDTH = 1280
HEIGHT = 720
FPS = 20

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RLCarla Manual Drive")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# -------------------------------------------------
# Create Environment
# -------------------------------------------------
env = gym.make("RLCarla-v0")
obs, info = env.reset()

base_env = env.unwrapped
vehicle = base_env.vehicle
world = base_env.world
bp_lib = world.get_blueprint_library()

# -------------------------------------------------
# Chase Camera
# -------------------------------------------------
camera_bp = bp_lib.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(WIDTH))
camera_bp.set_attribute("image_size_y", str(HEIGHT))
camera_bp.set_attribute("fov", "110")

camera_tf = carla.Transform(
    carla.Location(x=-7.0, z=3.0),
    carla.Rotation(pitch=-15)
)

camera = world.spawn_actor(camera_bp, camera_tf, attach_to=vehicle)
img_queue = queue.Queue()
camera.listen(img_queue.put)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def get_speed():
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

# -------------------------------------------------
# Main Loop
# -------------------------------------------------
running = True
step = 0

while running:

    # -----------------------------
    # Events
    # -----------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        running = False

    # -----------------------------
    # Controls
    # -----------------------------
    throttle = 0.0
    steer = 0.0
    brake = 0.0

    if keys[pygame.K_w]:
        throttle = 0.55

    if keys[pygame.K_s]:
        brake = 0.55

    if keys[pygame.K_a]:
        steer = -0.45

    if keys[pygame.K_d]:
        steer = 0.45

    # Handbrake style stop
    if keys[pygame.K_SPACE]:
        brake = 1.0
        throttle = 0.0

    action = np.array([throttle, steer, brake], dtype=np.float32)

    # -----------------------------
    # Step Environment
    # -----------------------------
    obs, reward, done, trunc, info = env.step(action)

    # -----------------------------
    # Reset if R pressed
    # -----------------------------
    if keys[pygame.K_r]:
        camera.stop()
        camera.destroy()

        obs, info = env.reset()

        base_env = env.unwrapped
        vehicle = base_env.vehicle
        world = base_env.world

        camera = world.spawn_actor(camera_bp, camera_tf, attach_to=vehicle)
        img_queue = queue.Queue()
        camera.listen(img_queue.put)

    # -----------------------------
    # Render Camera
    # -----------------------------
    if not img_queue.empty():
        image = img_queue.get()

        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((HEIGHT, WIDTH, 4))
        arr = arr[:, :, :3]
        arr = arr[:, :, ::-1]

        surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        screen.blit(surface, (0, 0))

    # -----------------------------
    # HUD
    # -----------------------------
    speed = get_speed()

    txt1 = font.render(f"Speed: {speed:.2f} m/s", True, (255,255,255))
    txt2 = font.render(f"Reward: {reward:.2f}", True, (255,255,255))
    txt3 = font.render("W/S throttle-brake | A/D steer | R reset | ESC quit", True, (255,255,255))

    screen.blit(txt1, (20, 20))
    screen.blit(txt2, (20, 55))
    screen.blit(txt3, (20, 90))

    pygame.display.flip()
    clock.tick(FPS)

    step += 1

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
camera.stop()
camera.destroy()
env.close()
pygame.quit()

print("Manual drive closed.")
