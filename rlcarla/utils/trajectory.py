import numpy as np
import cv2
import carla


def get_camera_extrinsic(camera_transform):
    pitch = np.radians(camera_transform.rotation.pitch)
    yaw   = np.radians(camera_transform.rotation.yaw)
    roll  = np.radians(camera_transform.rotation.roll)

    Rz = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [0,             0,           1]
    ])
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0,              1, 0            ],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0,            0           ],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    R_world = Rz @ Ry @ Rx

    t = np.array([
        camera_transform.location.x,
        camera_transform.location.y,
        camera_transform.location.z
    ])

    axis_swap = np.array([
        [0,  1,  0],
        [0,  0, -1],
        [1,  0,  0]
    ], dtype=np.float64)

    R_cam             = axis_swap @ R_world.T
    t_cam             = -R_cam @ t
    extrinsic         = np.eye(4)
    extrinsic[:3, :3] = R_cam
    extrinsic[:3,  3] = t_cam

    return extrinsic


def world_to_pixel(world_point, intrinsic, extrinsic, img_w, img_h):
    p_world = np.array([
        world_point.x,
        world_point.y,
        world_point.z,
        1.0
    ])

    p_cam = extrinsic @ p_world

    if p_cam[2] <= 0.1:
        return None

    p_img = intrinsic @ p_cam[:3]
    u     = int(p_img[0] / p_img[2])
    v     = int(p_img[1] / p_img[2])

    if 0 <= u < img_w and 0 <= v < img_h:
        return (u, v)
    return None


def draw_trajectory(
    image,
    waypoints,
    intrinsic,
    camera_transform,
    img_w,
    img_h,
    color      = (0, 220, 0),
    thickness  = 3,
    dot_radius = 5,
    max_points = 20,
):
    extrinsic = get_camera_extrinsic(camera_transform)

    pixels = []
    for wp in waypoints[:max_points]:
        loc       = wp.transform.location
        loc_raised = carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.3)
        px        = world_to_pixel(
            loc_raised, intrinsic, extrinsic, img_w, img_h
        )
        if px is not None:
            pixels.append(px)

    if len(pixels) < 2:
        return image

    pts = np.array(pixels, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        image, [pts], isClosed=False,
        color=color, thickness=thickness,
        lineType=cv2.LINE_AA
    )

    for px in pixels:
        cv2.circle(image, px, dot_radius, color, -1, lineType=cv2.LINE_AA)

    if pixels:
        cv2.circle(image, pixels[0], dot_radius + 2, (255, 255, 255), -1)

    return image


def get_future_waypoints(vehicle, carla_map, n=20, spacing=3.0):
    loc = vehicle.get_location()
    tf  = vehicle.get_transform()

    forward   = tf.get_forward_vector()
    ahead_loc = carla.Location(
        x = loc.x + forward.x * 4.0,
        y = loc.y + forward.y * 4.0,
        z = loc.z
    )

    wp = carla_map.get_waypoint(ahead_loc, project_to_road=True)

    if wp is None:
        wp = carla_map.get_waypoint(loc, project_to_road=True)

    if wp is None:
        return []

    waypoints = []
    current   = wp

    for _ in range(n):
        nexts = current.next(spacing)
        if not nexts:
            break
        current = nexts[0]
        waypoints.append(current)

    return waypoints
