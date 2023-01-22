import sys
import numpy as np
import subprocess

from DataTypes import Intrinsics
from Utils import FileUtils

__blender_cam_data_dir__ = 'blender_camera_info'
__intrinsics_file_name__ = 'intrinsics.txt'
__extrinsics_file_name__ = 'extrinsics.txt'

INTRINSICS = Intrinsics(np.zeros((3, 3)))
WORLD2CAM_MATRIX = np.zeros((4, 4))
CAM2WORLD_MATRIX = np.zeros((4, 4))
K_MATRIX_INV = np.zeros((3, 3))


def get_camera_resolution():
    return 4032, 1860


def get_intrinsics_as_projection_matrix():
    # https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
    fx, fy, cx, cy = INTRINSICS.fx, INTRINSICS.fy, INTRINSICS.cx, INTRINSICS.cy
    near = 0.001
    far = 100 * 1000.  # unit in mm
    w, h = get_camera_resolution()
    return np.array([
        [2.0 * fx / w, 0.0, (w - 2.0 * cx) / w, 0.0],
        [0.0, 2.0 * fy / h, (-h + 2.0 * cy) / h, 0.0],
        [0.0, 0.0, (-far - near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype='float32')


def fetch_blender_camera_info(blender_file, fetch_extrinsics=True, fetch_intrinsics=True):
    blender_prog_path = 'blender'
    global INTRINSICS, WORLD2CAM_MATRIX, K_MATRIX_INV, CAM2WORLD_MATRIX

    if not FileUtils.exists(__blender_cam_data_dir__):
        FileUtils.create_dir(__blender_cam_data_dir__)

    print('---')
    print('Fetching camera info from blender')

    blender_py_api = f'{blender_prog_path} {blender_file} --background --python FetchCamDataBlender.py -- '
    if fetch_intrinsics:
        blender_py_api = blender_py_api + 'intrinsics '
    if fetch_extrinsics:
        blender_py_api = blender_py_api + 'extrinsics'
    run_report = subprocess.run(blender_py_api.split(), capture_output=True, text=True)
    INTRINSICS = Intrinsics(np.loadtxt(FileUtils.join_paths(__blender_cam_data_dir__, __intrinsics_file_name__)))
    WORLD2CAM_MATRIX = np.loadtxt(FileUtils.join_paths(__blender_cam_data_dir__, __extrinsics_file_name__))
    K_MATRIX_INV = np.linalg.inv(INTRINSICS.matrix)
    CAM2WORLD_MATRIX = np.linalg.inv(WORLD2CAM_MATRIX)
    print(run_report.stdout)
    print('---')


def load_intrinsics(file_path=FileUtils.join_paths(__blender_cam_data_dir__, __intrinsics_file_name__)):
    global INTRINSICS, K_MATRIX_INV
    if FileUtils.exists(file_path):
        INTRINSICS = Intrinsics(np.loadtxt(file_path))
        K_MATRIX_INV = np.linalg.inv(INTRINSICS.matrix)
    else:
        print(f'Failed to load {file_path}! File does not exist!', file=sys.stderr)


def load_extrinsics(file_path=FileUtils.join_paths(__blender_cam_data_dir__, __extrinsics_file_name__)):
    global WORLD2CAM_MATRIX, CAM2WORLD_MATRIX
    if FileUtils.exists(file_path):
        WORLD2CAM_MATRIX = np.loadtxt(file_path)
        CAM2WORLD_MATRIX = np.linalg.inv(WORLD2CAM_MATRIX)
    else:
        print(f'Failed to load {file_path}! File does not exist!', file=sys.stderr)


def project_points_to_image_plane(points):
    points_cam_space = transform_points(points, WORLD2CAM_MATRIX)
    points_cam_plane = transform_points(points_cam_space, INTRINSICS.matrix).T
    homogeneous_points_cam_plane = points_cam_plane / (points_cam_plane[2, :] * np.ones_like(points_cam_plane))
    projection_helper = np.zeros((2, 3))
    projection_helper[0, 0] = projection_helper[1, 1] = 1.  # drops homogeneous coordinate from a 3x3 matrix
    pixels = projection_helper @ homogeneous_points_cam_plane
    return pixels.T


def transform_points(points, matrix):
    if len(matrix) == 4:
        projection_helper = np.zeros((3, 4))
        projection_helper[0, 0] = projection_helper[1, 1] = projection_helper[2, 2] = 1.  # drops homogeneous coordinate
        homogeneous_points = np.append(points, np.ones((len(points), 1)), axis=-1)
        transformed_points = projection_helper @ (matrix @ homogeneous_points.T)
    else:
        transformed_points = matrix @ points.T
    return transformed_points.T


def project_pixel_to_world(pixel):
    pixel = np.array(pixel).flatten()
    p_cam = K_MATRIX_INV @ [pixel[0], pixel[1], 1.]
    p_world = (CAM2WORLD_MATRIX @ [p_cam[0], p_cam[1], p_cam[2], 1.])[:3]
    cam_center_world = (CAM2WORLD_MATRIX @ [0., 0., 0., 1.])[:3]
    # x(t) = o + t*(b-o)
    vec_from_p_world_to_origin_world = (p_world - cam_center_world)
    t = - cam_center_world[2] / vec_from_p_world_to_origin_world[2]
    return cam_center_world + t * vec_from_p_world_to_origin_world
