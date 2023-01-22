import concurrent.futures
from typing import List

from Utils import ImageUtils, CameraUtils, FileUtils, GeneralUtils
from DataTypes import Mesh
import numpy as np
import subprocess
import concurrent.futures

_LIGHTS_ = np.empty(0)


def _check_lights_are_set_():
    if len(_LIGHTS_) == 0:
        print('Lights matrix was not set. Consider calling PSfS.set_lights_from_blender before performing any other ',
              'operation!')


def estimate_mesh_normals(mesh: Mesh, frame: np.ndarray) -> np.ndarray:
    """ Estimate for the mesh its normals in the provided frame.

    :param mesh: The mesh to estimate the normals for.
    :param frame: In which frame to read the intensity values.
    :return: An array of shape (n, 3) representing the normal for each point in the mesh.
    """
    _check_lights_are_set_()

    #  [ [[Rx - px, Ry - py, Rz - pz],
    #    [Gx - px, Gy - py, Gz - pz],
    #    [Bx - px, By - py, Bz - pz]],
    #
    #   [[Rx - qx, Ry - qy, Rz - qz],
    #    [Gx - qx, Gy - qy, Gz - qz],
    #    [Bx - qx, By - qy, Bz - qz]]]
    points = mesh.positions
    pixels = np.int0(CameraUtils.project_points_to_image_plane(points))
    r, g, b = ImageUtils.extract_color_channels(frame)
    to_light = _LIGHTS_ - points[:, None]
    norm_of_to_light = np.linalg.norm(to_light, ord=2, axis=2, keepdims=True)
    normalized_to_light = to_light / norm_of_to_light
    inverse_lights = np.linalg.inv(normalized_to_light)

    intensities = np.vstack([r[pixels[:, 1], pixels[:, 0]],
                             g[pixels[:, 1], pixels[:, 0]],
                             b[pixels[:, 1], pixels[:, 0]]])
    N = inverse_lights @ intensities.T.reshape(*intensities.shape[::-1], -1)
    RHO = np.linalg.norm(N, ord=2, axis=1, keepdims=True)
    #RHO[:] = 1
    return (N / RHO)[:, :, 0]


def estimate_meshes_normals_parallel(meshes: List[Mesh], frames: List[np.ndarray]) -> List[np.ndarray]:
    """ Estimate in parallel for each mesh in the list its normals.

    :param meshes: A list of meshes.
    :param frames: A list of frames corresponding to each mesh in the meshes list.
    :return: List of normals for each mesh. Each element of the list is an array of shape (n, 3), representing the
    normal for each point in the mesh.
    """
    print(f'---RUNNING THREAD POOL EXECUTOR---: Estimating normals for {len(meshes)} meshes.')
    with concurrent.futures.ThreadPoolExecutor(max_workers=GeneralUtils.MAX_THREADS) as executor:
        normals = list(executor.map(estimate_mesh_normals, meshes, frames))
    executor.shutdown(wait=False)
    print('---DONE---')
    return normals


def set_lights(blender_file: str = None) -> None:
    """ Set the lights matrix needed before performing any other operations with this file. Checks if a file named
    'light_vectors.txt' exists and loads its data. Alternatively, you can provide a blender_file from which to fetch
    the lights.

    :param blender_file: Optional. If provided, will fetch the lights matrix directly from blender. Otherwise,
    :param blender_file: it will look for 'light_vectors.txt'. If this file doesn't exist, you must pass a blender_file!
    """
    global _LIGHTS_
    lights_file_name = 'light_vectors.txt'
    # blender_prog_path = 'C:/Program Files/Blender Foundation/Blender 3.3/blender.exe'
    blender_prog_path = 'blender'
    print('---LOADING LIGHTS DATA---')
    if not FileUtils.exists(lights_file_name):
        print(f'\t "light_vectors.txt" does not exist. Loading from blender file: {blender_file}')
        if blender_file is None:
            print('\t You must provide a blender file in order to re-fetch the light vectors!')
            return
        prog_exec = f'{blender_prog_path} {blender_file} --background --python FetchLightVectorsBlender.py'
        run_report = subprocess.run(prog_exec.split(), capture_output=True, text=True)
        print(run_report.stdout)
    _LIGHTS_ = np.loadtxt(lights_file_name)
    print('---DONE---')


def estimate_normals(frame: np.ndarray) -> np.ndarray:
    """ Estimate normals in the provided frame without considering near-light photometric stereo.

    :param frame: The frame for which to estimate the normals.
    :return: Array of shape (height_of_frame * width_of_frame, 3) containing normals for each pixel.
    """
    _check_lights_are_set_()
    r, g, b = ImageUtils.extract_color_channels(frame)
    intensities = np.vstack([r.flatten(), g.flatten(), b.flatten()])
    norm_of_lights = np.linalg.norm(_LIGHTS_, ord=2, axis=1, keepdims=True)
    normalized_lights = _LIGHTS_ / norm_of_lights
    normals = np.linalg.inv(normalized_lights) @ intensities
    albedo_map = np.linalg.norm(normals, ord=2, axis=0, keepdims=True)
    normalized_normals = normals / albedo_map
    return normalized_normals.T.reshape((frame.shape[0] * frame.shape[1], 3))


def estimate_mesh_normals_for_loop(mesh: Mesh, frame: np.ndarray) -> np.ndarray:
    """ Estimate normals for mesh in the provided frame with a for-loop.

    :param mesh: The mesh to estimate the normals for.
    :param frame: Frame in which to look for intensity values.
    :return: An array of shape (n, 3) representing the normals for each point in the mesh.
    """
    _check_lights_are_set_()
    positions = mesh.positions
    pixels = CameraUtils.project_points_to_image_plane(positions)
    red_channel, green_channel, blue_channel = ImageUtils.extract_color_channels(frame)
    estimated_mesh_normals = []
    for index, pixel in enumerate(pixels):
        current_vertex = positions[index]
        u, v = int(pixel[0]), int(pixel[1])
        vertex_intensities = np.vstack([red_channel[v, u], green_channel[v, u], blue_channel[v, u]])
        vec_to_light = _LIGHTS_ - current_vertex
        norm_of_lights = np.linalg.norm(vec_to_light, ord=2, axis=1, keepdims=True)
        normalized_lights = vec_to_light / norm_of_lights
        normal = np.linalg.inv(normalized_lights) @ vertex_intensities

        albedo = np.linalg.norm(normal, ord=2, axis=0, keepdims=True)

        unit_normal = normal / albedo
        estimated_mesh_normals.append(unit_normal)
    estimated_mesh_normals = np.array(estimated_mesh_normals)
    estimated_mesh_normals = estimated_mesh_normals.reshape(estimated_mesh_normals.shape[:2])
    return estimated_mesh_normals
