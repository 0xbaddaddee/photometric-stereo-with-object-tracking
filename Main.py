import os
import pathlib
import subprocess
from typing import List
import numpy as np

import PSfS
from Tracker import Tracker
from Utils import CameraUtils, ImageUtils, FileUtils, GeneralUtils
from DataTypes import Mesh
import itertools


def perform_position_correction(input_mesh: Mesh, input_frame: np.ndarray, input_dir: str, input_tol: int = 1e-2):
    # https://w3.impa.br/~diego/software/NehEtAl05/reference.html
    print(f'---PERFORM POSITION CORRECTION---')
    print(f'\t Lambda=0. Tolerance until lambda increase: {input_tol}')
    print(f'\t Output directory: {input_dir}')
    if not FileUtils.exists(input_dir):
        print(f'\t Output directory does not exist. Creating directory \'{input_dir}\'')
        FileUtils.create_dir(input_dir)

    os.chdir(input_dir)
    try:
        plyfile_name = 'original_input.ply'
        input_mesh.normals = PSfS.estimate_mesh_normals(input_mesh, input_frame)
        input_mesh.write_to_ply_trimesh(plyfile_name)

        program = 'mesh_opt.exe'
        args: List = [plyfile_name, '-lambda', '0', '-blambda', '1']

        print('---RUNNING MESH_OPT---')
        while float(args[2]) < 1.:
            old_normals = input_mesh.normals
            plyfile_name = f'mesh_output.ply'
            subprocess.run([FileUtils.join_paths(pathlib.Path(__file__).parent, program)] + args + [plyfile_name])
            print(f'\t Loading refined mesh {plyfile_name}')
            input_mesh.read_from_ply_trimesh(plyfile_name)
            input_mesh.positions[:, 2][input_mesh.positions[:, 2] < 0] = 0.0
            new_normals = PSfS.estimate_mesh_normals(mesh, input_frame)
            rmse = np.sqrt(np.mean((new_normals - old_normals)**2))
            print(f'\t Root mean squared error between new and old normals: {rmse}')
            if rmse < input_tol:
                print('\t RMSE is below tolerance. Increasing lambda by 10%.')
                args[2] = str(float(args[2]) + 0.01)
            input_mesh.normals = new_normals
            input_mesh.write_to_ply_trimesh(plyfile_name)
            args[0] = plyfile_name
    except (RuntimeError, TypeError) as error_instance:
        print(error_instance.args)
    finally:
        # Make sure we switch back to the parent directory
        os.chdir(pathlib.Path(__file__).parent)


def create_height_maps(input_meshes: List[Mesh]) -> List[np.ndarray]:
    """
    Create a list of height maps for the input meshes.

    :param input_meshes: The input meshes for which to create the height maps.
    :return: List of height maps for each mesh.
    """
    ret_height_maps = []
    # Create for each mesh a look-up array containing the reconstructed z values
    for input_mesh in input_meshes:
        input_mesh_pixels = np.int0(CameraUtils.project_points_to_image_plane(input_mesh.positions))
        ret_height_map = np.zeros(frames[0].shape[:2], dtype='float32')
        normalized_depth = input_mesh.positions[:, 2]
        normalized_depth = normalized_depth - np.min(normalized_depth)
        normalized_depth = normalized_depth / np.max(normalized_depth)
        normalized_depth = normalized_depth * 2 - 1.0
        ret_height_map[input_mesh_pixels[:, 1], input_mesh_pixels[:, 0]] = normalized_depth
        ret_height_maps.append(ret_height_map)
    return ret_height_maps


def create_object_space_height_maps(input_meshes: List[Mesh], matrices: List[np.ndarray]) -> List[np.ndarray]:
    """
        Create a list of height maps for the input meshes from an object-space perspective with the first mesh
        as a reference mesh.

        :param input_meshes: List of input meshes to follow through scene.
        :param matrices: List of transformation matrices from frame 0 -> frame 1, frame 0 -> frame 2, and so on.
        :return: List of height maps from an object-space perspective for the first mesh as a reference frame.
    """
    _height_maps = create_height_maps(input_meshes)
    _object_space_heightmaps = [np.copy(_height_maps[0])]
    _reference_mesh = input_meshes[0]
    _reference_pixels = np.int0(CameraUtils.project_points_to_image_plane(_reference_mesh.positions))
    for index in range(len(matrices)):
        _mesh_element = Mesh(np.copy(_reference_mesh.positions), np.copy(_reference_mesh.indices),
                            np.copy(_reference_mesh.normals))
        _mesh_element.positions = CameraUtils.transform_points(_mesh_element.positions, matrices[index])
        _followed_pixels = np.int0(CameraUtils.project_points_to_image_plane(_mesh_element.positions))
        _followed_depth_values = _height_maps[i][_followed_pixels[:, 1], _followed_pixels[:, 0]]
        _object_space_heightmap = np.zeros_like(_height_maps[0])
        _object_space_heightmap[_reference_pixels[:, 1], _reference_pixels[:, 0]] = _followed_depth_values
        _object_space_heightmaps.append(_object_space_heightmap)
    return _object_space_heightmaps


def create_normal_maps(input_meshes: List[Mesh], input_frames: List[np.ndarray]) -> List[np.ndarray]:
    """
        Create a list of normal maps for the input meshes.

        :param input_meshes: The input meshes for which to create the normal maps.
        :param input_frames: List of frames in which to estimate the normals for the meshes.
        :return: List of normal maps for each mesh.
    """
    ret_normal_maps = []
    # Create for each mesh a look-up array containing the estimates normals
    for index in range(len(input_meshes)):
        input_mesh = input_meshes[index]
        input_pixels = np.int0(CameraUtils.project_points_to_image_plane(input_mesh.positions))
        ret_normal_map = np.zeros(input_frames[index].shape, dtype='float32')
        n = PSfS.estimate_mesh_normals(mesh, ImageUtils.convert_srgb2linear(input_frames[i]))
        ret_normal_map[input_pixels[:, 1], input_pixels[:, 0]] = n
        ret_normal_maps.append(ret_normal_map)
    return ret_normal_maps


if __name__ == '__main__':
    CameraUtils.fetch_blender_camera_info('simulated_scenery.blend')
    PSfS.set_lights('simulated_scenery.blend')
    frames_dir = 'frames'
    frames = GeneralUtils.parallel_load_frames(frames_dir)

    tracker = Tracker()
    tracker_struct = GeneralUtils.parallel_prepare_tracker_struct(frames)
    transformation_matrices = tracker.parallel_track_object_in_all_frame(tracker_struct)
    T = list(itertools.accumulate(transformation_matrices, lambda T0, T1: T0 @ T1))

    all_meshes = tracker_struct.get_all_meshes()
    for i in range(len(frames)):
        linear_frame = ImageUtils.convert_srgb2linear(frames[i])
        current_mesh = all_meshes[i]
        perform_position_correction(current_mesh, linear_frame, f'refined_position_frame_{i}')

    reconstructed_meshes = []
    for i in range(len(frames)):
        mesh = Mesh().read_from_ply_trimesh(FileUtils.join_paths(f'refined_position_frame_{i}', 'mesh_output.ply'))
        reconstructed_meshes.append(mesh)

    object_space_height_maps = create_object_space_height_maps(reconstructed_meshes, T)
    create_normal_maps(reconstructed_meshes, frames)

    print('Done')
