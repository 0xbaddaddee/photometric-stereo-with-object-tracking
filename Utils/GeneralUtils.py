import itertools
import os
from typing import List

import cv2 as cv
import numpy as np

from DataTypes import Mesh, TrackerObjects
from Utils import ImageUtils, CameraUtils, FileUtils
import concurrent.futures

MAX_THREADS = (os.cpu_count() >> 1) + (os.cpu_count() >> 2)


def load_frames(frames_dir: str) -> List[np.ndarray]:
    """ Load frames sequentially with a for loop.

    :param frames_dir: Directory containing the frames to load.
    :return: List of Numpy arrays representing the loaded images.
    """
    frames = []
    for file in FileUtils.get_dir_content(frames_dir):
        if not FileUtils.has_ext(file, 'jpg'):
            continue
        rgb_image = ImageUtils.load_rgb_image(FileUtils.join_paths(frames_dir, file))
        frames.append(rgb_image)
    return frames


def parallel_load_frames(frames_dir: str) -> List[np.ndarray]:
    """ Load frames in parallel.

        :param frames_dir: Directory containing the frames to load.
        :return: List of Numpy arrays representing the loaded images.
    """
    print(f'---RUNNING THREAD POOL EXECUTOR---: Loading frames from directory \'{frames_dir}\'')
    file_names = FileUtils.get_dir_content(frames_dir)
    jpg_files = list(filter(lambda file: FileUtils.has_ext(file, 'jpg'), file_names))
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        relative_path_file_names = list(executor.map(FileUtils.join_paths, itertools.repeat(frames_dir), jpg_files))
        frames = list(executor.map(ImageUtils.load_rgb_image, relative_path_file_names))
    executor.shutdown(wait=False)
    print(f'\t Total frames loaded: {len(frames)}')
    return frames


def parallel_prepare_tracker_struct(frames: List[np.ndarray]) -> TrackerObjects:
    """ Prepare a helper struct for the tracking algorithm. For each frame, it creates a mesh to interpolate in the
    fragment shader, so that the feature points that are found by the tracker will have 3D points correspondences.

    :param frames: List of frames to prepare for the tracking algorithm.
    :return: TrackerObjects struct for the tracking algorithm.
    """
    print('---RUNNING THREAD POOL EXECUTOR---: Creating objects for the tracker')

    from renderer.MeshInterpolator import Viewer
    tracker_helper = TrackerObjects()

    def generate_mesh_worker(object_center):
        return Mesh().generate_simple_rectangular_mesh(object_center, 150, 150)

    def update_mesh_position_worker(mesh, new_positions):
        mesh.set_interpolated_positions(new_positions)
        return mesh

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        print('\t Extracting masks..')
        masks = list(executor.map(extract_mask, frames))
        print('\t Calculating object centers with image moments..')
        obj_centers = list(executor.map(get_3d_centroid_of_object, masks))
        print('\t Creating simple meshes..')
        meshes = list(executor.map(generate_mesh_worker, obj_centers))
    executor.shutdown(wait=False)
    Viewer.set_all_meshes_to_render(meshes)
    Viewer.run()
    interpolated_positions = Viewer.get_rendered_data()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        print('\t Interpolating meshes...this might take a while.')
        interpolated_meshes = list(executor.map(update_mesh_position_worker, meshes, interpolated_positions))
        print('\t Done.')
    executor.shutdown(wait=False)
    tracker_helper.items = list(zip(frames, masks, interpolated_meshes))
    return tracker_helper


def extract_mask(rgb_image: np.ndarray) -> np.ndarray:
    """ Extract the mask of an object in the given RGB image. Uses Canny Edge Detection and morphological operations
    to fill the mask.

    :param rgb_image: The image from which to extract the mask of the object.
    :return: Array of the same size as the provided image, containing the mask.
    """
    gray_image = ImageUtils.convert_rgb2gray(rgb_image)
    edges = cv.Canny(gray_image, 40, 180, L2gradient=True)
    edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_RECT, (9, 9)), iterations=5)
    dilation_erosion_diff = cv.morphologyEx(edges, op=cv.MORPH_CLOSE,
                                            kernel=cv.getStructuringElement(cv.MORPH_RECT, (7, 7)), iterations=5)
    contours, hierarchy = cv.findContours(dilation_erosion_diff, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    mask = cv.drawContours(np.zeros_like(gray_image), [max(contours, key=cv.contourArea)], 0, 255, cv.FILLED)
    # dilate again for a bit larger mask
    mask = cv.morphologyEx(mask, op=cv.MORPH_DILATE, kernel=cv.getStructuringElement(cv.MORPH_CROSS, (30, 30)))
    return mask


def get_3d_centroid_of_object(mask: np.ndarray) -> np.ndarray:
    """ Calculate the center of the object by its mask using image moments. This method assumes only the mask of the
    object is seen in the scene. Otherwise, the calculated centroid is misplaced.

    :param mask: The mask used for the image moments.
    :return: Centroid of the mask as 3D point.
    """
    if CameraUtils.K_MATRIX_INV[2, 2] == 0 or CameraUtils.CAM2WORLD_MATRIX[3, 3] == 0:
        raise ValueError('Camera matrices were not initialized. Fetch matrices first!')
    M = cv.moments(mask)
    if M['m00'] == 0:
        raise ArithmeticError('Central moments cannot be calculated. Would result in division by 0. Check the mask!')
    cx, cy = M['m10'] // M['m00'], M['m01'] // M['m00']
    return CameraUtils.project_pixel_to_world([cx, cy])
