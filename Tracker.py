from typing import Tuple

import cv2 as cv
import numpy as np

from DataTypes import TrackerObjects
from Utils import ImageUtils
import concurrent.futures
from Utils import GeneralUtils


class Tracker:
    def __init__(self):
        self.object_tracker = cv.ORB_create()  # ORB is faster than SIFT
        self.feature_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # match features with brute force

    def parallel_track_object_in_all_frame(self, tracker_helper: TrackerObjects):
        print(f'---RUNNING THREAD POOL EXECUTOR---: Tracking objects in all {len(tracker_helper.items)} frames.')
        # Constructs a list of tuples containing tuples
        # E.g., a list of 2
        #   -----------------------FST PAIR------------------   ----------------------SND PAIR--------------------
        # [ ( (frame0, mask0, mesh0), (frame1, mask1, mesh1) ), ( (frame1, mask1, mesh1), (frame2, mask2, mesh2) ) ]
        # in the above example: used for tracking from frame0 to frame1, and frame1 and frame2
        all_pairs = list(zip(tracker_helper.items, tracker_helper.items[1:]))
        # Create a thread pool with 75% amount of all CPU cores
        print(f'\t Creating thread pool with {GeneralUtils.MAX_THREADS} threads.')
        with concurrent.futures.ThreadPoolExecutor(max_workers=GeneralUtils.MAX_THREADS) as executor:
            print(f'\t Tracking objects using {self.object_tracker}..')
            result = list(executor.map(self.track_object_worker, all_pairs))
        executor.shutdown(wait=False)
        print('\t Done.')
        return result

    def track_object_worker(self, pairs: Tuple[Tuple, Tuple]):
        from_frame, from_mask, from_mesh = pairs[0][0], pairs[0][1], pairs[0][2]
        to_frame, to_mask, to_mesh = pairs[1][0], pairs[1][1], pairs[1][2]

        frame_0_gray = ImageUtils.convert_rgb2gray(from_frame)
        frame_1_gray = ImageUtils.convert_rgb2gray(to_frame)
        interpolated_src_points = from_mesh.interpolated_grid
        interpolated_dst_points = to_mesh.interpolated_grid
        kp0, des0 = self.object_tracker.detectAndCompute(frame_0_gray, from_mask)
        kp1, des1 = self.object_tracker.detectAndCompute(frame_1_gray, to_mask)
        matches = self.feature_matcher.match(des0, des1)

        # convert to integer pixels
        src_pixels = np.int0([kp0[match.queryIdx].pt for match in matches])
        dst_pixels = np.int0([kp1[match.trainIdx].pt for match in matches])

        src_points = np.array([interpolated_src_points[pixel[1], pixel[0]].tolist() for pixel in src_pixels])
        dst_points = np.array([interpolated_dst_points[pixel[1], pixel[0]].tolist() for pixel in dst_pixels])

        # high confidence -> slow convergence
        ret_val, M, inliers = cv.estimateAffine3D(src_points, dst_points, ransacThreshold=1.6, confidence=0.9999)
        assert M is not None
        R = M[:3, :3]
        T = M[:, 3]
        RT = np.eye(4)
        RT[:3, :3] = R
        RT[:3, 3] = T
        return RT
