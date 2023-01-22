from typing import Tuple

import cv2 as cv
import numpy as np


def extract_color_channels(image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]


def load_rgb_image(path: str) -> np.ndarray:
    # https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html#gga61d9b0126a3e57d9277ac48327799c80af660544735200cbe942eea09232eb822
    bgr_image = cv.imread(path, cv.IMREAD_COLOR)
    return convert_bgr2rgb(bgr_image)


def load_gray_image(path: str) -> np.ndarray:
    return cv.imread(path, cv.IMREAD_GRAYSCALE)


def normalize_image(image):
    return image / 255.


def convert_srgb2linear(image, gamma=2.2):
    return normalize_image(image) ** gamma


def convert_linear2srgb(image, gamma=2.2):
    srgb_image = 255 * (image ** (1. / gamma))
    # min(max(0, VAL), 255)
    clamped_img = np.where(srgb_image > 0, srgb_image, 0)
    clamped_img = np.where(clamped_img < 255, clamped_img, 255)
    return clamped_img.astype('uint8')


def convert_rgb2gray(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def convert_rgb2hsv(image):
    return cv.cvtColor(image, cv.COLOR_RGB2HSV)


def convert_rgb2bgr(image):
    return cv.cvtColor(image, cv.COLOR_RGB2BGR)


def convert_bgr2rgb(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)
