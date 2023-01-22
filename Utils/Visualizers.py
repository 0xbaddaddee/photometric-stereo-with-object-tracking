import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import mayavi.mlab as mlab

from Utils import ImageUtils
from DataTypes import Mesh


def display_image(src, __WINDOW_NAME='Displaying image'):
    cv.namedWindow(__WINDOW_NAME, cv.WINDOW_KEEPRATIO)
    H, W = src.shape[:2]
    cv.resizeWindow(__WINDOW_NAME, W // 2, H // 2)
    cv.imshow(__WINDOW_NAME, src)
    cv.waitKey(0)
    cv.destroyWindow(__WINDOW_NAME)


def display_intensities_histogram(src):
    __TITLE = 'INTENSITIES HISTOGRAM'
    vals = src.mean(axis=2).flatten() if len(src.shape) > 2 else src.flatten()
    counts, bins = np.histogram(vals, range(257))
    plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
    plt.xlim([-0.5, 255.5])
    plt.show()


def display_color_channels(src):
    r, g, b = ImageUtils.extract_color_channels(src)
    _image = np.concatenate((r, g, b), axis=1)
    display_image(_image)


def display_mesh_normals_2d(mesh: Mesh, component='z'):
    x = np.linspace(-1, 1, mesh.size_x)
    y = np.linspace(-1, 1, mesh.size_y)

    coordinates = {'x': 0, 'y': 1, 'z': 2}

    vals_to_display = mesh.normals[:, coordinates[component]].reshape(mesh.size_y, mesh.size_x)
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.surf(x, y, vals_to_display, warp_scale='auto', colormap='gray')
    mlab.show()


def display_mesh_normals_1d(mesh: Mesh, component='z'):
    coordinates = {'x': 0, 'y': 1, 'z': 2}
    vals_to_display = mesh.normals[:, coordinates[component]]
    plt.plot(vals_to_display, label=component)
    plt.show()
