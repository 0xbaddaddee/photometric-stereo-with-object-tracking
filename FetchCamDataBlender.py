import bpy
import numpy as np
import sys
import os
from bpy import data as D
from bpy import context as C
from mathutils import *
from math import *

# ~ PYTHON INTERACTIVE CONSOLE 3.10.2 (main, Jan 27 2022, 08:34:43) [MSC v.1928 64 bit (AMD64)]
# ~
# ~ Builtin Modules:       bpy, bpy.data, bpy.ops, bpy.props, bpy.types, bpy.context, bpy.utils, bgl, blf, mathutils
# ~ Convenience Imports:   from mathutils import *; from math import *
# ~ Convenience Variables: C = bpy.context, D = bpy.data
# ~
cam = bpy.data.objects['Camera']
camd = cam.data
scene = bpy.context.scene

# Reference relative to blend-file
dir_name = os.path.join(os.path.dirname(bpy.data.filepath), 'blender_camera_info')
intrinsics_file_name = 'intrinsics'
extrinsics_file_name = 'extrinsics'


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_3x3_intrinsic_camera_matrix_K(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
         (0, s_v, v_0),
         (0, 0, 1)))
    K = np.array(K)
    np.savetxt(f'{dir_name}/{intrinsics_file_name}.txt', K)


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_4x4_extrinsic_matrix_RT(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    location = Matrix.Scale(1000.0, 3) @ location  # Our application works with mm instead of meters!
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    RT = np.array(RT)
    EX = np.append(RT, [0, 0, 0, 1]).reshape(4, 4)
    np.savetxt(f'{dir_name}/{extrinsics_file_name}.txt', EX)


def fetch_intrinsics():
    assert scene.render.resolution_percentage == 100
    assert cam_data.sensor_fit != 'VERTICAL'
    f_in_mm = cam_data.lens
    sensor_width_in_mm = cam_data.sensor_width
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    fx = f_in_mm / sensor_width_in_mm * w
    fy = fx * pixel_aspect_ratio
    cx = w * (0.5 - cam_data.shift_x)
    cy = h * 0.5 + w * cam_data.shift_y
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    np.savetxt(f'{dir_name}/{intrinsics_file_name}.txt', K)


def fetch_extrinsics():
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = -1 * R_world2bcam @ location
    reflect_around_x_axis = np.eye(3, 3)
    reflect_around_x_axis[0, 0] = -1.  # from right to left coordinate system
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = reflect_around_x_axis @ np.array(R_world2bcam)
    extrinsics[:3, 3] = np.array(T_world2bcam)
    np.savetxt(f'{dir_name}/{extrinsics_file_name}.txt', extrinsics)


program_args = sys.argv[sys.argv.index('--') + 1:]  # get all args after '--' for this script
# blender ignores empty args after "--"
if intrinsics_file_name in sys.argv:
    get_3x3_intrinsic_camera_matrix_K(camd)
if extrinsics_file_name in sys.argv:
    get_4x4_extrinsic_matrix_RT(cam)
