import bpy
from bpy import data as D
from bpy import context as C
from mathutils import *
from math import *
import numpy as np
import os

red_light = np.array(bpy.data.objects['RED'].location)
green_light = np.array(bpy.data.objects['GREEN'].location)
blue_light = np.array(bpy.data.objects['BLUE'].location)

lights_file_name = 'light_vectors.txt'
file_path = os.path.join(os.path.dirname(bpy.data.filepath), lights_file_name)
lights = np.vstack([red_light, green_light, blue_light]) * 1000.
np.savetxt(file_path, lights)
