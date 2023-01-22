from abc import ABCMeta
import moderngl_window as mglw
from pathlib import Path
from Utils import CameraUtils


class Renderer(mglw.WindowConfig, metaclass=ABCMeta):
    gl_version = 4, 3
    window_size = CameraUtils.get_camera_resolution()
    resource_dir = (Path(__file__).parent / 'shaders').resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
