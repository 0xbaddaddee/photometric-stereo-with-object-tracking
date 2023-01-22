from array import array
import moderngl
import numpy as np

from Utils import CameraUtils
from renderer.Base import Renderer


class Viewer(Renderer):
    INVALID_POINTS = np.array([-999_999., -999_999., -999_999.])
    FRAMEBUFFER_COMPONENTS = 3
    _meshes_to_render_ = []
    _fbo_results_ = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.program_mesh = self.load_program(vertex_shader='mesh.vert', fragment_shader='mesh.frag')
        self.program_quad = self.load_program(vertex_shader='quad.vert', fragment_shader='quad.frag')

        self.quad = self.create_render_quad(self.program_quad)
        self.mesh = None

        self.fbo = self.create_framebuffer()

        self.set_uniform(self.program_mesh, 'projection_matrix', CameraUtils.get_intrinsics_as_projection_matrix())
        self.set_uniform(self.program_mesh, 'world2cam_matrix', CameraUtils.WORLD2CAM_MATRIX)
        self.set_uniform(self.program_quad, 'invalid_position', Viewer.INVALID_POINTS)

    def render(self, time: float, frame_time: float):
        self.fbo.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        print(f'Rendering {len(Viewer._meshes_to_render_)} meshes.')
        for mesh in Viewer._meshes_to_render_:
            self.mesh = self.create_render_mesh(self.program_mesh, mesh)
            self.fbo.clear()
            self.quad.render()
            self.mesh.render()
            raw_data = self.fbo.read(components=self.FRAMEBUFFER_COMPONENTS, dtype='f4')
            render_result = np.frombuffer(raw_data, dtype='float32').reshape((*self.fbo.size[1::-1],
                                                                              self.FRAMEBUFFER_COMPONENTS))
            Viewer._fbo_results_.append(render_result)
        self.wnd.close()

    def create_render_mesh(self, shader_program, _mesh):
        vbo = self.ctx.buffer(
            array(
                'f',
                _mesh.positions.flatten()
            )
        )
        ibo = self.ctx.buffer(_mesh.indices)
        return self.ctx.vertex_array(shader_program, [(vbo, '3f', 'in_position')], index_buffer=ibo)

    def create_render_quad(self, shader_program):
        vertex_buffer_object = self.ctx.buffer(
            array(
                'f',
                [
                    # Triangles creating a full-screen quad
                    # x, y
                    -1, 1,  # upper left
                    -1, -1,  # lower left
                    1, 1,  # upper right
                    1, -1,  # lower right
                ]
            )
        )
        ibo = self.ctx.buffer(np.array([0, 1, 2, 1, 2, 3], dtype='i4'))
        return self.ctx.vertex_array(shader_program, [(vertex_buffer_object, '2f', 'in_position')], index_buffer=ibo)

    def create_framebuffer(self):
        return self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(self.window_size, components=self.FRAMEBUFFER_COMPONENTS,
                                                     dtype='f4')]
        )

    @staticmethod
    def set_uniform(shader_program: moderngl.Program, uniform_name, uniform_val):
        if isinstance(uniform_val, np.ndarray):
            # Actually copy the array and reorder it as c-contiguous. numpy.T makes it Fortran contiguous!
            # Shader expects it as c-contiguous
            uniform_val = np.copy(uniform_val.transpose(), order='C')
            assert uniform_val.flags['C_CONTIGUOUS']
            uniform_val = uniform_val.astype('float32')
        try:
            shader_program[uniform_name].write(uniform_val)
        except KeyError:
            print(f'uniform {uniform_name} - not found / used in the shader!')

    @staticmethod
    def get_rendered_data():
        return Viewer._fbo_results_

    @staticmethod
    def set_all_meshes_to_render(meshes_list):
        Viewer._meshes_to_render_ = meshes_list
