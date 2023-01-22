from dataclasses import dataclass, field
from typing import Tuple, List

import trimesh
import numpy as np


@dataclass
class Intrinsics:
    intrinsics_matrix: np.ndarray

    @property
    def matrix(self):
        return self.intrinsics_matrix

    @property
    def fx(self):
        return self.matrix[0, 0]

    @property
    def fy(self):
        return self.matrix[1, 1]

    @property
    def cx(self):
        return self.matrix[0, 2]

    @property
    def cy(self):
        return self.matrix[1, 2]


@dataclass
class Mesh:
    def __init__(self, positions=None, indices=None, normals=None):
        self.positions = positions
        self.indices = indices
        self.normals = normals
        self.size_x = 0
        self.size_y = 0
        self.ssbo = None  # shader storage buffer object with the corresponding pixels to the interpolated positions
        self.interpolated_grid = None

    def interpolate_positions(self):
        from renderer.MeshInterpolator import Viewer
        Viewer.set_all_meshes_to_render([self])
        Viewer.run()
        self.set_interpolated_positions(Viewer.get_rendered_data()[0])
        return self

    def set_interpolated_positions(self, interpolated_data):
        fixed_grid, ssbo = self._fix_interpolated_grid_(np.copy(interpolated_data))

        # some sanity checks that we are indeed rectangular
        assert np.all(np.isclose(fixed_grid[0, :, 1], fixed_grid[0, 0, 1], rtol=1e-2))  # same y-values in first row
        assert np.all(np.isclose(fixed_grid[:, 0, 0], fixed_grid[0, 0, 0], rtol=1e-2))  # same x-values in first col
        assert np.all(np.isclose(fixed_grid[-1, :, 1], fixed_grid[-1, 0, 1], rtol=1e-2))  # same y-values in last row
        assert np.all(np.isclose(fixed_grid[:, -1, 0], fixed_grid[-1, -1, 0], rtol=1e-2))  # same x-values in last col

        self.positions = fixed_grid.reshape((np.multiply(*fixed_grid.shape[:2]), 3))
        self.indices = self._generate_indices_(*fixed_grid.shape[:2])
        self.normals = np.ones_like(self.positions) * [0.0, 0.0, 1.0]
        self.size_y = fixed_grid.shape[0]
        self.size_x = fixed_grid.shape[1]
        self.ssbo = ssbo
        return self

    def _fix_interpolated_grid_(self, interpolated_pos):
        from renderer.MeshInterpolator import Viewer
        # In the rectangular grid, min and max of interpolated_mask represent the indices as pixels where
        # the first and last valid 3d point occurs. Due to interpolation, the rectangular grid might be broken,
        # so we need to fill the edges

        interpolated_pos_mask = np.where(interpolated_pos != Viewer.INVALID_POINTS)
        if len(interpolated_pos_mask) == 0:
            raise ValueError('No interpolated positions! Check mesh positions and fragment shader.')

        min_v, min_u, _ = np.min(interpolated_pos_mask, axis=1)
        max_v, max_u, _ = np.max(interpolated_pos_mask, axis=1)

        valid_positions = interpolated_pos[interpolated_pos != Viewer.INVALID_POINTS].reshape(-1, 3)
        max_x_val, max_y_val, _ = np.max(valid_positions, axis=0)
        min_x_val, min_y_val, _ = np.min(valid_positions, axis=0)
        # Fix broken grid
        interpolated_x_values = np.linspace(min_x_val, max_x_val, max_u - min_u + 1)
        interpolated_y_values = np.linspace(min_y_val, max_y_val, max_v - min_v + 1)

        top_row_left_to_right = np.array(np.meshgrid(interpolated_x_values, [max_y_val], [0.0])).reshape(3, -1).T
        bottom_row_left_to_right = np.array(np.meshgrid(interpolated_x_values, [min_y_val], [0.0])).reshape(3, -1).T

        left_column_top_to_bottom = np.array(np.meshgrid(
            [min_x_val],
            interpolated_y_values[::-1],
            [0.0])).reshape(3, -1).T
        right_column_top_to_bottom = np.array(np.meshgrid(
            [max_x_val],
            interpolated_y_values[::-1],
            [0.0])).reshape(3, -1).T

        fixed_grid = interpolated_pos[min_v:max_v + 1, min_u:max_u + 1]
        fixed_grid[0, :] = top_row_left_to_right
        fixed_grid[-1, :] = bottom_row_left_to_right
        fixed_grid[:, 0] = left_column_top_to_bottom
        fixed_grid[:, -1] = right_column_top_to_bottom
        ssbo = np.array(np.meshgrid(np.arange(min_u, max_u + 1), np.arange(min_v, max_v + 1))).reshape(2, -1).T
        self.interpolated_grid = interpolated_pos
        return fixed_grid, ssbo

    def generate_simple_rectangular_mesh(self, center, size_x, size_y):
        """ Generate a rectangular mesh consisting of 2 triangles only."""
        # y
        # ^
        # |
        # o --->x

        # v0---v3
        # |    |
        # v1---v2
        start_x = center[0] - size_x
        start_y = center[1] + size_y
        end_x = center[0] + size_x
        end_y = center[1] - size_y
        v0 = [start_x, start_y, 0.0]
        v1 = [start_x, end_y, 0.0]
        v2 = [end_x, end_y, 0.0]
        v3 = [end_x, start_y, 0.0]

        self.positions = np.array([v0, v1, v2, v3], dtype='float32')
        self.indices = np.array([[0, 1, 2], [0, 2, 3]], dtype='i4')
        self.normals = np.ones_like(self.positions) * [0, 0, 1]
        return self

    def generate_rectangular_grid(self, start, end, subdivide_every_x_mm):
        divisions = np.abs(np.array(start).flatten() - np.array(end).flatten()) * 1. / subdivide_every_x_mm
        x_start, y_start = start[0], start[1]
        x_end, y_end = end[0], end[1]
        div_x, div_y = int(divisions[0]), int(divisions[1])
        x_values = np.linspace(x_start, x_end, div_x)
        y_values = np.linspace(y_start, y_end, div_y)
        self.positions = np.array(np.meshgrid(x_values, y_values, [0])).reshape(3, -1).T
        self.normals = np.ones_like(self.positions) * [0, 0, 1]
        self.size_x = len(x_values)
        self.size_y = len(y_values)
        self.indices = self._generate_indices_(self.size_x, self.size_y)
        return self

    @staticmethod
    def _generate_indices_(x_size, y_size):
        triangles = np.zeros(((x_size - 1) * (y_size - 1) * 2, 3), dtype='i4')  # 32-bit signed integer
        t_idx = v_idx = 0
        for y in range(y_size - 1):
            for x in range(x_size - 1):
                triangles[t_idx] = [v_idx, v_idx + x_size, v_idx + x_size + 1]
                triangles[t_idx + 1] = [v_idx, v_idx + x_size + 1, v_idx + 1]

                t_idx = t_idx + 2  # 6 faces for 2 quads per loop step
                v_idx = v_idx + 1
            v_idx = v_idx + 1
        return triangles

    def project_vertices_to_hemisphere(self, cx, cy, hemisphere_radius):
        positions_xy_coordinates = self.positions[:, :2]
        hemisphere_center = np.array([cx, cy])
        # (x - cx)² + (y - cy)² + (z - cz)² = r² => z² = r² - [(x-cx)² + (y-cy)²]
        vertex_distances_to_circle_center = np.sum((positions_xy_coordinates - hemisphere_center)**2, axis=1)
        vertices_inside_circle = vertex_distances_to_circle_center < hemisphere_radius ** 2
        vertex_zs_projected = hemisphere_radius ** 2 - vertex_distances_to_circle_center
        vertex_old_zs = self.positions[:, 2]
        vertices_new_z_vals = np.where(vertices_inside_circle, vertex_zs_projected, vertex_old_zs)
        self.positions[:, 2] = vertices_new_z_vals ** 0.5
        for idx, vertex in enumerate(self.positions):
            if (vertex[0] - cx)**2 + (vertex[1] - cy)**2 < hemisphere_radius ** 2:
                hemisphere_normal = np.array([vertex[0] - cx, vertex[1] - cy, vertex[2] - 0.0])
                hemisphere_mag = np.linalg.norm(hemisphere_normal)
                hemisphere_unit_normal = hemisphere_normal / hemisphere_mag
                self.normals[idx] = hemisphere_unit_normal
        return self

    def read_from_ply_trimesh(self, plyfile_name: str):
        mesh = trimesh.load_mesh(plyfile_name)
        self.positions = np.array(mesh.vertices)
        self.indices = np.array(mesh.faces)
        if 'nx' in mesh.vertex_attributes:
            self.normals = np.array(
                [mesh.vertex_attributes['nx'], mesh.vertex_attributes['ny'], mesh.vertex_attributes['nz']]).T
        else:
            # Set the normals to zero if the normal fields are not present
            self.normals = np.ones_like(self.positions) * [0, 0, 1]
        return self

    def write_to_ply_trimesh(self, plyfile_name: str):
        mesh = trimesh.Trimesh(vertices=self.positions, faces=self.indices)
        normals = self.normals.astype('float32')
        mesh.vertex_attributes.update({'nx': normals[:, 0], 'ny': normals[:, 1], 'nz': normals[:, 2]})
        mesh.export(plyfile_name)

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    @property
    def normals(self) -> np.ndarray:
        return self._normals

    @positions.setter
    def positions(self, new_positions):
        self._positions = new_positions

    @indices.setter
    def indices(self, new_indices):
        self._indices = new_indices

    @normals.setter
    def normals(self, new_normals):
        self._normals = new_normals


@dataclass
class TrackerObjects:
    items: List[Tuple] = field(default_factory=lambda: [])

    def get_first_mesh(self) -> Mesh:
        """ Return the first underlying mesh in the struct.
        :return: The first underlying mesh in the Tracker structure.
        """

        try:
            return self.items[0][-1]
        except IndexError:
            print('Index not found. Returning empty mesh.')
            return Mesh()

    def get_mesh_at_index(self, index: int) -> Mesh:
        """ Return the underlying mesh in the struct at the specified index.
        :param index: Index at which to retrieve the underlying mesh.
        :return: The underlying mesh in the Tracker structure at the specified index.
        """
        try:
            return self.items[index][-1]
        except IndexError:
            print('Index not found. Returning empty mesh.')
            return Mesh()

    def get_all_meshes(self) -> List[Mesh]:
        """ Return all underlying meshes in the struct.
        :return: List of underlying meshes in the Tracker structure.
        """
        return list(np.array(self.items)[:, -1])
