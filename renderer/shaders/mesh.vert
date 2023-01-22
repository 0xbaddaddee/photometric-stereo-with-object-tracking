#version 430

precision highp float;  // this affects all collections of basic type float, including scalars, vectors and matrices.

in vec3 in_position;

uniform mat4 projection_matrix;  // projection matrix built from OpenCV intrinsics
uniform mat4 world2cam_matrix;

out vec4 world_point;


void main() {
    world_point = vec4(in_position, 1.0);
    vec4 point_cam_space = world2cam_matrix * world_point;
    vec4 point_opengl_space = vec4(point_cam_space.xy, -point_cam_space.z, point_cam_space.w);
    gl_Position = projection_matrix * point_opengl_space;
}