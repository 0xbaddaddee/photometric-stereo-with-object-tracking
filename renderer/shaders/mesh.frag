#version 430

precision highp float;  // see vertex shader

in vec4 world_point;

out vec3 fragment_color;

void main() {
    fragment_color = vec3(world_point);
}