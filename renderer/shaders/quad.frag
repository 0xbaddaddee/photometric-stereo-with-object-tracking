#version 430

uniform vec3 invalid_position;

out vec3 fragment_color;

void main() {
    fragment_color = invalid_position;
}