#version 450 core
// Adapted from https://www.shadertoy.com/view/lsKSWR

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;
uniform vec2 screenSize;
uniform sampler2D screenTexture;

uniform vec4 color; // %0.0, 0.0, 0.0, 1.0%
uniform float extend; // %0.25% 0..1
uniform float radius; // %15.0%

void main() {
  vec2 new_uv = uv * (1.0 - uv.yx);
  float vig = new_uv.x * new_uv.y * radius;
  vig = pow(vig, extend);
  fragColor = mix(color, texture(screenTexture, uv), vig);
}
