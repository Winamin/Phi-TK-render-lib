#version 450 core
// Adapted from https://www.shadertoy.com/view/lsdXDH

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;
uniform sampler2D screenTexture;

uniform float factor; // %1.0% 0..1

void main() {
  vec4 color = texture(screenTexture, uv);
  vec3 lum = vec3(0.299, 0.587, 0.114);
  vec3 gray = vec3(dot(lum, color.xyz));
  fragColor = vec4(mix(color.xyz, gray, factor), color.a);
}
