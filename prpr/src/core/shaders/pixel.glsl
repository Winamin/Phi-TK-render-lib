#version 450 core
// Adapted from https://godotshaders.com/shader/pixelate-2/

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;
uniform vec2 screenSize;
uniform sampler2D screenTexture;

uniform float size; // %10.0%

void main() {
  vec2 factor = screenSize / size;
  float x = floor(uv.x * factor.x + 0.5) / factor.x;
  float y = floor(uv.y * factor.y + 0.5) / factor.y;
  fragColor = texture(screenTexture, vec2(x, y));
}
