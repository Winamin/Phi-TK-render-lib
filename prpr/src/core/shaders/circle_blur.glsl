#version 450 core
// Adapted from https://godotshaders.com/shader/artsy-circle-blur-type-thingy/

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;
uniform vec2 screenSize;
uniform sampler2D screenTexture;

uniform float size; // %10.0%

void main() {
  vec4 c = texture(screenTexture, uv);
  float length = dot(c, c);
  vec2 pixel_size = 1.0 / screenSize;
  for (float x = -size; x < size; x++) {
    for (float y = -size; y < size; ++y) {
      if (x * x + y * y > size * size) continue;
      vec4 new_c = texture(screenTexture, uv + pixel_size * vec2(x, y));
      float new_length = dot(new_c, new_c);
      if (new_length > length) {
        length = new_length;
        c = new_c;
      }
    }
  }
  fragColor = c;
}
