#version 450 core

uniform sampler2D screenTexture;  // 替换u_texture和CC_Texture0
uniform float time;               // 替换u_time
uniform float frequency; // %10.0%
uniform float amplitude; // %0.02%
uniform vec2 screenSize;          // 屏幕大小

layout(location = 0) in vec2 uv;                  // 替换v_texCoord
layout(location = 0) out vec4 fragColor;

void main()
{
    vec2 offset = vec2(sin(uv.y * frequency + time) * amplitude, cos(uv.x * frequency + time) * amplitude);

    vec2 distortedUV = uv + offset;

    vec4 color = texture(screenTexture, distortedUV);

    fragColor = color;
    fragColor.a = 1.0;
}
