#version 450 core

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;
uniform sampler2D screenTexture;

uniform vec4 fogColor; // %0.8, 0.8, 0.8, 1.0%
uniform float fogStart; // %0.8%
uniform float fogEnd; // %-0.2%
uniform float time;

void main() {
    // 读取纹理颜色
    vec4 color = texture(screenTexture, uv);

    // 使用Y坐标计算雾的基础浓度
    float fogFactor = clamp((fogEnd - uv.y) / (fogEnd - fogStart), 0.0, 1.0);

    // 将雾的颜色与原始颜色混合
    vec4 finalColor = mix(fogColor, color, fogFactor);

    fragColor = vec4(finalColor.rgb, 1.0); // 最终颜色，使用原始透明度
}
