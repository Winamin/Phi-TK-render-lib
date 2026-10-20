#version 450 core

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;
uniform sampler2D screenTexture;
uniform float time;

uniform float power; // %0.2%
uniform float density; // %400.0%

float rand(vec2 co) {
    return fract(sin(dot(co.xy * time * 0.05, vec2(12.9898, 78.233))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 ip = floor(p);
    vec2 u = fract(p);
    u = u*u*(3.0-2.0*u);
    
    float res = mix(
        mix(rand(ip), rand(ip + vec2(1.0, 0.0)), u.x),
        mix(rand(ip + vec2(0.0, 1.0)), rand(ip + vec2(1.0, 1.0)), u.x),
        u.y
    );
    return res;
}

void main() {
    vec4 color = texture(screenTexture, uv);
    
    // 噪点随时间变化，加入uv扰动
    float noisyFactor = noise(uv * density + time * 0.1) * 2.0 - 1.0;
    
    // 添加噪点
    vec4 noisyColor = color + vec4(vec3(noisyFactor) * power, 0.0);
    
    fragColor = noisyColor;
    fragColor.a = 1.0;
}