#ifdef DEF_VERTEX_ATTRIBUTES
layout(location = 0) in vec3 in_attr_pos;
layout(location = 1) in vec2 in_attr_uv;
layout(location = 2) in vec4 in_attr_color;
layout(location = 3) in vec4 in_attr_inst_pos;
layout(location = 4) in vec4 in_attr_inst_uv;
layout(location = 5) in vec4 in_attr_inst_data;
layout(location = 6) in vec4 in_attr_inst_color;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 _mvp;
    float _local_coords;
    vec3 _emitter_position;
};

mat2 rotate2d(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c);
}

vec4 particle_transform_vertex() {
    mat2 rot = rotate2d(in_attr_inst_pos.z);

    vec3 rotated_pos = vec3(rot * in_attr_pos.xy, in_attr_pos.z) * in_attr_inst_pos.w;

    vec3 offset = in_attr_inst_pos.xyz;
    offset += _emitter_position * step(0.5, _local_coords);

    return _mvp * vec4(rotated_pos + offset, 1.0);
}

vec2 particle_transform_uv() {
    return in_attr_uv * in_attr_inst_uv.zw + in_attr_inst_uv.xy;
}

float rand(vec2 co) {
    // 保留必要的mod操作保证随机质量
    float dt = dot(co, vec2(12.9898, 78.233));
    float sn = mod(dt, 3.1415926535); // 使用π值避免精度问题

    return fract(sin(sn) * 43758.5453);
}

#define particle_ix(data) (data.x)
#define particle_lifetime(data) (data.y)

#endif