#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_tex_living;
layout(rgba32f, set = 0, binding = 1) uniform writeonly image2D out_kernel;

layout(std430, set = 0, binding = 2) buffer Params {
    vec2 u_res;
    float u_R;
    float _pad0;
} params;

float gaussian(float r, float mu, float sigma) {
    return exp(-0.5 * ((r - mu) / sigma) * ((r - mu) / sigma));
}

// Kernel de Lenia adaptado a especies
float K(float r, float mu) {
    float k1 = gaussian(r, 0.15, 0.08);
    float k2 = gaussian(r, 0.5 + mu * 0.2, 0.12);
    float k3 = gaussian(r, 0.85, 0.08);
    return mix(k1 * 0.5 + k2, k2 + k3 * 0.8, mu);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = (vec2(pos) + 0.5) / params.u_res;
    
    if (pos.x >= int(params.u_res.x) || pos.y >= int(params.u_res.y)) {
        return;
    }
    
    float sum = 0.0;
    float totalWeight = 0.0;
    float centerMu = texture(u_tex_living, uv).g;
    
    int radius = int(params.u_R);
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            vec2 offset = vec2(float(x), float(y));
            float r = length(offset) / float(radius);
            if (r <= 1.0) {
                float w = K(r, centerMu);
                sum += texture(u_tex_living, uv + offset / params.u_res).r * w;
                totalWeight += w;
            }
        }
    }
    
    if (totalWeight > 0.0) {
        sum /= totalWeight;
    }
    
    imageStore(out_kernel, pos, vec4(sum, 0.0, 0.0, 1.0));
}
