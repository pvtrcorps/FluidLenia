#[compute]
#version 450

// Lenia Sum Reduction Shader
// Sums Living + Waste mass to calculate total system mass.
// Optimized reduction using shared memory.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_tex_living;
layout(set = 0, binding = 1) uniform sampler2D u_tex_waste;

layout(std430, set = 0, binding = 2) volatile buffer SumBuffer {
    uint total_mass_x1000;
};


layout(std430, set = 0, binding = 3) buffer Params {
    vec2 u_res;
    float _pad0;
    float _pad1;
} params;

// Shared memory for workgroup reduction (8x8 = 64 threads)
shared float s_mass[64];

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    uint tid = gl_LocalInvocationIndex; // 0 to 63
    
    float mass = 0.0;
    
    if (pos.x < int(params.u_res.x) && pos.y < int(params.u_res.y)) {
        vec2 uv = (vec2(pos) + 0.5) / params.u_res;
        
        mass += texture(u_tex_living, uv).r;
        mass += texture(u_tex_waste, uv).r;
    }
    
    s_mass[tid] = mass;
    barrier();
    
    // Parallel Reduction in Shared Memory
    // 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    if (tid < 32) s_mass[tid] += s_mass[tid + 32]; barrier();
    if (tid < 16) s_mass[tid] += s_mass[tid + 16]; barrier();
    if (tid < 8)  s_mass[tid] += s_mass[tid + 8];  barrier();
    if (tid < 4)  s_mass[tid] += s_mass[tid + 4];  barrier();
    if (tid < 2)  s_mass[tid] += s_mass[tid + 2];  barrier();
    
    if (tid == 0) {
        float groupSum = s_mass[0] + s_mass[1];
        
        // Atomic Add to global buffer using fixed point
        uint fixedPoint = uint(groupSum * 1000.0);
        atomicAdd(total_mass_x1000, fixedPoint);
    }
}

