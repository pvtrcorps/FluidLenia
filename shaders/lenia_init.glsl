#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba32f, set = 0, binding = 0) uniform writeonly image2D out_living;
layout(rgba32f, set = 0, binding = 1) uniform writeonly image2D out_living_b; // We init both to be safe
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D out_waste;
layout(rgba32f, set = 0, binding = 3) uniform writeonly image2D out_waste_b;

layout(std430, set = 0, binding = 4) buffer Params {
    vec2 u_res;
    float u_seed;
    float u_density;
    float u_initGrid;
    float _pad0;
    float _pad1;
    float _pad2;
} params;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = (vec2(pos) + 0.5) / params.u_res;
    
    if (pos.x >= int(params.u_res.x) || pos.y >= int(params.u_res.y)) {
        return;
    }
    
    vec2 cellID = floor(uv * params.u_initGrid);
    float cellProb = hash(cellID + params.u_seed * 33.33);
    
    // Block-based initialization (Solid blocks)
    float mass = step(1.0 - params.u_density, cellProb);
    
    float cellSeed = hash(cellID + params.u_seed * 77.7);
    
    float muStruct = cellSeed;
    float drift = (hash(cellID * 1.1) - 0.5) * 0.2;
    float muDiet = fract(muStruct + drift);
    float sig = 0.015 + hash(cellID * 1.5 + params.u_seed) * 0.035;
    
    imageStore(out_living, pos, vec4(mass, muStruct, muDiet, sig));
    
    // Inicio con residuos ambientales para arrancar el metabolismo
    float wasteNoise = hash(uv * 10.0 + params.u_seed);
    vec4 wasteVal = vec4(wasteNoise * 0.2, hash(uv * 5.0), 0.0, 0.0);
    
    imageStore(out_waste, pos, wasteVal);
    imageStore(out_waste_b, pos, wasteVal);
    imageStore(out_living_b, pos, vec4(0.0)); // Clear secondary buffer
}
