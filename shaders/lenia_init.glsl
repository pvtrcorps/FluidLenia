#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba32f, set = 0, binding = 0) uniform writeonly image2D out_living;
layout(rgba32f, set = 0, binding = 1) uniform writeonly image2D out_living_b; 
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

layout(r32ui, set = 0, binding = 5) uniform writeonly uimage2D out_species;
layout(r32ui, set = 0, binding = 6) uniform writeonly uimage2D out_species_b;
layout(rgba32f, set = 0, binding = 7) uniform writeonly image2D out_genes_aux;
layout(rgba32f, set = 0, binding = 8) uniform writeonly image2D out_genes_aux_b;
layout(rgba32f, set = 0, binding = 9) uniform writeonly image2D out_env;

// Robust hash function with even distribution across 0-1
float hash(vec2 p) {
    p = fract(p * vec2(443.8975, 397.2973));
    p += dot(p.yx, p + vec2(19.19, 47.47));
    return fract(p.x * p.y);
}


void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = (vec2(pos) + vec2(0.5)) / params.u_res;
    
    if (pos.x >= int(params.u_res.x) || pos.y >= int(params.u_res.y)) {
        return;
    }
    
    vec2 cellID = floor(uv * params.u_initGrid);
    
    // SAFE SEEDING: Handle large time-based seeds by wrapping them.
    // Modulo 1000.0 keeps it in a safe float precision range without losing variety.
    float safeSeed = mod(params.u_seed, 1000.0); 
    vec2 seedVec = vec2(safeSeed, safeSeed + 31.41);
    
    // ALL cells get mass (density controls the amount, not probability)
    float mass = params.u_density;
    
    // --- ROBUST RANDOMIZATION ---
    // Use distinct offsets for each gene to avoid correlation
    // Use the safe seed in the hash chain
    
    float cellHash = hash(cellID + seedVec);
    
    float muStruct = hash(cellID + vec2(cellHash * 12.3)); // Structure: Base Hue
    
    float drift = (hash(cellID * 1.1 + vec2(safeSeed)) - 0.5) * 0.4;
    float muDiet = fract(muStruct + drift);
    
    // Sigma: 0.01 to 0.05
    float sig = 0.01 + hash(cellID * 1.5 + vec2(muStruct)) * 0.04;
    
    // Species ID (Unique INT)
    uint speciesID = uint(hash(cellID + vec2(safeSeed * 0.1)) * 4000000000.0) + 1u;
    
    // Aux Genes
    float g_speed = 0.5 + hash(cellID * 2.1 + vec2(safeSeed)) * 1.0; // 0.5 - 1.5
    float g_aggro = hash(cellID * 3.7 + vec2(g_speed));              // 0.0 - 1.0
    float g_def = hash(cellID * 4.2 + vec2(safeSeed * 0.5));         // 0.0 - 1.0
    float g_meta = 0.8 + hash(cellID * 5.5 + vec2(g_aggro)) * 0.4;   // 0.8 - 1.2
    
    vec4 auxGenes = vec4(g_speed, g_aggro, g_def, g_meta);



    // Write living state
    imageStore(out_living, pos, vec4(mass, muStruct, muDiet, sig));
    imageStore(out_living_b, pos, vec4(0.0));
    
    // Write species
    imageStore(out_species, pos, uvec4(speciesID, 0u, 0u, 0u));
    imageStore(out_species_b, pos, uvec4(0u, 0u, 0u, 0u));
    
    // Write aux genes
    imageStore(out_genes_aux, pos, auxGenes);
    imageStore(out_genes_aux_b, pos, vec4(0.0));
    
    // Environment: Simple gradients (no noise for now)
    float temp = uv.y;
    float resVal = 1.0 - length(uv - vec2(0.5)) * 1.5;
    resVal = clamp(resVal, 0.2, 1.0);
    float hazard = 0.0;
    
    imageStore(out_env, pos, vec4(temp, resVal, hazard, 1.0));

    
    // Waste
    float wasteNoise = hash(uv * vec2(10.0) + seedVec);
    vec4 wasteVal = vec4(wasteNoise * 0.2, hash(uv * vec2(5.0)), 0.0, 0.0);
    imageStore(out_waste, pos, wasteVal);
    imageStore(out_waste_b, pos, wasteVal);
}
