#[compute]
#version 450

// Lenia Stats Shader
// Aggregates population and gene statistics for each species ID.
// Uses a fixed-size hash map in a storage buffer.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform usampler2D u_tex_species;
layout(set = 0, binding = 1) uniform sampler2D u_tex_genes_aux;
layout(set = 0, binding = 4) uniform sampler2D u_tex_living; // For Mu


struct SpeciesStats {
    uint species_id;
    uint count;
    uint speed_sum_x1000; // Fixed point accumulators
    uint aggro_sum_x1000;
    uint mu_sum_x1000;    // NEW: For phenotype color
    uint _pad0;           // Padding for alignment (struct now 24 bytes)
};

// Max distinct species to track
const uint MAX_SPECIES = 256; 

layout(std430, set = 0, binding = 2) volatile buffer StatsBuffer {
    SpeciesStats species[256]; 
};



layout(std430, set = 0, binding = 3) buffer Params {
    vec2 u_res;
    float _pad0;
    float _pad1;
} params;

// Simple Jenkins hash for probing
uint hash(uint x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= int(params.u_res.x) || pos.y >= int(params.u_res.y)) {
        return;
    }
    
    vec2 uv = (vec2(pos) + 0.5) / params.u_res;
    
    // Read Species ID
    uint id = texture(u_tex_species, uv).r;
    
    if (id == 0) return; // Ignore void
    
    // Read Stats (Speed, Aggro, Mu)
    vec4 aux = texture(u_tex_genes_aux, uv);
    uint speed_val = uint(aux.r * 1000.0);
    uint aggro_val = uint(aux.g * 1000.0);
    
    vec4 living = texture(u_tex_living, uv);
    uint mu_val = uint(living.g * 1000.0); // Mu is in green channel

    
    // Linear Probing to find slot
    uint h = hash(id) % MAX_SPECIES;
    
    for (uint i = 0; i < MAX_SPECIES; i++) {
        uint slot = (h + i) % MAX_SPECIES;
        
        // Try to find or claim slot
        uint prev_id = atomicCompSwap(species[slot].species_id, 0u, id);
        
        if (prev_id == 0u || prev_id == id) {
             atomicAdd(species[slot].count, 1u);
             atomicAdd(species[slot].speed_sum_x1000, speed_val);
             atomicAdd(species[slot].aggro_sum_x1000, aggro_val);
             atomicAdd(species[slot].mu_sum_x1000, mu_val);

             break;

        }
        // If prev_id != 0 and != id, slot is occupied by collision -> continue probing
    }
}
