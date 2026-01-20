#[compute]
#version 450

// Advection Shader: Neighbor Gather & Winner-Takes-All
// Reads from Living (Mass + Genes) + Velocity + Species + AuxGenes.
// Writes to Advected Texture (Mass + Dominant Genes + Species).

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_tex_living;
layout(set = 0, binding = 1) uniform sampler2D u_tex_velocity;

// Single RGBA32F output for Mass + Genes
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D out_advected;

layout(std430, set = 0, binding = 3) buffer Params {
    vec2 u_res;
    float u_dt;
    float _pad0;
} params;

// NEW BINDINGS FOR SPECIES & GENES
layout(set = 0, binding = 4) uniform usampler2D u_tex_species;
layout(set = 0, binding = 5) uniform sampler2D u_tex_genes_aux;

layout(r32ui, set = 0, binding = 6) uniform writeonly uimage2D out_advected_species;
layout(rgba32f, set = 0, binding = 7) uniform writeonly image2D out_advected_aux;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = (vec2(pos) + 0.5) / params.u_res;
    vec2 px = 1.0 / params.u_res;
    
    if (pos.x >= int(params.u_res.x) || pos.y >= int(params.u_res.y)) {
        return;
    }
    
    // GATHER Logic: Loop neighbors to find who moved HERE.
    int R = 2;
    
    float totalMass = 0.0;
    
    // Winner Tracking
    float winnerMass = -1.0;
    vec3 winnerGenes = vec3(0.0); // Default traits
    uint winnerSpecies = 0;
    vec4 winnerAuxGenes = vec4(0.0);
    
    for (int y = -R; y <= R; y++) {
        for (int x = -R; x <= R; x++) {
            vec2 offset = vec2(float(x), float(y));
            vec2 srcUV = uv + offset * px;
            
            // Source state
            vec4 srcState = texture(u_tex_living, srcUV);
            float m = srcState.r;
            
            if (m > 0.0001) {
                // Check flow
                vec2 vel = texture(u_tex_velocity, srcUV).rg;
                
                // Where does src land?
                vec2 destUV = srcUV + vel * params.u_dt * px;
                
                // Distance to ME (uv)
                vec2 distVec = (destUV - uv) * params.u_res; 
                
                // Bilinear weight
                float weight = max(0.0, 1.0 - abs(distVec.x)) * max(0.0, 1.0 - abs(distVec.y));
                
                if (weight > 0.0) {
                    float incomingMass = m * weight;
                    totalMass += incomingMass;
                    
                    // Winner Takes All: Gene Logic
                    if (incomingMass > winnerMass) {
                        winnerMass = incomingMass;
                        winnerGenes = srcState.gba;
                        
                        // Sample extra data
                        winnerSpecies = texture(u_tex_species, srcUV).r;
                        winnerAuxGenes = texture(u_tex_genes_aux, srcUV);
                    }

                }
            }
        }
    }
    
    // Fallback: If mass exists but somehow winnerSpecies is 0 (should correspond to winnerGenes logic)
    // Actually, if totalMass > 0, winnerMass > -1, so we must have entered the if(incoming > winner) block at least once.
    // UNLESS srcState.r was > 0.0001 but texture(species) was 0.
    
    // Output
    imageStore(out_advected, pos, vec4(totalMass, winnerGenes));

    imageStore(out_advected_species, pos, uvec4(winnerSpecies, 0u, 0u, 0u));
    imageStore(out_advected_aux, pos, winnerAuxGenes);
}

