#[compute]
#version 450

// Advection Shader: Neighbor Gather & Winner-Takes-All
// Reads from Living (Mass + Genes) + Velocity.
// Writes to Advected Texture (Mass + Dominant Genes).

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

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = (vec2(pos) + 0.5) / params.u_res;
    vec2 px = 1.0 / params.u_res;
    
    if (pos.x >= int(params.u_res.x) || pos.y >= int(params.u_res.y)) {
        return;
    }
    
    // GATHER Logic: Loop neighbors to find who moved HERE.
    // Search Radius must cover max velocity * dt. 
    // Max vel ~ 4.0? dt ~ 0.25? -> 1 px.
    // Let's use Radius 2 to be safe for larger flows.
    int R = 2;
    
    float totalMass = 0.0;
    float winnerMass = -1.0;
    vec3 winnerGenes = vec3(0.0); // Default traits
    
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
                // Note: UV space logic
                vec2 destUV = srcUV + vel * params.u_dt * px;
                
                // Distance to ME (uv)
                // Use pixels for precise distance
                vec2 distVec = (destUV - uv) * params.u_res; 
                
                // Bilinear weight (standard scatter/gather dual)
                // weight = max(0, 1 - |dx|) * max(0, 1 - |dy|)
                float weight = max(0.0, 1.0 - abs(distVec.x)) * max(0.0, 1.0 - abs(distVec.y));
                
                if (weight > 0.0) {
                    float incomingMass = m * weight;
                    totalMass += incomingMass;
                    
                    // Winner Takes All: Gene Logic
                    if (incomingMass > winnerMass) {
                        winnerMass = incomingMass;
                        winnerGenes = srcState.gba;
                    }
                }
            }
        }
    }
    
    // Keep consistent genes if empty? Default genes handled in Growth.
    // Just output what we found.
    imageStore(out_advected, pos, vec4(totalMass, winnerGenes));
}
