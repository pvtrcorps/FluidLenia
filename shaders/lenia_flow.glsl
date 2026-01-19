#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_tex_living;
layout(set = 0, binding = 1) uniform sampler2D u_tex_waste;
layout(set = 0, binding = 2) uniform sampler2D u_tex_kernel;
layout(rgba32f, set = 0, binding = 3) uniform writeonly image2D out_living;
layout(rgba32f, set = 0, binding = 4) uniform writeonly image2D out_waste;

layout(std430, set = 0, binding = 5) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_decay;
    float u_eat_rate; // Reuse for general growth/metabolism rate if needed
    float u_diet_selectivity; // Unused for now or used for waste type matching
    float u_chemotaxis; // Unused in pure Lenia
    float u_mutation_rate;
    float u_inertia; // Genetic inertia
    float u_dietOffset;
    // -- Physics params removed --
    float _pad0;
    float _pad1;
    float _pad2;
    float _pad3;
    float _pad4;
    float _pad5;
    
    // Mouse/Brush
    vec2 u_mouseWorld;
    float u_mouseClick;
    float u_brushSize;
    float u_brushHue;
    float u_brushMode;
    float _pad6;
    float _pad7;
} params;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float getEffectiveMu(float gene) {
    return 0.08 + gene * 0.42;
}

float G(float u, float mu, float sigma) {
    float effMu = getEffectiveMu(mu);
    return 2.0 * exp(-0.5 * pow((u - effMu) / sigma, 2.0)) - 1.0;
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = (vec2(pos) + 0.5) / params.u_res;
    
    if (pos.x >= int(params.u_res.x) || pos.y >= int(params.u_res.y)) {
        return;
    }
    
    // Read current state
    vec4 state = texture(u_tex_living, uv);
    float mass = state.r;
    vec3 genes = state.gba; // r=Type, g=Diet, b=Sigma
    
    // Read Potential (Convolution result)
    float potential = texture(u_tex_kernel, uv).r;
    
    // --- GROWTH ---
    float growth = G(potential, genes.x, genes.z); // genes.x = Mu (Species Type)
    
    // Euler Step: Mass += dt * growth
    float newMass = mass + growth * params.u_dt;
    newMass = clamp(newMass, 0.0, 1.0);
    
    // --- GENETICS (Mutation & Inheritance) ---
    vec3 finalGenes = genes;
    
    // Only mutate if there is significant mass and growth is positive
    if (newMass > 0.01 && growth > 0.0 && params.u_mutation_rate > 0.0) {
        if (hash(uv + params.u_seed) < 0.1) { // 10% chance to mutate per step
             float drift = (hash(uv * 1.1 + params.u_seed) - 0.5) * params.u_mutation_rate;
             finalGenes.x = clamp(finalGenes.x + drift, 0.0, 1.0); // Drift Mu
             finalGenes.z = clamp(finalGenes.z + drift * 0.5, 0.01, 0.1); // Drift Sigma
        }
    }
    
    // If mass was very low (spawn), inherit from neighbors effectively? 
    // In pure grid Lenia, usually we just keep existing genes or blend with neighbors.
    // Here we rely on the fact that 'state' is sampled from the previous frame.
    // However, if we want "movement", genes must propagate. 
    // In Lenia, genes usually don't "flow", but if we want multi-species, we might need 
    // to sample genes from the "dominant" neighbor or just keep local.
    // For now, PRESERVE LOCAL GENES (simplest). 
    // *Critique*: If genes don't move, a moving glider will lose its identity as it enters empty space.
    // *Fix*: We need to sample genes from the 'source' of the growth.
    // But in pure Lenia without advection, this is tricky. 
    // Standard Multi-Species Lenia usually involves multiple channels or complex update rules.
    // Let's stick to: "If I am empty but growing, I inherit genes from the kernel convolution average?"
    // The kernel doesn't carry genes.
    // Let's just keep local genes for now. Gliders might work if the envelope moves.
    
    // --- WASTE ---
    // Simple production/decay without diffusion
    vec4 wData = texture(u_tex_waste, uv);
    float wMass = wData.r;
    float wType = wData.g;
    
    // Waste production roughly proportional to metabolism/mass
    float wasteProd = newMass * 0.05 * params.u_dt; 
    wMass += wasteProd;
    wMass -= wMass * params.u_decay * params.u_dt;
    wMass = clamp(wMass, 0.0, 1.0);
    
    // Update waste type to current species type if producing waste
    if (wasteProd > 0.0) {
        wType = mix(wType, finalGenes.x, 0.1);
    }

    // --- MOUSE INTERACTION ---
    vec2 dMouse = (uv - params.u_mouseWorld);
    dMouse.x *= params.u_res.x / params.u_res.y; // Aspect correction
    if (params.u_mouseClick > 0.5 && length(dMouse) * params.u_res.y < params.u_brushSize) {
        if (params.u_brushMode > 0.0) {
            newMass = 0.8;
            finalGenes = vec3(params.u_brushHue, params.u_brushHue, 0.03);
            wMass = 0.0;
        } else {
            newMass = 0.0;
            wMass = 0.0;
        }
    }
    
    // Write output
    imageStore(out_living, pos, vec4(newMass, finalGenes));
    imageStore(out_waste, pos, vec4(wMass, wType, 0.0, 0.0));
}
