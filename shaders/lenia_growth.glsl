#[compute]
#version 450

// Growth/Resolve Shader: Reads advected mass from accumulator, applies Lenia growth.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Integer accumulator from advection pass
// Advected input (Mass + Genes)
layout(set = 0, binding = 0) uniform sampler2D u_tex_advected;

// Previous living state (still needed for fallback/logic?)
layout(set = 0, binding = 1) uniform sampler2D u_tex_living_prev;

// Output
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D out_living;

// Kernel/Potential for growth calculation
layout(set = 0, binding = 3) uniform sampler2D u_tex_kernel;

// Waste (Source and Dest)
layout(set = 0, binding = 5) uniform sampler2D u_tex_waste;
layout(rgba32f, set = 0, binding = 6) uniform writeonly image2D out_waste;

layout(std430, set = 0, binding = 7) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_mutation_rate;
    float u_eat_rate;
    float u_decay;
    float u_inertia;
    float u_diet_selectivity;
} params;

const int PRECISION = 16384;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// CRITICAL: Maps gene to effective mu (matches prototype exactly)
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
    
    // Read advected state (Mass + Winner Genes)
    vec4 advectedState = texture(u_tex_advected, uv);
    float mass = advectedState.r;
    
    // Genes from previous state (for fallback / inertia)
    vec4 prevState = texture(u_tex_living_prev, uv);
    
    // Default genes logic
    vec3 genes = advectedState.gba; // The winner genes
    
    // If no mass arrived, or genes are empty/invalid, use fallback
    if (mass < 0.0001 || (genes.x == 0.0 && genes.z == 0.0)) {
         // Inherit local if possible, or spawn default
         if (prevState.r > 0.01) {
             genes = prevState.gba;
         } else {
             genes = vec3(0.15, 0.15, 0.03); 
         }
    }
    
    // MASS CONSERVATION: Growth affects MOVEMENT (in velocity shader), not mass creation!
    // Mass is ONLY transferred between Living <-> Waste, never created or destroyed.
    // The advected mass IS the new mass - no growth term adding/removing mass.
    float newMass = mass;
    
    // Mutation (based on whether cell is healthy/active, use potential as proxy)
    float potential = texture(u_tex_kernel, uv).r;
    float mu = genes.x;
    float sigma = genes.z;
    float growth = G(potential, mu, sigma); // Used only for mutation trigger, not mass
    
    vec3 finalGenes = genes;
    if (newMass > 0.01 && growth > 0.0 && params.u_mutation_rate > 0.0) {
        if (hash(uv + params.u_seed) < 0.2) {
            float drift = (hash(uv * 1.1 + params.u_seed) - 0.5) * params.u_mutation_rate;
            finalGenes.y = clamp(finalGenes.y + drift * 4.0, 0.0, 1.0); // Diet
            finalGenes.x = clamp(finalGenes.x + drift, 0.0, 1.0); // Struct
            finalGenes.z = clamp(finalGenes.z + drift * 0.5, 0.01, 0.1); // Sigma
        }
    }
    
    // Inertia (Resistance to gene change)
    if (prevState.r > 0.01) {
        finalGenes = mix(finalGenes, prevState.gba, params.u_inertia);
    }
    
    // Note: We DON'T write out_living here yet - we do it AFTER metabolism
    
    // --- METABOLIC CYCLE & WASTE ---
    
    // 1. Read Waste state
    vec4 wCenter = texture(u_tex_waste, uv);
    
    // 2. Simple Diffusion (Blur)
    vec2 px = 1.0 / params.u_res;
    vec4 wL = texture(u_tex_waste, uv - vec2(px.x, 0));
    vec4 wR = texture(u_tex_waste, uv + vec2(px.x, 0));
    vec4 wT = texture(u_tex_waste, uv + vec2(0, px.y));
    vec4 wB = texture(u_tex_waste, uv - vec2(0, px.y));
    
    vec4 wasteDiffusion = (wL + wR + wT + wB) * 0.25 - wCenter;
    vec4 diffusedWaste = wCenter + wasteDiffusion * 0.5;
    
    float wMass = max(0.0, diffusedWaste.r);
    float wType = diffusedWaste.g;
    
    // 3. Eating (Waste -> Mass) with DIET SELECTIVITY
    // Prototype: Cells only efficiently eat waste matching their diet
    float eaten = 0.0;
    if (newMass > 0.001 && wMass > 0.001) {
        // Calculate diet matching
        float targetType = finalGenes.y; // Diet gene
        float enzyme = getEffectiveMu(targetType);
        float wasteStruct = getEffectiveMu(wType);
        
        // How similar is the enzyme to the waste structure?
        float similarity = 1.0 - abs(enzyme - wasteStruct);
        
        // Efficiency uses smoothstep - sharper cutoff based on selectivity
        // Higher selectivity = narrower efficiency window
        float threshold = 1.0 - (0.5 / params.u_diet_selectivity);
        float efficiency = smoothstep(threshold, 1.0, similarity);
        
        eaten = min(wMass, newMass * params.u_eat_rate * efficiency * params.u_dt);
        newMass += eaten;
        wMass -= eaten;
    }
    
    // 4. Decay (Mass -> Waste)
    // Prototype: crowdPenalty = max(0.0, newMass - 1.2) * 1.5;
    float crowdPenalty = max(0.0, newMass - 1.2) * 1.5; // Changed from 1.0 to 1.2
    float deathRate = params.u_decay + crowdPenalty;
    float deadMass = min(newMass, newMass * deathRate);
    
    newMass -= deadMass;
    float newWasteMass = wMass + deadMass;
    
    // 5. Waste Type mixing
    float newWasteType = wType;
    if (newWasteMass > 0.0001 && deadMass > 0.0) {
        newWasteType = mix(wType, finalGenes.x, deadMass / newWasteMass);
    }
    
    // Update Living with new mass after metabolism
    imageStore(out_living, pos, vec4(newMass, finalGenes));
    
    // Update Waste
    imageStore(out_waste, pos, vec4(newWasteMass, newWasteType, 0.0, 1.0));
}
