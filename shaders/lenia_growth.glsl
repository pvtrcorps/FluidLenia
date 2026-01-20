#[compute]
#version 450

// Growth/Resolve Shader: Reads advected mass from accumulator, applies Lenia growth.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// INPUTS
layout(set = 0, binding = 0) uniform sampler2D u_tex_advected;      // Mass + Genes1
layout(set = 0, binding = 1) uniform sampler2D u_tex_living_prev;   // Fallback

// NEW INPUTS (Advected Winner)
layout(set = 0, binding = 8) uniform usampler2D u_tex_advected_species;
layout(set = 0, binding = 9) uniform sampler2D u_tex_advected_aux;

// OUTPUTS
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D out_living;
// NEW OUTPUTS
layout(r32ui, set = 0, binding = 10) uniform writeonly uimage2D out_species;
layout(rgba32f, set = 0, binding = 11) uniform writeonly image2D out_genes_aux;


// Context
layout(set = 0, binding = 3) uniform sampler2D u_tex_kernel;
layout(set = 0, binding = 5) uniform sampler2D u_tex_waste;
layout(rgba32f, set = 0, binding = 6) uniform writeonly image2D out_waste;
// NEW: Environment (Read Only)
layout(set = 0, binding = 12) uniform sampler2D u_tex_env;


layout(std430, set = 0, binding = 7) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_mutation_rate;
    float u_eat_rate;
    float u_decay;
    float u_inertia;
    float u_diet_selectivity;
    float u_global_scale; // NEW: Forced Normalization Factor
} params;


const float SPECIATION_THRESHOLD = 0.15; // genetic distance to trigger new species

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
    
    // 1. Read Inputs
    vec4 advectedState = texture(u_tex_advected, uv);
    float mass = advectedState.r; // New consolidated mass
    vec3 genes = advectedState.gba; // Winner Genes1
    
    // Read Environment (Global)
    vec4 env = texture(u_tex_env, uv);
    float localTemp = env.r;
    float localHazard = env.b;

    
    // Read Advected Extra Data
    uint speciesID = texture(u_tex_advected_species, uv).r;
    vec4 auxGenes = texture(u_tex_advected_aux, uv); // Winner Genes2 [Speed, Aggro, Def, Meta]
    
    vec4 prevState = texture(u_tex_living_prev, uv);
    
    // 2. Fallback Logic (if empty space became occupied by tiny amount, or pure birth)
    if (mass < 0.0001 || (genes.x == 0.0 && genes.z == 0.0)) {
         if (prevState.r > 0.01) {
             // Keep old local genes if we didn't get overridden by flow
             genes = prevState.gba;
             // We can't easily read old species/aux here without binding old textures... 
             // Assume they are lost or need separate binding?
             // For now, if mass is low, speciesID might effectively die or be noise.
         } else {
             // Spontaneous generation / Default
             genes = vec3(0.15, 0.15, 0.03); 
             // Don't set species to 0 immediately, generate a new one if it was 0
             if (speciesID == 0u) {
                 speciesID = uint(hash(uv + params.u_seed * 111.1) * 4000000000.0) + 1u;
             }

         }

    }
    
    float newMass = mass;
    
    // 3. Mutation Logic
    float potential = texture(u_tex_kernel, uv).r;
    float growth = G(potential, genes.x, genes.z); 
    
    vec3 finalGenes = genes;
    vec4 finalAux = auxGenes;
    uint finalSpecies = speciesID;
    
    if (newMass > 0.01 && growth > 0.0 && params.u_mutation_rate > 0.0) {
        if (hash(uv + vec2(params.u_seed)) < 0.2) {
            float drift = (hash(uv * 1.1 + vec2(params.u_seed)) - 0.5) * params.u_mutation_rate;

            
            // Mutate Primary Genes
            vec3 oldGenes = finalGenes;
            finalGenes.y = clamp(finalGenes.y + drift * 4.0, 0.0, 1.0); // Diet
            finalGenes.x = clamp(finalGenes.x + drift, 0.0, 1.0); // Struct
            finalGenes.z = clamp(finalGenes.z + drift * 0.5, 0.01, 0.1); // Sigma
            
            // Mutate Aux Genes
            finalAux += vec4(drift); 
            finalAux = clamp(finalAux, 0.0, 1.0); // Normalize 0-1
            
            // Speciation Event?
            float genDist = length(finalGenes - oldGenes);
            if (genDist > SPECIATION_THRESHOLD * 0.1) { // lowered trigger for testing
                 // Generate new ID (Probabilistic)
                 if (hash(uv * 9.0 + vec2(params.u_seed)) < 0.01) {
                     finalSpecies = uint(hash(uv + vec2(params.u_seed, mass)) * 4000000000.0);
                 }
            }

        }
    }
    
    // 4. Inertia (Genotype resistance)
    // Only applies to Genes1 for now as we don't bind PrevAux
    if (prevState.r > 0.01) {
        finalGenes = mix(finalGenes, prevState.gba, params.u_inertia);
    }
    
    // 5. Metabolism (Eating Waste)
    vec4 wCenter = texture(u_tex_waste, uv);
    vec2 px = 1.0 / params.u_res;
    vec4 wL = texture(u_tex_waste, uv - vec2(px.x, 0));
    vec4 wR = texture(u_tex_waste, uv + vec2(px.x, 0));
    vec4 wT = texture(u_tex_waste, uv + vec2(0, px.y));
    vec4 wB = texture(u_tex_waste, uv - vec2(0, px.y));
    
    vec4 wasteDiffusion = (wL + wR + wT + wB) * 0.25 - wCenter;
    vec4 diffusedWaste = wCenter + wasteDiffusion * 0.5;
    
    float wMass = max(0.0, diffusedWaste.r);
    float wType = diffusedWaste.g;
    
    float eaten = 0.0;
    if (newMass > 0.001 && wMass > 0.001) {
        float targetType = finalGenes.y; 
        float enzyme = getEffectiveMu(targetType);
        float wasteStruct = getEffectiveMu(wType);
        float similarity = 1.0 - abs(enzyme - wasteStruct);
        float threshold = 1.0 - (0.5 / params.u_diet_selectivity);
        float efficiency = smoothstep(threshold, 1.0, similarity);
        
        // METABOLISM GENE EFFECT (Aux.a) -> Temperature Preference
        // vec4 env = texture(u_tex_env, uv); // MOVED UP
        // float localTemp = env.r; // MOVED UP
        float optimalTemp = finalAux.a; // Gene: Preferred Temperature
        
        // Bell curve efficiency based on Temp matching
        float tempDist = localTemp - optimalTemp;

        float thermalEff = exp(-5.0 * tempDist * tempDist); // Narrow niche
        
        // Combine diet efficiency with thermal efficiency
        eaten = min(wMass, newMass * params.u_eat_rate * efficiency * thermalEff * params.u_dt);

        newMass += eaten;
        wMass -= eaten;
    }
    
    // 6. STARVATION-BASED DEATH (Mass Conserving)
    // Death occurs when:
    // - Overcrowding (mass > threshold)
    // - Starvation (didn't eat enough to sustain)
    // - Hazard damage
    
    float crowdPenalty = max(0.0, newMass - 1.5) * 2.0; // Overcrowding
    
    // Starvation: If growth is negative and not eating, starve
    float starvationRate = 0.0;
    if (growth < 0.0 && eaten < 0.001) {
        starvationRate = params.u_decay * 2.0; // Starving cells decay faster
    }
    
    // Hazard Effect
    float defense = finalAux.b; // Gene: Defense/Armor
    float hazardDamage = max(0.0, localHazard - defense * 0.8) * 0.1;
    
    // Total death rate (minimal base decay to ensure some turnover)
    float deathRate = params.u_decay * 0.1 + crowdPenalty + starvationRate + hazardDamage;
    
    float deadMass = min(newMass, newMass * deathRate * params.u_dt);
    
    newMass -= deadMass;
    float newWasteMass = wMass + deadMass; // Mass is conserved: living -> waste
    
    float newWasteType = wType;
    if (newWasteMass > 0.0001 && deadMass > 0.0) {
        newWasteType = mix(wType, finalGenes.x, deadMass / newWasteMass);
    }

    
    // If mass is too low, treat as Void (Clean background)
    // CRITICAL: Conserve this mass! Don't just delete it.
    if (newMass < 0.01) {
        finalSpecies = 0u; 
        
        // Recycle the tiny mass into waste to prevent energy leaks
        newWasteMass += newMass;
        if (newWasteMass > 0.0001) {
             // Keeps waste type same or mixes slightly, but mass is safe
        }
        
        newMass = 0.0; // Clean exit
    }



    
    
    // 7. GLOBAL NORMALIZATION (Forced Constant Mass)
    // Apply the scale factor calculated by the Sum shader
    newMass *= params.u_global_scale;
    newWasteMass *= params.u_global_scale;

    
    // 8. Write Outputs
    imageStore(out_living, pos, vec4(newMass, finalGenes));
    imageStore(out_waste, pos, vec4(newWasteMass, newWasteType, 0.0, 1.0));

    
    imageStore(out_species, pos, uvec4(finalSpecies, 0u, 0u, 0u));
    imageStore(out_genes_aux, pos, finalAux);

}
