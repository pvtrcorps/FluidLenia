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
    float u_eat_rate;
    float u_diet_selectivity;
    float u_chemotaxis;
    float u_mutation_rate;
    float u_inertia;
    float u_dietOffset;
    float u_gravity;
    float u_floor;
    float u_immiscibility;
    float u_friction;
    float u_vel_impact;
    float _pad0;
    // Mouse/Brush
    vec2 u_mouseWorld;
    float u_mouseClick;
    float u_brushSize;
    float u_brushHue;
    float u_brushMode;
    float _pad1;
    float _pad2;
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
    vec2 px = 1.0 / params.u_res;
    
    if (pos.x >= int(params.u_res.x) || pos.y >= int(params.u_res.y)) {
        return;
    }
    
    // --- 1. ADVECCIÓN DE BIOMASA + MOMENTO ---
    float newMass = 0.0;
    
    // Acumuladores
    vec3 traitAccumulator = vec3(0.0);
    float weightAccumulator = 0.0;
    vec2 momentumAccumulator = vec2(0.0);
    
    int searchR = 3;
    
    for (int y = -searchR; y <= searchR; y++) {
        for (int x = -searchR; x <= searchR; x++) {
            vec2 offset = vec2(float(x), float(y));
            vec2 srcUV = uv + offset * px;
            
            if (params.u_floor > 0.5 && (srcUV.y < 0.0 || srcUV.y > 1.0)) continue;
            
            vec4 srcState = texture(u_tex_living, srcUV);
            float m = srcState.r;
            
            if (m > 0.001) {
                // --- FÍSICA NEWTONIANA ---
                vec2 oldVel = texture(u_tex_waste, srcUV).ba;
                
                // Calcular Fuerza (Aceleración)
                float K_C = texture(u_tex_kernel, srcUV).r;
                float K_L = texture(u_tex_kernel, srcUV - vec2(px.x, 0)).r;
                float K_R = texture(u_tex_kernel, srcUV + vec2(px.x, 0)).r;
                float K_B = texture(u_tex_kernel, srcUV - vec2(0, px.y)).r;
                float K_T = texture(u_tex_kernel, srcUV + vec2(0, px.y)).r;
                
                float muStr = srcState.g;
                float muDiet = srcState.b;
                float sig = srcState.a;
                vec2 grad_U = vec2(G(K_R, muStr, sig) - G(K_L, muStr, sig), 
                                   G(K_T, muStr, sig) - G(K_B, muStr, sig)) * 0.5;
                
                float M_L = texture(u_tex_living, srcUV - vec2(px.x, 0)).r;
                float M_R = texture(u_tex_living, srcUV + vec2(px.x, 0)).r;
                float M_B = texture(u_tex_living, srcUV - vec2(0, px.y)).r;
                float M_T = texture(u_tex_living, srcUV + vec2(0, px.y)).r;
                vec2 grad_M = vec2(M_R - M_L, M_T - M_B) * 0.5;
                
                vec2 grad_Food = vec2(0.0);
                if (params.u_chemotaxis > 0.0) {
                    vec4 wC = texture(u_tex_waste, srcUV);
                    float hunger = max(0.0, 1.0 - wC.r * 1.5);
                    if (hunger > 0.0) {
                        vec4 wL = texture(u_tex_waste, srcUV - vec2(px.x, 0));
                        vec4 wR = texture(u_tex_waste, srcUV + vec2(px.x, 0));
                        vec4 wB = texture(u_tex_waste, srcUV - vec2(0, px.y));
                        vec4 wT = texture(u_tex_waste, srcUV + vec2(0, px.y));
                        float targetType = fract(muDiet + params.u_dietOffset);
                        float affL = (1.0 - abs(wL.g - targetType)) * wL.r;
                        float affR = (1.0 - abs(wR.g - targetType)) * wR.r;
                        float affB = (1.0 - abs(wB.g - targetType)) * wB.r;
                        float affT = (1.0 - abs(wT.g - targetType)) * wT.r;
                        grad_Food = vec2(affR - affL, affT - affB) * hunger;
                    }
                }
                
                vec2 acceleration = grad_U - grad_M * 2.0 + grad_Food * params.u_chemotaxis;
                acceleration.y -= params.u_gravity * 8.0;
                
                // Integración de Euler
                vec2 dynamicVel = oldVel * params.u_friction + acceleration * params.u_dt;
                
                // Limite de velocidad
                if (length(dynamicVel) > 5.0) dynamicVel = normalize(dynamicVel) * 5.0;
                
                // Desplazamiento
                vec2 dest = srcUV + dynamicVel * params.u_dt * px;
                
                if (params.u_floor > 0.5) dest.y = clamp(dest.y, 0.001, 0.999);
                
                vec2 distVec = (dest - uv) * params.u_res;
                float weight = max(0.0, 1.0 - abs(distVec.x)) * max(0.0, 1.0 - abs(distVec.y));
                
                if (weight > 0.0) {
                    float incomingMass = m * weight;
                    newMass += incomingMass;
                    
                    // Mezcla de fluidos potenciada por velocidad
                    float velocityFactor = 1.0 + length(dynamicVel) * params.u_vel_impact;
                    float effectiveImpact = incomingMass * velocityFactor;
                    float sharpWeight = pow(effectiveImpact, params.u_immiscibility);
                    
                    traitAccumulator += srcState.gba * sharpWeight;
                    weightAccumulator += sharpWeight;
                    
                    // Conservación de momento
                    momentumAccumulator += dynamicVel * incomingMass;
                }
            }
        }
    }
    
    vec3 finalTraits = vec3(0.0, 0.0, 0.03);
    vec2 finalVel = vec2(0.0);
    
    if (newMass > 0.0001) {
        if (weightAccumulator > 0.0) {
            finalTraits = traitAccumulator / weightAccumulator;
        }
        finalVel = momentumAccumulator / newMass;
    }
    
    // Mutación
    if (newMass > 0.01 && params.u_mutation_rate > 0.0) {
        if (hash(uv + params.u_seed) < 0.2) {
            float drift = (hash(uv * 1.1 + params.u_seed) - 0.5) * params.u_mutation_rate;
            finalTraits.y = clamp(finalTraits.y + drift * 4.0, 0.0, 1.0);
            finalTraits.x = clamp(finalTraits.x + drift, 0.0, 1.0);
            finalTraits.z = clamp(finalTraits.z + drift * 0.5, 0.01, 0.1);
        }
    }
    
    // Inercia Genética
    vec4 oldState = texture(u_tex_living, uv);
    if (oldState.r > 0.01) {
        finalTraits = mix(finalTraits, oldState.gba, params.u_inertia);
    }
    
    // --- 2. GESTIÓN DE RESIDUOS Y DIFUSIÓN ---
    vec4 wCenter = texture(u_tex_waste, uv);
    vec4 wL = texture(u_tex_waste, uv - vec2(px.x, 0));
    vec4 wR = texture(u_tex_waste, uv + vec2(px.x, 0));
    vec4 wT = texture(u_tex_waste, uv + vec2(0, px.y));
    vec4 wB = texture(u_tex_waste, uv - vec2(0, px.y));
    
    // Difusión simple
    vec4 wasteDiffusion = (wL + wR + wT + wB) * 0.25 - wCenter;
    vec4 diffusedWaste = wCenter + wasteDiffusion * 0.5;
    
    float wMass = max(0.0, diffusedWaste.r);
    float wType = diffusedWaste.g;
    
    // Digestión
    float targetType = fract(finalTraits.y + params.u_dietOffset);
    float enzyme = getEffectiveMu(targetType);
    float wasteStruct = getEffectiveMu(wType);
    float similarity = 1.0 - abs(enzyme - wasteStruct);
    float efficiency = smoothstep(1.0 - (0.5 / params.u_diet_selectivity), 1.0, similarity);
    
    float eaten = 0.0;
    if (newMass > 0.001 && wMass > 0.001) {
        eaten = min(wMass, newMass * params.u_eat_rate * efficiency * params.u_dt);
        newMass += eaten;
        wMass -= eaten;
    }
    
    // Muerte/Hacinamiento
    float crowdPenalty = max(0.0, newMass - 1.2) * 1.5;
    float deathRate = params.u_decay + crowdPenalty;
    float deadMass = min(newMass, newMass * deathRate);
    
    newMass -= deadMass;
    float newWasteMass = wMass + deadMass;
    
    float newWasteType = wType;
    if (newWasteMass > 0.0001 && deadMass > 0.0) {
        newWasteType = mix(wType, finalTraits.x, deadMass / newWasteMass);
    }
    
    // Pincel de Dios
    vec2 dMouse = (uv - params.u_mouseWorld);
    dMouse.x *= params.u_res.x / params.u_res.y;
    if (params.u_mouseClick > 0.5 && length(dMouse) * params.u_res.y < params.u_brushSize) {
        if (params.u_brushMode > 0.0) {
            newMass += 0.5;
            finalTraits = vec3(params.u_brushHue, params.u_brushHue, 0.03);
        } else {
            newMass = 0.0;
            newWasteMass = 0.0;
            finalVel = vec2(0.0);
        }
    }
    
    // Guardar velocidad en canales B y A de residuos
    vec2 outputVel = mix(diffusedWaste.ba, finalVel, 0.5);
    if (newMass < 0.001) outputVel = vec2(0.0);
    
    imageStore(out_living, pos, vec4(newMass, finalTraits));
    imageStore(out_waste, pos, vec4(newWasteMass, newWasteType, outputVel.x, outputVel.y));
}
