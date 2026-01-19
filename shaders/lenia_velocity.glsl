#[compute]
#version 450

// Velocity Shader: Matches Prototype frag-flow Logic
// Calculates flow based on:
// 1. Growth Gradient (grad_U) - Move toward favorable growth
// 2. Mass Repulsion (grad_M) - Avoid overcrowding  
// 3. Chemotaxis (grad_Food) - Seek food based on diet

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_tex_kernel;
layout(set = 0, binding = 1) uniform sampler2D u_tex_living;
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D out_velocity;
layout(set = 0, binding = 3) uniform sampler2D u_tex_waste;

layout(std430, set = 0, binding = 4) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_friction;
    float u_chemotaxis;
} params;

// CRITICAL: Maps gene to effective mu (matches prototype exactly)
float getEffectiveMu(float gene) {
    return 0.08 + gene * 0.42;
}

// Growth function (Lenia) - MUST use effective mu
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
    
    // Read THIS cell's state (genes)
    vec4 state = texture(u_tex_living, uv);
    float mass = state.r;
    float muStruct = state.g;
    float muDiet = state.b;
    float sigma = state.a;
    
    // Default to small velocity if no mass
    if (mass < 0.001) {
        imageStore(out_velocity, pos, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    
    // Ensure sigma is valid
    if (sigma < 0.01) sigma = 0.03;
    
    // Sample kernel (potential field) for gradient
    float K_C = texture(u_tex_kernel, uv).r;
    float K_L = texture(u_tex_kernel, uv - vec2(px.x, 0)).r;
    float K_R = texture(u_tex_kernel, uv + vec2(px.x, 0)).r;
    float K_B = texture(u_tex_kernel, uv - vec2(0, px.y)).r;
    float K_T = texture(u_tex_kernel, uv + vec2(0, px.y)).r;
    
    // 1. Growth Gradient (grad_U) - using THIS cell's genes
    vec2 grad_U = vec2(
        G(K_R, muStruct, sigma) - G(K_L, muStruct, sigma),
        G(K_T, muStruct, sigma) - G(K_B, muStruct, sigma)
    ) * 0.5;
    
    // 2. Mass Repulsion (grad_M) - CRITICAL for preventing explosions
    float M_L = texture(u_tex_living, uv - vec2(px.x, 0)).r;
    float M_R = texture(u_tex_living, uv + vec2(px.x, 0)).r;
    float M_B = texture(u_tex_living, uv - vec2(0, px.y)).r;
    float M_T = texture(u_tex_living, uv + vec2(0, px.y)).r;
    vec2 grad_M = vec2(M_R - M_L, M_T - M_B) * 0.5;
    
    // 3. Chemotaxis (grad_Food)
    vec2 grad_Food = vec2(0.0);
    if (params.u_chemotaxis > 0.0) {
        vec4 wC = texture(u_tex_waste, uv);
        // Hunger factor: if food is already here, don't desperately chase more
        float hunger = max(0.0, 1.0 - wC.r * 1.5);
        
        if (hunger > 0.0) {
            vec4 wL = texture(u_tex_waste, uv - vec2(px.x, 0));
            vec4 wR = texture(u_tex_waste, uv + vec2(px.x, 0));
            vec4 wB = texture(u_tex_waste, uv - vec2(0, px.y));
            vec4 wT = texture(u_tex_waste, uv + vec2(0, px.y));
            
            // Affinity based on diet matching (prototype logic)
            // float targetType = fract(muDiet + u_dietOffset); // We skip dietOffset for now
            float targetType = muDiet;
            
            float affL = (1.0 - abs(wL.g - targetType)) * wL.r;
            float affR = (1.0 - abs(wR.g - targetType)) * wR.r;
            float affB = (1.0 - abs(wB.g - targetType)) * wB.r;
            float affT = (1.0 - abs(wT.g - targetType)) * wT.r;
            
            grad_Food = vec2(affR - affL, affT - affB) * hunger;
        }
    }
    
    // Combine forces (MATCHES PROTOTYPE EXACTLY)
    // flow = grad_U - grad_M * 2.0 + grad_Food * u_chemotaxis
    vec2 flow = grad_U - grad_M * 2.0 + grad_Food * params.u_chemotaxis;
    
    // CRITICAL: Clamp maximum velocity to prevent explosions
    float maxVel = 4.0;
    if (length(flow) > maxVel) {
        flow = normalize(flow) * maxVel;
    }
    
    imageStore(out_velocity, pos, vec4(flow, 0.0, 1.0));
}
