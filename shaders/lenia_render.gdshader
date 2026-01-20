shader_type canvas_item;

uniform sampler2D u_tex_living : hint_default_black;
uniform sampler2D u_tex_waste : hint_default_black;
uniform sampler2D u_tex_species : hint_default_black; // R32F (actually uint)

uniform float u_show_waste : hint_range(0.0, 1.0) = 1.0;
uniform float u_show_species : hint_range(0.0, 1.0) = 0.0;
uniform float u_show_env : hint_range(0.0, 1.0) = 0.0;
// New Env Binding
uniform sampler2D u_tex_env : hint_default_black;
uniform sampler2D u_tex_genes_aux : hint_default_black;



uniform vec4 u_camera = vec4(0.0, 0.0, 1.0, 0.0); // x, y, zoom, unused

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Hash function for UINT ID
vec3 hash3_uint(uint p) {
    p = p * 1103515245u + 12345u;
    p = ((p >> 16u) ^ p) * 0x45d9f3bu;
    p = ((p >> 16u) ^ p) * 0x45d9f3bu;
    p = (p >> 16u) ^ p;
    
    float r = float(p & 0xFFu) / 255.0;
    float g = float((p >> 8u) & 0xFFu) / 255.0;
    float b = float((p >> 16u) & 0xFFu) / 255.0;
    return vec3(r, g, b);
}


void fragment() {
    vec2 uv = UV;
    
    // Camera transform
    uv -= 0.5;
    uv /= u_camera.z;
    uv += u_camera.xy;
    uv += 0.5;
    
    vec4 lData = texture(u_tex_living, uv);
    vec4 wData = texture(u_tex_waste, uv);
    
    float lMass = lData.r;
    float lStruct = lData.g; // HUE
    float lDiet = lData.b;   // SATURATION
    
    float wMass = wData.r;
    float wType = wData.g;
    
    vec3 col = vec3(0.0);
    
    if (u_show_env > 0.5) {
        // --- ENV MODE ---
        vec4 env = texture(u_tex_env, uv);
        col = env.rgb;
        // Tweak visibility: Temp(R), Resource(G), Hazard(B)
        // Adjust for readability
        col.g *= 0.8; 
    } else if (u_show_species > 0.5) {
        
        // --- PHENOTYPE MODE ---
        // Hue = Mu (Structure) -> lData.g
        // Saturation = Aggression -> aux.g
        // Value = Speed -> aux.r
        
        float hue = lData.g; 
        
        vec4 aux = texture(u_tex_genes_aux, uv);
        float speed = aux.r; // 0.5 to 1.5 range typically
        float aggro = aux.g; // 0.0 to 1.0 range
        
        // Saturation: Aggressive = vivid, Passive = pale
        float s = 0.3 + aggro * 0.7;
        
        // Value: Fast = Bright, Slow = Dim
        // Normalize speed (0.5-1.5) to approx (0.6-1.0)
        float v = clamp(speed * 0.6, 0.4, 1.0);
        
        vec3 geneColor = hsv2rgb(vec3(hue, s, v));
        
        // Mask by Mass
        float visibleMass = smoothstep(0.05, 0.2, lMass);
        col = geneColor * visibleMass;


        
    } else {
        // --- STANDARD GENE MODE ---
        float saturation = 0.1 + lDiet * 0.9;
        float visibleMass = 1.0 - exp(-lMass * 1.5);
        vec3 livingCol = hsv2rgb(vec3(lStruct, saturation, visibleMass));
        
        // Visualización de Residuos
        vec3 wasteCol = vec3(0.0);
        if (u_show_waste > 0.5) {
            float visibleWaste = 1.0 - exp(-wMass * 0.8);
            wasteCol = hsv2rgb(vec3(wType, 0.3, visibleWaste * 0.5));
        }
        
        col = livingCol + wasteCol;
    }
    
    col = col / (1.0 + col); // Tone mapping
    col = pow(col, vec3(0.8)); // Gamma
    
    // Darken out-of-bounds
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        col *= 0.1;
    }
    
    COLOR = vec4(col, 1.0);
}
