shader_type canvas_item;

uniform sampler2D u_tex_living : hint_default_black;
uniform sampler2D u_tex_waste : hint_default_black;
uniform float u_show_waste : hint_range(0.0, 1.0) = 1.0;
uniform vec4 u_camera = vec4(0.0, 0.0, 1.0, 0.0); // x, y, zoom, unused

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
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
    
    // Visualización de Vida
    float saturation = 0.1 + lDiet * 0.9;
    float visibleMass = 1.0 - exp(-lMass * 1.5);
    vec3 livingCol = hsv2rgb(vec3(lStruct, saturation, visibleMass));
    
    // Visualización de Residuos
    vec3 wasteCol = vec3(0.0);
    if (u_show_waste > 0.5) {
        float visibleWaste = 1.0 - exp(-wMass * 0.8);
        wasteCol = hsv2rgb(vec3(wType, 0.3, visibleWaste * 0.5));
    }
    
    vec3 col = livingCol + wasteCol;
    col = col / (1.0 + col); // Tone mapping
    col = pow(col, vec3(0.8)); // Gamma
    
    // Darken out-of-bounds
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        col *= 0.1;
    }
    
    COLOR = vec4(col, 1.0);
}
