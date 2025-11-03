shader_type canvas_item;

void fragment() {
    vec4 tex_color = texture(TEXTURE, UV);
    // Balik warna
    tex_color.rgb = vec3(1.0) - tex_color.rgb;
    COLOR = tex_color;
}
