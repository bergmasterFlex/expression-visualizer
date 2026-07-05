#import bevy_pbr::forward_io::VertexOutput

#ifdef OIT_ENABLED
#import bevy_core_pipeline::oit::oit_draw
#endif

struct GridParams {
    plane_color: vec4<f32>,
    line_color: vec4<f32>,
    spacing: f32,
    fade_start: f32,
    fade_end: f32,
    line_thickness: f32,
    hover_pos: vec2<f32>,
    hover_active: f32,
    _pad: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: GridParams;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = in.world_position.xz;

    let coord = world_pos / params.spacing;
    let grid = abs(fract(coord - 0.5) - 0.5);
    let line_width = fwidth(coord);

    let line_x = 1.0 - saturate(grid.x / (line_width.x * params.line_thickness));
    let line_y = 1.0 - saturate(grid.y / (line_width.y * params.line_thickness));
    let lines = max(line_x, line_y);

    let density = max(line_width.x, line_width.y);
    let density_blend = smoothstep(0.2, 0.6, density);

    let grid_color = mix(params.plane_color, params.line_color, lines);
    let avg_color = mix(params.plane_color, params.line_color, 0.3);
    var color = mix(grid_color, avg_color, density_blend);

    // Hover feedback: locally brighten grid lines around the hovered
    // crossing and add a faint disc on the plane.
    if params.hover_active > 0.5 {
        let hover_d = length(world_pos - params.hover_pos);
        // Line brightening — falls off over ~1.5 world units.
        let line_boost = (1.0 - smoothstep(0.0, 1.5, hover_d)) * lines * 0.9;
        color = vec4<f32>(color.rgb + params.line_color.rgb * line_boost, color.a);
        // Faint disc — visible within ~0.6 world units.
        let disc = 1.0 - smoothstep(0.0, 0.6, hover_d);
        let disc_alpha = disc * 0.25;
        color = vec4<f32>(
            mix(color.rgb, params.line_color.rgb, disc_alpha),
            color.a + disc_alpha,
        );
    }

    let dist = length(world_pos);
    let fade = 1.0 - saturate((dist - params.fade_start) / (params.fade_end - params.fade_start));

    let out_color = vec4<f32>(color.rgb, color.a * fade);

#ifdef OIT_ENABLED
    // Submit fragment to OIT layer buffer, then discard so the regular
    // forward pass doesn't also write blended color.
    oit_draw(in.position, out_color);
    discard;
#endif

    return out_color;
}
