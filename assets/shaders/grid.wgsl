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
    // World-space (x, z) of the hovered cell's center.
    hover_pos: vec2<f32>,
    hover_active: f32,
    _pad: f32,
    // World-space (x, z) of the outer boundary.
    border_min: vec2<f32>,
    border_max: vec2<f32>,
    border_color: vec4<f32>,
    border_active: f32,
    border_thickness: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: GridParams;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = in.world_position.xz;

    let coord = world_pos / params.spacing;
    // Lines drawn at half-integer coord so integer grid positions
    // (world = int * spacing) sit at cell centers, not crossings.
    let grid = abs(fract(coord) - 0.5);
    let line_width = fwidth(coord);

    let line_x = 1.0 - saturate(grid.x / (line_width.x * params.line_thickness));
    let line_y = 1.0 - saturate(grid.y / (line_width.y * params.line_thickness));
    let lines = max(line_x, line_y);

    let density = max(line_width.x, line_width.y);
    let density_blend = smoothstep(0.2, 0.6, density);

    let grid_color = mix(params.plane_color, params.line_color, lines);
    let avg_color = mix(params.plane_color, params.line_color, 0.3);
    var color = mix(grid_color, avg_color, density_blend);

    // Outer boundary highlight for the current-context AST grid: paint the
    // four mesh-edge grid lines in border_color instead of line_color.
    if params.border_active > 0.5 {
        let d_x = min(
            abs(world_pos.x - params.border_min.x),
            abs(world_pos.x - params.border_max.x),
        );
        let d_y = min(
            abs(world_pos.y - params.border_min.y),
            abs(world_pos.y - params.border_max.y),
        );
        let bw = fwidth(world_pos) * params.line_thickness * params.border_thickness;
        let on_border = max(
            1.0 - saturate(d_x / bw.x),
            1.0 - saturate(d_y / bw.y),
        );
        color = vec4<f32>(
            mix(color.rgb, params.border_color.rgb, on_border),
            max(color.a, on_border * params.border_color.a),
        );
    }

    // Hover feedback: fill the hovered cell (side = spacing) with a soft
    // wash, anti-aliased against fwidth so it stays crisp at oblique angles.
    if params.hover_active > 0.5 {
        let d = abs(world_pos - params.hover_pos);
        let half_cell = params.spacing * 0.5;
        let aa = max(fwidth(world_pos.x), fwidth(world_pos.y));
        let mask_x = 1.0 - smoothstep(half_cell - aa, half_cell + aa, d.x);
        let mask_y = 1.0 - smoothstep(half_cell - aa, half_cell + aa, d.y);
        let cell = mask_x * mask_y;
        let fill_alpha = cell * 0.28;
        color = vec4<f32>(
            mix(color.rgb, params.line_color.rgb, fill_alpha),
            color.a + fill_alpha,
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
