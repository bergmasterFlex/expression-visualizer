#import bevy_pbr::forward_io::VertexOutput

#ifdef OIT_ENABLED
#import bevy_core_pipeline::oit::oit_draw
#endif

// Must match `MAX_FOOTPRINTS` in `src/grid.rs`.
const MAX_FOOTPRINTS: u32 = 16u;

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
    footprint_count: u32,
    _pad2: f32,
    // World-space footprints of multi-cell nodes: `xy = min.xz`,
    // `zw = max.xz`. Interior grid lines are suppressed inside these.
    footprints: array<vec4<f32>, 16>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: GridParams;

// Returns 1.0 iff both `a` and `b` lie inside the same footprint (world
// XZ coords). Used to suppress the grid line that separates them.
fn line_between_inside_same_footprint(a: vec2<f32>, b: vec2<f32>) -> f32 {
    for (var i: u32 = 0u; i < params.footprint_count; i = i + 1u) {
        let fp = params.footprints[i];
        let a_in = all(a >= fp.xy) && all(a <= fp.zw);
        let b_in = all(b >= fp.xy) && all(b <= fp.zw);
        if a_in && b_in {
            return 1.0;
        }
    }
    return 0.0;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = in.world_position.xz;

    let coord = world_pos / params.spacing;
    // Lines drawn at half-integer coord so integer grid positions
    // (world = int * spacing) sit at cell centers, not crossings.
    let grid = abs(fract(coord) - 0.5);
    let line_width = fwidth(coord);

    var line_x = 1.0 - saturate(grid.x / (line_width.x * params.line_thickness));
    var line_y = 1.0 - saturate(grid.y / (line_width.y * params.line_thickness));

    // Suppress interior grid lines of multi-cell node footprints. The nearest
    // line along each axis separates the current cell (round(coord)) from
    // its neighbour on the same side as the fragment (`sign(coord - cell)`).
    // If both cells are inside the same footprint we skip that line.
    let cell = round(coord);
    let side = sign(coord - cell);
    let curr_center = cell * params.spacing;
    let x_neighbour = vec2<f32>(cell.x + side.x, cell.y) * params.spacing;
    let y_neighbour = vec2<f32>(cell.x, cell.y + side.y) * params.spacing;
    let suppress_x = line_between_inside_same_footprint(curr_center, x_neighbour);
    let suppress_y = line_between_inside_same_footprint(curr_center, y_neighbour);
    line_x = line_x * (1.0 - suppress_x);
    line_y = line_y * (1.0 - suppress_y);
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
