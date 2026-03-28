#import bevy_pbr::forward_io::VertexOutput

struct GridParams {
    plane_color: vec4<f32>,
    line_color: vec4<f32>,
    spacing: f32,
    fade_start: f32,
    fade_end: f32,
    line_thickness: f32,
}

@group(2) @binding(0) var<uniform> params: GridParams;

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
    let color = mix(grid_color, avg_color, density_blend);

    let dist = length(world_pos);
    let fade = 1.0 - saturate((dist - params.fade_start) / (params.fade_end - params.fade_start));

    return vec4<f32>(color.rgb, color.a * fade);
}
