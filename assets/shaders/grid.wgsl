#import bevy_pbr::forward_io::VertexOutput

@group(2) @binding(0) var<uniform> plane_color: vec4<f32>;
@group(2) @binding(1) var<uniform> line_color: vec4<f32>;
@group(2) @binding(2) var<uniform> spacing: f32;
@group(2) @binding(3) var<uniform> fade_start: f32;
@group(2) @binding(4) var<uniform> fade_end: f32;
@group(2) @binding(5) var<uniform> line_thickness: f32;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = in.world_position.xz;

    let coord = world_pos / spacing;
    let grid = abs(fract(coord - 0.5) - 0.5);
    let line_width = fwidth(coord);

    let line_x = 1.0 - saturate(grid.x / (line_width.x * line_thickness));
    let line_y = 1.0 - saturate(grid.y / (line_width.y * line_thickness));
    let lines = max(line_x, line_y);

    // As density increases, blend toward a uniform average of plane + line color
    let density = max(line_width.x, line_width.y);
    let density_blend = smoothstep(0.2, 0.6, density);

    // Sharp grid near camera, smooth average at distance
    let grid_color = mix(plane_color, line_color, lines);
    let avg_color = mix(plane_color, line_color, 0.3); // approximate line coverage ratio
    let color = mix(grid_color, avg_color, density_blend);

    // Distance fade
    let dist = length(world_pos);
    let fade = 1.0 - saturate((dist - fade_start) / (fade_end - fade_start));

    return vec4<f32>(color.rgb, color.a * fade);
}
