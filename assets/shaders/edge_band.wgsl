#import bevy_pbr::forward_io::VertexOutput

#ifdef OIT_ENABLED
#import bevy_core_pipeline::oit::oit_draw
#endif

struct EdgeParams {
    band_color: vec4<f32>,
    // Seconds since app start. Nothing reads it yet — it is kept against a
    // coming edge animation, and a uniform slot is cheaper to leave standing
    // than to take out and put back.
    time: f32,
    line_mode_start: f32,
    line_half_thickness: f32,
    line_mode_end: f32,
    arc_total: f32,
    dash_period: f32,
    dash_duty: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: EdgeParams;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Hairline coverage: a soft indicator around uv.y = 0.5, so a strand
    // carrying a value reads as a line rather than as a band. Its world
    // thickness is `line_half_thickness` of the ribbon's own height, which is
    // why a hairline ribbon is built short (see `RIBBON_LINE_HEIGHT`).
    let dist_from_line = abs(in.uv.y - 0.5);
    let line_coverage = 1.0 - smoothstep(
        params.line_half_thickness - 0.02,
        params.line_half_thickness + 0.02,
        dist_from_line);

    // How far along the ribbon this fragment sits. uv.x is raw arc length and
    // is constant across the ribbon's height (both vertices of a column carry
    // the same value), so this varies only along the length — exactly linear in
    // the along-curve parameter, with no cross-talk from uv.y. The guard covers
    // a degenerate curve, whose total length is zero.
    let along = clamp(in.uv.x / max(params.arc_total, 1e-6), 0.0, 1.0);
    let line_mode = mix(params.line_mode_start, params.line_mode_end, along);
    // Vertical cut-outs across the band. `uv.x` is raw arc length in world
    // units, so the period is a world length and not a share of the edge: a
    // long edge gets more dashes, never longer ones. `0.0` leaves the band
    // whole, which is what everything but a pending edge asks for.
    var dash = 1.0;
    if params.dash_period > 0.0 {
        let phase = fract(in.uv.x / params.dash_period);
        // The derivative is taken on the unfolded coordinate: `fract` jumps at
        // the end of every period and `uv.x` does not. Same precaution
        // `grid.wgsl` takes for its lines.
        let aa = max(fwidth(in.uv.x) / params.dash_period, 1e-5);
        // The dash is centred on `phase = 0.5`, so both of its flanks fall
        // inside the period and neither is cut off at the wrap.
        let edge_dist = params.dash_duty * 0.5 - abs(phase - 0.5);
        dash = smoothstep(-aa, aa, edge_dist);
    }

    let coverage = mix(1.0, line_coverage, line_mode) * dash;

    let out_color = vec4<f32>(params.band_color.rgb, coverage * params.band_color.a);

#ifdef OIT_ENABLED
    oit_draw(in.position, out_color);
    discard;
#endif

    return out_color;
}
