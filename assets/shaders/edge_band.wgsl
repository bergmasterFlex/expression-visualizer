// A strand between two anchors.
//
// Cut out rather than blended: the material is `AlphaMode::Mask`, so coverage
// decides whether a fragment is drawn at all and never how much of it shows
// through. That is what puts a strand in the depth buffer, which is what lets
// the depth cue lay a shadow under it and draw its silhouette.
//
// `params.opacity` is the one thing that changes that, and only where the level
// of detail has taken something off the strand's scope. `EdgeMaterial` keys
// such a strand `BLEND_ALPHA` instead, which is the only pass Bevy defines
// `OIT_ENABLED` for — so the branch below is live exactly when the opacity is
// not 1, and dead the rest of the time. The coverage cut is untouched either
// way: it is geometry, and geometry does not fade.

#import bevy_pbr::forward_io::VertexOutput
#ifdef OIT_ENABLED
#import bevy_core_pipeline::oit::oit_draw
#endif

// Must match `COVERAGE_CUTOFF` in `src/edge.rs`, which names the same number to
// the material.
const COVERAGE_CUTOFF: f32 = 0.5;

struct EdgeParams {
    // Where the strand leaves, and where it arrives. Equal for every strand
    // whose two ends carry the same type; a cast is where they part.
    band_color_start: vec4<f32>,
    band_color_end: vec4<f32>,
    // Seconds since app start. Nothing reads it yet — it is kept against a
    // coming edge animation, and a uniform slot is cheaper to leave standing
    // than to take out and put back.
    time: f32,
    line_mode_start: f32,
    line_half_thickness: f32,
    line_mode_end: f32,
    height_start: f32,
    height_end: f32,
    arc_total: f32,
    dash_period: f32,
    dash_duty: f32,
    // What survives of the strand at the distance its scope stands from the
    // caret's. 1.0 for everything at full strength, which is also the only
    // value that keeps the masked path.
    opacity: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: EdgeParams;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // How far along the ribbon this fragment sits. uv.x is raw arc length and
    // is constant across the ribbon's height (both vertices of a column carry
    // the same value), so this varies only along the length — exactly linear in
    // the along-curve parameter, with no cross-talk from uv.y. The guard covers
    // a degenerate curve, whose total length is zero.
    let along = clamp(in.uv.x / max(params.arc_total, 1e-6), 0.0, 1.0);

    // Half of what is actually drawn, in `uv.y`: the hairline's half-thickness
    // where the line mode is 1, half the ribbon where it is 0, and a width
    // that grows evenly between them. A strand carrying a value reads as a
    // line, one carrying a type as a band, and a strand that becomes the other
    // opens from one into the other along its whole length.
    //
    // Worked out in world units and divided back into `uv.y`, which is the
    // whole of why the two heights are passed. `uv.y` is a share of the local
    // height and the local height ramps from one end to the other, so a share
    // interpolated on its own would be two ramps multiplied: a strand that
    // opens slowly and finishes fast. Asking for a world width and dividing by
    // the height that is actually there leaves the width linear, which is what
    // the eye follows along a curve whose ends are different shapes.
    //
    // Before either, coverage was `mix(1.0, line_coverage, line_mode)` — which
    // outside the hairline is just `1.0 - line_mode`, above the cut-off for
    // every line mode below a half and below it for every one above. The width
    // did not open at all then, it *switched*, in one step wherever the mix
    // crossed 0.5.
    let mesh_height = mix(params.height_start, params.height_end, along);
    let drawn_start =
        mix(0.5, params.line_half_thickness, params.line_mode_start) * params.height_start;
    let drawn_end =
        mix(0.5, params.line_half_thickness, params.line_mode_end) * params.height_end;
    let half_height = mix(drawn_start, drawn_end, along) / max(mesh_height, 1e-6);
    let dist_from_line = abs(in.uv.y - 0.5);
    // Where the line mode is 0 this lands on a half, and the band keeps its
    // full height: the outermost sample sits exactly on the smoothstep's
    // midpoint, which is exactly the cut-off, which is drawn rather than cut.
    let line_coverage = 1.0 - smoothstep(
        half_height - 0.02,
        half_height + 0.02,
        dist_from_line);

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

    let coverage = line_coverage * dash;

    // Coverage is purely geometric now — it says where the strand is, not how
    // solid it is. Both of the smoothsteps above cross 0.5 exactly on the shape
    // they describe, so cutting there draws the strand at its stated width and
    // a dash at its true flank.
    if coverage < COVERAGE_CUTOFF {
        discard;
    }

    // Along the length and nothing else. `uv.x` is constant across the
    // ribbon's height — both vertices of a column carry the same value — so
    // the ramp is exactly linear in the along-curve parameter with no
    // cross-talk from `uv.y`, the same property the width ramp above relies
    // on. At `along = 0` this is exactly the start colour and at
    // `along = 1` exactly the end one, which is what lets it meet the flat
    // anchor segment at either end without a seam.
    let rgb = mix(params.band_color_start.rgb, params.band_color_end.rgb, along);

    let out_color = vec4<f32>(rgb, params.opacity);

#ifdef OIT_ENABLED
    // Submit fragment to OIT layer buffer, then discard so the regular
    // forward pass doesn't also write blended color. The same hand-off
    // `grid.wgsl` makes, for the same reason: neither shader goes through
    // `pbr_functions.wgsl`, which is where Bevy would otherwise do it.
    oit_draw(in.position, out_color);
    discard;
#endif

    return out_color;
}
