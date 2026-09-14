// A strand between two anchors.
//
// Cut out rather than blended: the material is `AlphaMode::Mask`, so coverage
// decides whether a fragment is drawn at all and never how much of it shows
// through. That is what puts a strand in the depth buffer, which is what lets
// the depth cue lay a shadow under it and draw its silhouette.
//
// There is no OIT path here any more. Bevy only defines `OIT_ENABLED` for a
// pass keyed `BLEND_ALPHA`, so for a masked material the branch could never be
// taken.

#import bevy_pbr::forward_io::VertexOutput

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

    // Coverage is purely geometric now — it says where the strand is, not how
    // solid it is. Both of the smoothsteps above cross 0.5 exactly on the shape
    // they describe, so cutting there draws the hairline at its stated
    // thickness and a dash at its true flank.
    if coverage < COVERAGE_CUTOFF {
        discard;
    }

    // Along the length and nothing else. `uv.x` is constant across the
    // ribbon's height — both vertices of a column carry the same value — so
    // the ramp is exactly linear in the along-curve parameter with no
    // cross-talk from `uv.y`, the same property the `line_mode` mix above
    // relies on. At `along = 0` this is exactly the start colour and at
    // `along = 1` exactly the end one, which is what lets it meet the flat
    // anchor segment at either end without a seam.
    let rgb = mix(params.band_color_start.rgb, params.band_color_end.rgb, along);

    return vec4<f32>(rgb, 1.0);
}
