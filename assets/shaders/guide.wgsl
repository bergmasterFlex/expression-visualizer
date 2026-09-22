// A dashed guide running out from a caret to the face of its volume.
//
// Cut out rather than blended, the reading `edge_band.wgsl` takes for a
// hairline: coverage says whether a fragment is drawn at all and never how much
// of it shows through. That is what puts a guide in the depth buffer, which is
// what lets whatever stands in front of it cut it off — and where a guide
// breaks is itself worth seeing, because it says something passes in front.
//
// No hand-off to `oit_draw` here, unlike the grid and the strands. Bevy defines
// `OIT_ENABLED` only for the blended pass, and `GuideMaterial` is
// `AlphaMode::Mask` and never anything else, so such a branch would be dead
// every frame. Should a guide ever fade with the level of detail the way
// everything else does, it would need adding back along with the alpha.

#import bevy_pbr::forward_io::VertexOutput

// Must match `COVERAGE_CUTOFF` in `src/guide.rs`, which names the same number
// to key the material's alpha mode with.
const COVERAGE_CUTOFF: f32 = 0.5;

struct GuideParams {
    color: vec4<f32>,
    // The world axis the dashes are counted along. `w` is unused padding.
    //
    // Measured from the world origin rather than from the caret the guide
    // leaves, and that is deliberate: with a period that divides the cell,
    // every guide in the scene then breaks on the same boundaries, and a dash
    // is a fact about the grid instead of a fact about where a line started.
    // Two guides crossing meet at a dash rather than at whatever phase each
    // happened to carry.
    axis: vec4<f32>,
    // World length of one dash-plus-gap, and the share of it that is drawn.
    dash_period: f32,
    dash_duty: f32,
    _pad: vec2<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: GuideParams;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // How far along its own axis this fragment sits. The line is thin on the
    // other two, so this is the coordinate that varies over its length and the
    // only one the pattern can be read off.
    let along = dot(in.world_position.xyz, params.axis.xyz);
    let phase = fract(along / params.dash_period);
    // The derivative is taken on the unfolded coordinate: `fract` jumps at the
    // end of every period and `along` does not. The same precaution
    // `edge_band.wgsl` and `grid.wgsl` both take.
    let aa = max(fwidth(along) / params.dash_period, 1e-5);
    // The dash is centred on `phase = 0.5`, so both of its flanks fall inside
    // the period and neither is cut off at the wrap.
    let edge_dist = params.dash_duty * 0.5 - abs(phase - 0.5);
    let coverage = smoothstep(-aa, aa, edge_dist);

    // The smoothstep crosses 0.5 exactly on the flank it describes, so cutting
    // there draws each dash at its true length.
    if coverage < COVERAGE_CUTOFF {
        discard;
    }

    return params.color;
}
