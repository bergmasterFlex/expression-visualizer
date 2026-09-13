// Depth cue: a full-screen pass that recovers depth from the depth buffer.
//
// Two effects off one buffer, because they answer two different halves of the
// same question:
//
// - **Depth darkening** (Luft/Colditz/Deussen, SIGGRAPH 2006). Compare each
//   pixel's distance against a blurred copy of the same. Where a pixel sits
//   further away than its surroundings, it is the far side of a silhouette, and
//   darkening it there lays a soft shadow behind the nearer thing. This is the
//   part that says *which of two overlapping nodes is in front*.
// - **A silhouette line** off the same distances. Flat unlit boxes have no
//   shading to read a shape from, and a hard contour at a depth step is worth
//   more on them than any amount of soft halo.
//
// Neither touches hue: both come out as a multiplier on whatever the main pass
// painted, so a type's colour still means exactly what it meant. That is the
// reason the cue lives here at all rather than in the lighting — lighting would
// have made the same colour read differently per face orientation.
//
// The depth read is the *main* depth texture. Order-independent transparency
// already asks for `TEXTURE_BINDING` on it, so no prepass and no second pass
// over the geometry is needed — Bevy's own OIT resolve reads it the same way.
// Only opaque geometry writes depth, which is the right set: the node bodies
// are what have to be told apart, while edges, type markers, caret faces and
// the grid blend and stay out of it.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct DepthCue {
    // The bound projection's clip planes, for turning stored depth back into
    // world units. `near` is normally negative — the projection box reaches
    // behind the camera deliberately.
    near: f32,
    far: f32,
    // Reach of the halo, in pixels.
    halo_radius: f32,
    // How dark the halo goes at full separation, 0..1.
    halo_strength: f32,
    // Separation, in cells, at which the halo reaches `halo_strength`. Past
    // this the darkening stops growing, which is what keeps the background —
    // hundreds of units behind everything — from saturating every silhouette.
    halo_depth: f32,
    // Depth step, in cells, that counts as a silhouette rather than as a face
    // leaning away from the viewer.
    edge_threshold: f32,
    // How dark the silhouette line goes, 0..1.
    edge_strength: f32,
    mode: u32,
}

const MODE_OFF: u32 = 0u;
const MODE_CUE: u32 = 1u;
const MODE_DEPTH: u32 = 2u;

// Sample pattern for the blur: rings of spokes around the centre. A separable
// two-pass Gaussian would be cheaper per unit of radius, but it needs an
// intermediate target and a second node; at this radius the ring pattern costs
// 25 loads and buys the same soft falloff without either.
const RING_COUNT: u32 = 3u;
const SPOKE_COUNT: u32 = 8u;
const TAU: f32 = 6.283185307;
// Golden angle, used to twist each ring against the last so the spokes don't
// line up into a star.
const GOLDEN_ANGLE: f32 = 2.399963;

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var screen_sampler: sampler;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
@group(0) @binding(3) var<uniform> cue: DepthCue;

/// World-space distance from the camera at a stored depth value.
///
/// Two things make this a plain lerp rather than the usual reciprocal. Bevy
/// projects reverse-Z, so the stored value runs 1.0 at `near` down to 0.0 at
/// `far` — hence `1.0 - d`. And the bound projection is parallel, so depth is
/// linear across that span with no perspective divide to undo. The camera's
/// depth shear leaves this alone: it slides X and Y by Z and leaves Z itself.
///
/// Pixels no geometry covered hold the cleared 0.0 and so read as `far`. That
/// is not a special case to strip out — the background genuinely is behind
/// everything, and both effects below want it to say so.
fn view_distance(d: f32) -> f32 {
    return cue.near + (1.0 - d) * (cue.far - cue.near);
}

/// Distance at a pixel, with the lookup held inside the texture.
///
/// Clamping rather than wrapping matters at the frame's border: a wrapped
/// sample would pair the top of the picture with the bottom and draw a
/// silhouette along an edge where there is no geometry at all.
fn distance_at(coord: vec2<i32>, size: vec2<i32>) -> f32 {
    let held = clamp(coord, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
    return view_distance(textureLoad(depth_texture, held, 0));
}

/// The local average distance — the low-pass half of the unsharp mask.
fn blurred_distance(coord: vec2<i32>, size: vec2<i32>) -> f32 {
    var total = distance_at(coord, size);
    var weight = 1.0;

    // Sigma at half the radius puts the outermost ring around two sigma, where
    // a Gaussian has all but died out. Rings beyond that would cost taps and
    // contribute nothing.
    let sigma = max(cue.halo_radius * 0.5, 0.001);

    for (var ring = 1u; ring <= RING_COUNT; ring = ring + 1u) {
        let radius = cue.halo_radius * f32(ring) / f32(RING_COUNT);
        let w = exp(-0.5 * radius * radius / (sigma * sigma));
        let phase = f32(ring) * GOLDEN_ANGLE;

        for (var spoke = 0u; spoke < SPOKE_COUNT; spoke = spoke + 1u) {
            let angle = phase + f32(spoke) * TAU / f32(SPOKE_COUNT);
            let offset = vec2<f32>(cos(angle), sin(angle)) * radius;
            total = total + distance_at(coord + vec2<i32>(round(offset)), size) * w;
            weight = weight + w;
        }
    }

    return total / weight;
}

/// How much this pixel looks like it sits on a depth step, 0..1.
///
/// The four-neighbourhood is enough: a step is a step, and it shows on at least
/// one axis. The threshold is what separates it from a face leaning away from
/// the viewer — those are everywhere here, since the projection is oblique, but
/// they change distance by a fraction of a cell per pixel, far under the step a
/// real silhouette makes.
fn silhouette(coord: vec2<i32>, size: vec2<i32>, centre: f32) -> f32 {
    var worst = 0.0;
    worst = max(worst, abs(distance_at(coord + vec2<i32>(1, 0), size) - centre));
    worst = max(worst, abs(distance_at(coord + vec2<i32>(-1, 0), size) - centre));
    worst = max(worst, abs(distance_at(coord + vec2<i32>(0, 1), size) - centre));
    worst = max(worst, abs(distance_at(coord + vec2<i32>(0, -1), size) - centre));
    return smoothstep(cue.edge_threshold, cue.edge_threshold * 2.0, worst);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(screen_texture, screen_sampler, in.uv);

    if cue.mode == MODE_OFF {
        return color;
    }

    let size = vec2<i32>(textureDimensions(depth_texture));
    let coord = vec2<i32>(in.position.xy);
    let dist = distance_at(coord, size);

    // Contour bands, one per world unit — which is one grid cell, since `CELL`
    // is 1.0 and `LAYOUT_SCALE` only flips signs. This is what the spike drew,
    // kept as a way to see the depth the cue is reading rather than the cue.
    if cue.mode == MODE_DEPTH {
        let band = fract(dist);
        return vec4<f32>(band, band, band, 1.0);
    }

    // Positive where this pixel is further away than its surroundings: the
    // background at a near object's rim, and the further of two overlapping
    // nodes. Negative on the near side, which is clamped off — a nearer pixel
    // is not what the cue is about, and brightening it would fight the type
    // colours for attention.
    let behind = max(dist - blurred_distance(coord, size), 0.0);
    let halo = 1.0 - cue.halo_strength * saturate(behind / max(cue.halo_depth, 0.001));

    let edge = 1.0 - cue.edge_strength * silhouette(coord, size, dist);

    // Multiplicative, and in linear HDR because the pass sits before the
    // tonemapper: the same factor darkens a bright node and a dim one by the
    // same proportion. Alpha is passed through untouched — this writes to the
    // view target, not into a blend.
    return vec4<f32>(color.rgb * halo * edge, color.a);
}
