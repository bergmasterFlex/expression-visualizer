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
    // The depth-carrying entries of `clip_from_view`: the projection sends a
    // view-space z to `clip.z = a*z + b`, `clip.w = c*z + d`, and the buffer
    // holds their quotient. Four numbers, and they invert any projection —
    // parallel, converging, or the blend of the two.
    clip_z_scale: f32,
    clip_z_offset: f32,
    clip_w_scale: f32,
    clip_w_offset: f32,
    // Pixels one cell covers. Every length below is in cells; this is what
    // brings them into the pixels this pass works in.
    cell_pixels: f32,
    // Reach of the halo, in cells.
    halo_radius: f32,
    // How dark the halo goes at full separation, 0..1.
    halo_strength: f32,
    // Separation, in cells, at which the halo reaches `halo_strength`. Past
    // this the darkening stops growing, which is what keeps the background —
    // hundreds of units behind everything — from saturating every silhouette.
    halo_depth: f32,
    // Reach of the dark edge, in cells. Much tighter than the halo — this is
    // the rim, not the shadow.
    edge_radius: f32,
    // How dark the edge goes at full separation, 0..1.
    edge_strength: f32,
    // Separation, in cells, at which the edge reaches `edge_strength`. Small,
    // so that even a shallow step still gets a readable rim.
    edge_depth: f32,
    mode: u32,
}

const MODE_OFF: u32 = 0u;
const MODE_CUE: u32 = 1u;

// Sample pattern for the blur: rings of spokes around the centre. A separable
// two-pass Gaussian would be cheaper per unit of radius, but it needs an
// intermediate target and a second node; at this radius the ring pattern costs
// 25 loads and buys the same soft falloff without either.
const HALO_RINGS: u32 = 3u;
/// One ring is enough at the edge's reach: a couple of pixels out, a second
/// would land on the pixels the first already covered.
const EDGE_RINGS: u32 = 1u;
/// Spokes on the innermost ring; each ring further out gets a multiple of it,
/// so the disc is covered evenly by area. Three rings therefore cost 8 + 16 +
/// 24 taps — enough that no single neighbour holds more than about a
/// twentieth of the result, which is the difference between a smooth falloff
/// and a visible contour step.
const SPOKES_PER_RING: u32 = 8u;
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
/// Inverting `d = (a*z + b) / (c*z + d)` for z, then negating because view
/// space looks down its own −Z. One rational expression covers every
/// projection the bound camera can be in, which is the point: a parallel one
/// leaves `c` at zero and the expression collapses to a lerp on its own, while
/// a converging one puts the reciprocal back exactly where it belongs.
///
/// Pixels no geometry covered hold the cleared 0.0 and so read as the far
/// plane. That is not a special case to strip out — the background genuinely
/// is behind everything, and both effects below want it to say so.
fn view_distance(raw: f32) -> f32 {
    // Nudged off the far plane before anything else, and that one `max` is
    // load-bearing. A converging projection is built infinite-far, which puts
    // `clip_z_scale` at zero and `clip_w_scale` at minus one — so the
    // denominator below is just `-d`, and the cleared background sits at
    // exactly zero. Dividing there does not merely overflow: approached from
    // above the quotient runs to plus infinity, and any guard that patches the
    // denominator without keeping its sign lands on minus infinity instead.
    // The background then reads as the nearest thing in the frame, `behind`
    // fires on the node instead of on what is behind it, and the whole cue
    // turns itself inside out. Stepping the depth just off the plane keeps the
    // sign and costs nothing anywhere else — a parallel projection has a
    // constant non-zero denominator and does not notice.
    let d = max(raw, 1e-6);
    let numerator = cue.clip_z_offset - d * cue.clip_w_offset;
    let denominator = d * cue.clip_w_scale - cue.clip_z_scale;
    return -numerator / denominator;
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

/// How far this pixel should be darkened for lying behind its surroundings,
/// 0..1 — how much of a disc of radius `radius_cells` around it is nearer than
/// it is.
///
/// This is the whole of the cue, and both terms in the pass are it at two
/// scales: a broad one for the shadow a nearer thing casts, a tight one for the
/// dark rim at its edge.
///
/// **Each neighbour is capped before it is averaged, not after**, and that
/// ordering is the difference between a gradient and a stencil. Averaging the
/// raw distances first and capping the result sounds equivalent and is not:
/// the background sits effectively at infinity, so a single tap landing on
/// nearer geometry drags the average past any cap by itself. The whole cue
/// then collapses to *did any tap hit something* — full strength or nothing,
/// with the lit region coming out as the geometry dilated by the two dozen
/// integer tap offsets, which is a polygon with visible stair steps. Capping
/// per tap bounds every neighbour's say at one `depth_cells`, so what comes
/// out is the fraction of the disc that is nearer, and that fraction falls off
/// smoothly as the pixel moves away from the edge.
///
/// Two properties are worth naming, because a thresholded edge test had to
/// arrange for both. It is **one-sided**: the sum only survives `max` where
/// the neighbourhood is nearer, so the darkening falls outside the silhouette
/// and never onto the nearer thing. And it **ignores a slope**: on a face
/// leaning away from the viewer the nearer half and the further half cancel
/// term for term, so a plain ramp comes out at nothing and only a real step
/// registers.
fn darkening(
    coord: vec2<i32>,
    size: vec2<i32>,
    centre: f32,
    radius_cells: f32,
    strength: f32,
    depth_cells: f32,
    rings: u32,
) -> f32 {
    let wanted = radius_cells * cue.cell_pixels;
    // Below a pixel there is nothing left to shrink, so it fades instead.
    // Holding it at one pixel would make the cue heavier in proportion the
    // further the camera pulls back, which is the opposite of the point.
    let reach = max(wanted, 1.0);
    let fade = saturate(wanted);

    // Sigma at half the reach puts the outermost ring around two sigma, where
    // a Gaussian has all but died out.
    let sigma = reach * 0.5;
    let scale = 1.0 / max(depth_cells, 0.001);

    var total = 0.0;
    var weight = 0.0;

    for (var ring = 1u; ring <= rings; ring = ring + 1u) {
        let radius = reach * f32(ring) / f32(rings);
        let w = exp(-0.5 * radius * radius / (sigma * sigma));
        // Spokes in proportion to the ring, so the disc is sampled evenly by
        // area rather than crowding the taps into the middle. It also keeps
        // any single tap's share of the total small, which is what stops a
        // neighbour crossing the silhouette from showing up as a step.
        let spokes = SPOKES_PER_RING * ring;
        let phase = f32(ring) * GOLDEN_ANGLE;

        for (var spoke = 0u; spoke < spokes; spoke = spoke + 1u) {
            let angle = phase + f32(spoke) * TAU / f32(spokes);
            let offset = vec2<f32>(cos(angle), sin(angle)) * radius;
            let neighbour = distance_at(coord + vec2<i32>(round(offset)), size);
            // Signed, so a nearer neighbour and a further one cancel on a
            // slope; capped, so neither can speak louder than one `depth_cells`.
            total = total + w * clamp((centre - neighbour) * scale, -1.0, 1.0);
            weight = weight + w;
        }
    }

    // A straight silhouette puts half the disc on the nearer side, so doubling
    // is what makes `strength` mean the darkening at an ordinary edge rather
    // than at an impossible one.
    let occlusion = saturate(2.0 * max(total / weight, 0.0));
    return fade * strength * occlusion;
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

    // The same mask at two scales. The broad one is the shadow a nearer thing
    // casts across what lies behind it; the tight one is the dark edge at its
    // rim, which is what gives an unlit box a readable outline. Both fall away
    // outward on their own, and both land outside the silhouette rather than
    // on the nearer thing.
    let halo = darkening(
        coord, size, dist,
        cue.halo_radius, cue.halo_strength, cue.halo_depth, HALO_RINGS,
    );
    let edge = darkening(
        coord, size, dist,
        cue.edge_radius, cue.edge_strength, cue.edge_depth, EDGE_RINGS,
    );

    // Multiplied rather than added, so the two compound where they overlap and
    // neither can drive the colour negative. In linear HDR, because the pass
    // sits before the tonemapper: the same factor darkens a bright node and a
    // dim one by the same proportion. Alpha is passed through untouched — this
    // writes to the view target, not into a blend.
    return vec4<f32>(color.rgb * (1.0 - halo) * (1.0 - edge), color.a);
}
