#import bevy_pbr::forward_io::VertexOutput

#ifdef OIT_ENABLED
#import bevy_core_pipeline::oit::oit_draw
#endif

struct EdgeParams {
    band_color: vec4<f32>,
    letter_color: vec4<f32>,
    scroll_speed: f32,
    tile_length: f32,
    time: f32,
    line_mode: f32,
    line_half_thickness: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: EdgeParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var label_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var label_smp: sampler;

/// OR together several vertical taps at the same u so glyphs whose ink
/// doesn't happen to cross the line row (digits with high crossbars, thin
/// ascenders, descenders) still count as "letter present in this column".
fn column_ink(u: f32) -> f32 {
    let a = textureSample(label_tex, label_smp, vec2<f32>(u, 0.15)).r;
    let b = textureSample(label_tex, label_smp, vec2<f32>(u, 0.35)).r;
    let c = textureSample(label_tex, label_smp, vec2<f32>(u, 0.55)).r;
    let d = textureSample(label_tex, label_smp, vec2<f32>(u, 0.75)).r;
    return max(max(a, b), max(c, d));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // uv.x carries arc-length in world units (see build_ribbon_mesh); dividing
    // by tile_length gives a repeat coord that stays scale-consistent no matter
    // how long the ribbon is. Subtracting time*speed makes the marquee flow
    // from output → input (u decreases with time, so the texture appears to
    // move in +u direction, i.e. toward the target anchor).
    let u = in.uv.x / params.tile_length
          - params.time * params.scroll_speed / params.tile_length;
    let letter_alpha = textureSample(label_tex, label_smp, vec2<f32>(u, in.uv.y)).r;

    // Column mask, horizontally dilated by ±du in u-space. du is chosen
    // wider than the inter-letter tracking of the marquee font so adjacent
    // glyphs' masks overlap — the hairline stays hidden all the way through
    // each word and only reappears where the source text has real
    // whitespace (the "  " padding between value and type name, and around
    // the outside of the marquee). Extra taps at ±du/2 fill in the middle
    // so narrow letters like "i" don't leave a gap that pokes through.
    let du = 0.04;
    let column_mask = max(
        max(column_ink(u - du), column_ink(u - du * 0.5)),
        max(max(column_ink(u), column_ink(u + du * 0.5)), column_ink(u + du)));

    // Hairline coverage: soft indicator around uv.y = 0.5, cut wherever any
    // glyph ink lies in the same column — so the line vanishes under each
    // character and reappears cleanly in the spaces between words.
    let dist_from_line = abs(in.uv.y - 0.5);
    let line_edge = 1.0 - smoothstep(
        params.line_half_thickness - 0.02,
        params.line_half_thickness + 0.02,
        dist_from_line);
    let line_coverage = line_edge * (1.0 - column_mask);

    let coverage = mix(1.0, line_coverage, params.line_mode);

    let rgb = mix(params.band_color.rgb, params.letter_color.rgb, letter_alpha);
    let a = max(coverage * params.band_color.a, letter_alpha * params.letter_color.a);
    let out_color = vec4<f32>(rgb, a);

#ifdef OIT_ENABLED
    oit_draw(in.position, out_color);
    discard;
#endif

    return out_color;
}
