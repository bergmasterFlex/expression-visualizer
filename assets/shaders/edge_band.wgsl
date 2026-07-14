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
    _pad: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: EdgeParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var label_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var label_smp: sampler;

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

    let rgb = mix(params.band_color.rgb, params.letter_color.rgb, letter_alpha);
    let a = max(params.band_color.a, letter_alpha * params.letter_color.a);
    let out_color = vec4<f32>(rgb, a);

#ifdef OIT_ENABLED
    oit_draw(in.position, out_color);
    discard;
#endif

    return out_color;
}
