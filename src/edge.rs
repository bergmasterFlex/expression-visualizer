use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use bevy::asset::{load_internal_asset, uuid_handle};
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::*;
use bevy::shader::ShaderRef;

use crate::infer::EType;

/// Height of a solid edge band: one anchor row, so band and type marker line
/// up exactly.
pub const RIBBON_HEIGHT: f32 = crate::render::CELL;
/// Ribbon height used when a source anchor carries an graph literal. The band
/// itself is invisible in that mode (see `edge_band.wgsl`); the height is
/// kept as-is so the rasterised marquee glyphs stay readable.
pub const RIBBON_LINE_HEIGHT: f32 = crate::render::CELL / 4.0;
/// Half-thickness of the value-edge hairline in `uv.y` space (i.e. as a
/// fraction of `RIBBON_LINE_HEIGHT`). Thin like a grid line.
pub const RIBBON_LINE_HALF_THICKNESS_UV: f32 = 0.1;
pub const RIBBON_SEGMENTS: usize = 40;
const LABEL_TEX_WIDTH: u32 = 256;
const LABEL_TEX_HEIGHT: u32 = 32;

/// Bytes of the bundled JetBrainsMono TTF. Exposed so callers that need to
/// rasterise marquee text at spawn time (see `rasterize_marquee_text`) don't
/// have to `include_bytes!` the same path themselves.
pub const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf");

/// Straight cubic Bézier between two anchor world positions. Both tangents
/// leave along the Z flow axis: `p1` sits in −Z from the source (outputs
/// point in −Z), `p2` sits in +Z from the target (inputs receive from +Z).
pub struct EdgeCurve {
    pub p0: Vec3,
    pub p1: Vec3,
    pub p2: Vec3,
    pub p3: Vec3,
}

impl EdgeCurve {
    pub fn from_endpoints(from_world: Vec3, to_world: Vec3) -> Self {
        let dz = to_world.z - from_world.z;
        // Clamp the handle length so same-Z-rank connections still bulge
        // visibly instead of degenerating into a straight line.
        let mut l = (crate::render::CELL * 0.5)
            .max(0.5 * dz.abs() + 0.25 * (to_world - from_world).length());
        // When the target sits behind the source in −Z (normal flow), cap the
        // handle length at the Z-gap: the cubic's Z-derivative stays ≤ 0 iff
        // l ≤ |dz|, so the curve runs strictly toward −Z and never swings back
        // toward +Z mid-span.
        if dz < 0.0 {
            l = l.min(-dz);
        }
        Self {
            p0: from_world,
            p1: from_world + Vec3::NEG_Z * l,
            p2: to_world + Vec3::Z * l,
            p3: to_world,
        }
    }

    pub fn sample(&self, t: f32) -> Vec3 {
        let it = 1.0 - t;
        self.p0 * (it * it * it)
            + self.p1 * (3.0 * it * it * t)
            + self.p2 * (3.0 * it * t * t)
            + self.p3 * (t * t * t)
    }
}

/// Five-way discriminant matching the leaf-type ordering used by anchor stacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeafKind {
    Bool,
    Char,
    Int,
    String,
    None,
}

impl LeafKind {
    /// Marquee text for this leaf. Spaces on both sides keep visible gaps
    /// between marquee repetitions once the texture is tiled.
    fn marquee_text(self) -> &'static str {
        match self {
            LeafKind::Bool => "  Bool  ",
            LeafKind::Char => "  Char  ",
            LeafKind::Int => "  Integer  ",
            LeafKind::String => "  String  ",
            LeafKind::None => "  None  ",
        }
    }

    /// Bare type name, without the marquee padding — used to compose the
    /// alternating "value + type" marquee on value-carrying edges.
    pub fn type_name(self) -> &'static str {
        match self {
            LeafKind::Bool => "Bool",
            LeafKind::Char => "Char",
            LeafKind::Int => "Integer",
            LeafKind::String => "String",
            LeafKind::None => "None",
        }
    }

    pub const ALL: [LeafKind; 5] = [
        LeafKind::Bool,
        LeafKind::Char,
        LeafKind::Int,
        LeafKind::String,
        LeafKind::None,
    ];
}

/// Map an `EType` leaf to its `LeafKind`. Sum types and unsupported variants
/// return `None`.
pub fn leaf_kind_of(t: &EType) -> Option<LeafKind> {
    match t {
        EType::Bool(..) => Some(LeafKind::Bool),
        EType::Char(..) => Some(LeafKind::Char),
        EType::Int(..) => Some(LeafKind::Int),
        EType::String(..) => Some(LeafKind::String),
        EType::None => Some(LeafKind::None),
        _ => None,
    }
}

/// Build a vertical ribbon that follows `curve`, extruded ±height/2 in Y
/// around `y_center(t) = mix(y_start, y_end, t)`. UV.x is arc length in world
/// units (so the shader can tile the label texture with `uv.x / tile_length`),
/// UV.y is 0 at the bottom edge and 1 at the top.
pub fn build_ribbon_mesh(curve: &EdgeCurve, y_start: f32, y_end: f32, height: f32) -> Mesh {
    let n = RIBBON_SEGMENTS;
    let half = height * 0.5;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((n + 1) * 2);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((n + 1) * 2);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity((n + 1) * 2);

    let mut arc = 0.0_f32;
    let mut prev = curve.sample(0.0);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let p = curve.sample(t);
        if i > 0 {
            arc += (p - prev).length();
        }
        prev = p;
        let y = y_start + (y_end - y_start) * t;
        // Bottom vertex (uv.y = 1), top vertex (uv.y = 0). Order chosen so
        // the strip winds consistently.
        positions.push([p.x, y - half, p.z]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([arc, 1.0]);
        positions.push([p.x, y + half, p.z]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([arc, 0.0]);
    }

    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);
    for i in 0..n {
        let i = i as u32;
        let b0 = i * 2; // bottom of segment i
        let t0 = i * 2 + 1; // top of segment i
        let b1 = i * 2 + 2; // bottom of segment i+1
        let t1 = i * 2 + 3; // top of segment i+1
        indices.extend_from_slice(&[b0, b1, t0, t0, b1, t1]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::mesh::Indices::U32(indices));
    mesh
}

pub const EDGE_SHADER_HANDLE: Handle<Shader> = uuid_handle!("45444745-0000-4000-8000-000000000001");

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct EdgeMaterial {
    #[uniform(0)]
    pub band_color: LinearRgba,
    #[uniform(0)]
    pub letter_color: LinearRgba,
    /// World units per second the marquee scrolls (output → input).
    #[uniform(0)]
    pub scroll_speed: f32,
    /// World units the label texture spans before repeating.
    #[uniform(0)]
    pub tile_length: f32,
    /// Seconds since app start, updated each frame.
    #[uniform(0)]
    pub time: f32,
    /// 0.0 = solid band (full ribbon coverage); 1.0 = hairline coverage that
    /// is only visible in a thin band around `uv.y = 0.5` and cut out where
    /// the marquee glyph texture has ink.
    #[uniform(0)]
    pub line_mode: f32,
    /// Half-thickness of the hairline in `uv.y` space when `line_mode == 1.0`.
    #[uniform(0)]
    pub line_half_thickness: f32,
    #[uniform(0)]
    pub _pad0: f32,
    #[uniform(0)]
    pub _pad1: f32,
    #[uniform(0)]
    pub _pad2: f32,
    #[texture(1)]
    #[sampler(2)]
    pub label: Handle<Image>,
}

impl Material for EdgeMaterial {
    fn fragment_shader() -> ShaderRef {
        EDGE_SHADER_HANDLE.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

#[derive(Resource)]
pub struct EdgeLabelTextures {
    pub by_kind: std::collections::HashMap<LeafKind, Handle<Image>>,
}

pub struct EdgePlugin;

impl Plugin for EdgePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            EDGE_SHADER_HANDLE,
            "../assets/shaders/edge_band.wgsl",
            Shader::from_wgsl
        );
        app.add_plugins(MaterialPlugin::<EdgeMaterial>::default())
            .add_systems(Startup, rasterize_label_textures)
            .add_systems(Update, update_edge_material_time);
    }
}

/// Push the current time into every `EdgeMaterial` so the shader can animate
/// the marquee. Iterating all materials each frame is cheap at edge counts
/// we expect (~O(30)) and matches Bevy's `animate_shader` example.
fn update_edge_material_time(time: Res<Time>, mut materials: ResMut<Assets<EdgeMaterial>>) {
    let t = time.elapsed_secs();
    for (_id, m) in materials.iter_mut() {
        m.time = t;
    }
}

/// Rasterize the five leaf-type labels into repeat-tileable R8Unorm images.
///
/// Uses `include_bytes!` on the JetBrainsMono TTF rather than the asset system
/// to sidestep WASM async-load races — the startup system must have a font
/// ready synchronously.
fn rasterize_label_textures(mut images: ResMut<Assets<Image>>, mut commands: Commands) {
    let font = FontRef::try_from_slice(FONT_BYTES).expect("bundled font is valid");

    let mut by_kind = std::collections::HashMap::new();
    for kind in LeafKind::ALL {
        let handle = rasterize_marquee_text(&font, kind.marquee_text(), &mut images);
        by_kind.insert(kind, handle);
    }
    commands.insert_resource(EdgeLabelTextures { by_kind });
}

/// Rasterise `text` fitted horizontally across a `LABEL_TEX_WIDTH`-wide
/// R8Unorm texture with wrap-repeat sampling, matching what the marquee
/// shader expects. Callers that only need the five per-leaf textures should
/// use the `EdgeLabelTextures` resource; this is for on-demand text
/// (e.g. value+type marquee strings on value-carrying edges).
pub fn rasterize_marquee_text(
    font: &FontRef<'_>,
    text: &str,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    let w = LABEL_TEX_WIDTH as usize;
    let h = LABEL_TEX_HEIGHT as usize;
    let mut buf = vec![0u8; w * h];

    // Fit the text horizontally across the full texture width so the marquee
    // period matches one "label-length" in UV space.
    let px = PxScale::from(LABEL_TEX_HEIGHT as f32 * 0.9);
    let scaled = font.as_scaled(px);
    let ascent = scaled.ascent();
    let baseline_y = ascent + (LABEL_TEX_HEIGHT as f32 - (ascent - scaled.descent())) * 0.5;

    let total_advance: f32 = text
        .chars()
        .map(|c| scaled.h_advance(scaled.font.glyph_id(c)))
        .sum();
    let scale_x = if total_advance > 0.0 {
        LABEL_TEX_WIDTH as f32 / total_advance
    } else {
        1.0
    };

    let mut pen_x = 0.0_f32;
    for c in text.chars() {
        let glyph_id = scaled.font.glyph_id(c);
        let advance = scaled.h_advance(glyph_id);
        let glyph =
            glyph_id.with_scale_and_position(px, ab_glyph::point(pen_x * scale_x, baseline_y));
        if let Some(outline) = font.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            outline.draw(|gx, gy, coverage| {
                let px_x = gx as i32 + bounds.min.x as i32;
                let px_y = gy as i32 + bounds.min.y as i32;
                if px_x < 0
                    || px_y < 0
                    || (px_x as u32) >= LABEL_TEX_WIDTH
                    || (px_y as u32) >= LABEL_TEX_HEIGHT
                {
                    return;
                }
                let idx = px_y as usize * w + px_x as usize;
                let v = (coverage * 255.0) as u8;
                if v > buf[idx] {
                    buf[idx] = v;
                }
            });
        }
        pen_x += advance;
    }

    let mut image = Image::new(
        Extent3d {
            width: LABEL_TEX_WIDTH,
            height: LABEL_TEX_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        buf,
        TextureFormat::R8Unorm,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    images.add(image)
}
