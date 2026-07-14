use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use bevy::asset::{load_internal_asset, uuid_handle};
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::*;
use bevy::shader::ShaderRef;

use crate::eval::EType;

pub const RIBBON_HEIGHT: f32 = 1.0;
pub const RIBBON_SEGMENTS: usize = 40;
const LABEL_TEX_WIDTH: u32 = 256;
const LABEL_TEX_HEIGHT: u32 = 32;

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
        // Clamp the handle length so same-Z-rank connections still bulge
        // visibly instead of degenerating into a straight line.
        let l = 1.5_f32
            .max(0.5 * (to_world.z - from_world.z).abs() + 0.25 * (to_world - from_world).length());
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

/// Six-way discriminant matching the leaf-type ordering used by anchor stacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LeafKind {
    Bool,
    Char,
    Int,
    Float,
    String,
    Undefined,
}

impl LeafKind {
    /// Marquee text for this leaf. Spaces on both sides keep visible gaps
    /// between marquee repetitions once the texture is tiled.
    fn marquee_text(self) -> &'static str {
        match self {
            LeafKind::Bool => "  bool  ",
            LeafKind::Char => "  char  ",
            LeafKind::Int => "  int  ",
            LeafKind::Float => "  float  ",
            LeafKind::String => "  string  ",
            LeafKind::Undefined => "  undefined  ",
        }
    }

    pub const ALL: [LeafKind; 6] = [
        LeafKind::Bool,
        LeafKind::Char,
        LeafKind::Int,
        LeafKind::Float,
        LeafKind::String,
        LeafKind::Undefined,
    ];
}

/// Map an `EType` leaf to its `LeafKind`. Sum types and unsupported variants
/// return `None`.
pub fn leaf_kind_of(t: &EType) -> Option<LeafKind> {
    match t {
        EType::Bool(..) => Some(LeafKind::Bool),
        EType::Char(..) => Some(LeafKind::Char),
        EType::Int(..) => Some(LeafKind::Int),
        EType::Float(..) => Some(LeafKind::Float),
        EType::String(..) => Some(LeafKind::String),
        EType::Undefined => Some(LeafKind::Undefined),
        _ => None,
    }
}

/// Build a vertical ribbon that follows `curve`, extruded ±0.5 in Y around
/// `y_center(t) = mix(y_start, y_end, t)`. UV.x is arc length in world units
/// (so the shader can tile the label texture with `uv.x / tile_length`),
/// UV.y is 0 at the bottom edge and 1 at the top.
pub fn build_ribbon_mesh(curve: &EdgeCurve, y_start: f32, y_end: f32) -> Mesh {
    let n = RIBBON_SEGMENTS;
    let half = RIBBON_HEIGHT * 0.5;

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
    #[uniform(0)]
    pub _pad: f32,
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

/// Rasterize the six leaf-type labels into repeat-tileable R8Unorm images.
///
/// Uses `include_bytes!` on the JetBrainsMono TTF rather than the asset system
/// to sidestep WASM async-load races — the startup system must have a font
/// ready synchronously.
fn rasterize_label_textures(mut images: ResMut<Assets<Image>>, mut commands: Commands) {
    const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf");
    let font = FontRef::try_from_slice(FONT_BYTES).expect("bundled font is valid");

    let mut by_kind = std::collections::HashMap::new();
    for kind in LeafKind::ALL {
        let handle = rasterize_one(&font, kind.marquee_text(), &mut images);
        by_kind.insert(kind, handle);
    }
    commands.insert_resource(EdgeLabelTextures { by_kind });
}

fn rasterize_one(font: &FontRef<'_>, text: &str, images: &mut Assets<Image>) -> Handle<Image> {
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
