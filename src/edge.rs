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
/// Mesh height of a ribbon drawn as a hairline. The band itself is invisible in
/// that mode (see `edge_band.wgsl`), so this is not a height that is *seen* —
/// it is the height the visible line is measured against, since
/// `RIBBON_LINE_HALF_THICKNESS_UV` is a fraction of it. The line comes out
/// `2 × 0.1 × CELL / 4`, i.e. a twentieth of a cell: thin like a grid line.
pub const RIBBON_LINE_HEIGHT: f32 = crate::render::CELL / 4.0;
/// Half-thickness of the value-edge hairline in `uv.y` space (i.e. as a
/// fraction of `RIBBON_LINE_HEIGHT`).
pub const RIBBON_LINE_HALF_THICKNESS_UV: f32 = 0.1;
pub const RIBBON_SEGMENTS: usize = 40;

/// Bytes of the bundled JetBrainsMono TTF. Exposed so callers that need to
/// rasterise text at spawn time (see `rasterize_face_text`) don't have to
/// `include_bytes!` the same path themselves.
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
///
/// It carries no data of its own — it exists so `leaf_kind_of` can answer two
/// questions the ribbon pass has to ask of every leaf: whether the leaf claims a
/// row at all, and whether it is `none`, which is always drawn as a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafKind {
    Bool,
    Char,
    Int,
    String,
    None,
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

/// One end of a ribbon: where its two edges sit in world Y, and how that end is
/// drawn — `0.0` a solid band, `1.0` a hairline.
///
/// The three travel together so a caller cannot pair the wrong `line_mode` with
/// a Y-pair. `build_tapered_ribbon_mesh` reads only the two Y values; the mode
/// is for the material, which is built from the same struct.
pub struct RibbonEnd {
    pub y_top: f32,
    pub y_bottom: f32,
    pub line_mode: f32,
}

impl RibbonEnd {
    /// A hairline centred on `y`.
    ///
    /// The mesh is `RIBBON_LINE_HEIGHT` tall rather than flat, and that is not
    /// a rounding-up: a ribbon of no height has no triangles to rasterise, and
    /// its `uv.y` would span nothing — but the *visible* line is the shader's,
    /// at `uv.y = 0.5`, and its thickness is a fraction of that span. So the
    /// mesh has to carry a height for the line to be a fraction *of*, and the
    /// line then falls exactly on `y`.
    fn hairline(y: f32) -> Self {
        let half = RIBBON_LINE_HEIGHT * 0.5;
        Self {
            y_top: y + half,
            y_bottom: y - half,
            line_mode: 1.0,
        }
    }
}

/// The end of a ribbon that meets leaf row `row_center_y`, claiming `span` of
/// it. `as_line` says whether that row is itself drawn as a line rather than as
/// a band — ask `render::leaf_is_drawn_as_line`, which is what the marker stack
/// asks.
///
/// Where a hairline sits depends on *why* it is one, and the two reasons are
/// not the same:
///
/// - the row is drawn as a line, so there is no band for the span to take a
///   share of. The ribbon meets the line where the marker stack draws it, on
///   the row's middle.
/// - the row is a band, but what claims it is a literal of an unbounded type,
///   which takes none of its height. It can then only attach at an edge, and
///   the edge is the top one: a band is read from the top down.
///
/// Everything else is a real share of the band — a whole one for a base type,
/// half of one for `true` or `false` — and is drawn as a band of that height.
pub fn ribbon_end(row_center_y: f32, span: &crate::infer::RowSpan, as_line: bool) -> RibbonEnd {
    let (y_top, y_bottom) = crate::render::row_span_world_y(row_center_y, span);
    if as_line {
        RibbonEnd::hairline(row_center_y)
    } else if span.is_degenerate() {
        RibbonEnd::hairline(y_top)
    } else {
        RibbonEnd {
            y_top,
            y_bottom,
            line_mode: 0.0,
        }
    }
}

/// Build a vertical ribbon that follows `curve`, with its top and bottom edge
/// interpolated independently from `start` to `end` — so a band can open out of
/// a line along the ribbon's length.
///
/// The curve's own Y is discarded: both ends state their Y outright, and the
/// curve is asked only where to be in X and Z. UV.x is arc length in world
/// units — the shader normalises it against the returned total to know how far
/// along the ribbon a fragment sits, and interpolates `line_mode` with it. UV.y
/// is 0 at the top edge and 1 at the bottom.
///
/// Returns the mesh and its total arc length.
pub fn build_tapered_ribbon_mesh(
    curve: &EdgeCurve,
    start: &RibbonEnd,
    end: &RibbonEnd,
) -> (Mesh, f32) {
    let n = RIBBON_SEGMENTS;

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
        let y_bottom = start.y_bottom + (end.y_bottom - start.y_bottom) * t;
        let y_top = start.y_top + (end.y_top - start.y_top) * t;
        // Bottom vertex (uv.y = 1), top vertex (uv.y = 0). Order chosen so
        // the strip winds consistently.
        positions.push([p.x, y_bottom, p.z]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([arc, 1.0]);
        positions.push([p.x, y_top, p.z]);
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
    (mesh, arc)
}

/// Ribbon of constant `height`, centred on a `y_start` → `y_end` ramp. What an
/// ordinary graph edge is drawn as, where both ends wear the same shape.
pub fn build_ribbon_mesh(curve: &EdgeCurve, y_start: f32, y_end: f32, height: f32) -> Mesh {
    let half = height * 0.5;
    let end_at = |y: f32| RibbonEnd {
        y_top: y + half,
        y_bottom: y - half,
        line_mode: 0.0,
    };
    build_tapered_ribbon_mesh(curve, &end_at(y_start), &end_at(y_end)).0
}

pub const EDGE_SHADER_HANDLE: Handle<Shader> = uuid_handle!("45444745-0000-4000-8000-000000000001");

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct EdgeMaterial {
    #[uniform(0)]
    pub band_color: LinearRgba,
    /// Seconds since app start, updated each frame by
    /// `update_edge_material_time`. Nothing reads it yet — it is kept against a
    /// coming edge animation.
    #[uniform(0)]
    pub time: f32,
    /// Coverage style at the ribbon's start, interpolated toward
    /// `line_mode_end` along its length: 0.0 = solid band (full ribbon
    /// coverage); 1.0 = hairline coverage, visible only in a thin band around
    /// `uv.y = 0.5`.
    ///
    /// It varies along the ribbon rather than being one value per edge because
    /// a value flowing into a Match arrives as a line and leaves the arm that
    /// accepts its whole type as a band. The two ends genuinely wear different
    /// shapes, and the strand between them has to become one from the other.
    #[uniform(0)]
    pub line_mode_start: f32,
    /// Half-thickness of the hairline in `uv.y` space, where the interpolated
    /// line mode is 1.0.
    #[uniform(0)]
    pub line_half_thickness: f32,
    /// Coverage style at the ribbon's end. Equal to `line_mode_start` for an
    /// ordinary edge, whose two ends are the same shape.
    #[uniform(0)]
    pub line_mode_end: f32,
    /// Total arc length of the ribbon in world units — what `uv.x` is divided
    /// by to place a fragment between the two line modes. `build_tapered_ribbon_mesh`
    /// returns it; anything non-zero will do where both modes agree.
    #[uniform(0)]
    pub arc_total: f32,
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
            .add_systems(Update, update_edge_material_time);
    }
}

/// Push the current time into every `EdgeMaterial`. Nothing in the shader reads
/// it yet; it is kept wound so a coming edge animation has a clock to start
/// from. Iterating all materials each frame is cheap at edge counts we expect
/// (~O(30)) and matches Bevy's `animate_shader` example.
fn update_edge_material_time(time: Res<Time>, mut materials: ResMut<Assets<EdgeMaterial>>) {
    let t = time.elapsed_secs();
    for (_id, m) in materials.iter_mut() {
        m.time = t;
    }
}

/// Pixels per cell in a body-face text texture.
///
/// A name is read at the near end of the zoom range (`camera::MAX_CELL_PIXELS`
/// is 160 px per cell) and a runtime image carries no mip chain, so the choice
/// is between blur up there and shimmer down at the default 40, where three
/// characters are thirteen pixels wide and unreadable either way. Sharp where
/// it is actually read.
const FACE_TEX_CELL_PX: u32 = 128;

/// Rasterise `text` across a `cells`-by-one-cell body face: fixed pitch,
/// `layout::NAME_CHARS_PER_CELL` characters per cell, white glyphs over
/// `background`.
///
/// The text is **not** stretched to the texture width. The pitch is the meaning
/// here — the body is as long as the name needs — so a character has to be
/// exactly a third of a cell, or the body's length stops being a count of
/// characters.
///
/// The background colour is baked in and the alpha left at 1 rather than
/// leaving the glyphs on transparency: the face then sits in the opaque pass,
/// writes depth (which is what keeps the scope's grid plane off the name), and
/// is indistinguishable from the body under it. Straight alpha would also
/// require painting the colour into fully transparent pixels, or bilinear
/// filtering drags dark seams around every glyph.
pub fn rasterize_face_text(
    font: &FontRef<'_>,
    text: &str,
    cells: u32,
    background: Color,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    let w = (cells.max(1) * FACE_TEX_CELL_PX) as usize;
    let h = FACE_TEX_CELL_PX as usize;
    let bg = background.to_srgba().to_u8_array();
    let mut buf: Vec<u8> = bg.iter().copied().cycle().take(w * h * 4).collect();

    let pitch = FACE_TEX_CELL_PX as f32 / crate::layout::NAME_CHARS_PER_CELL as f32;
    // `PxScale` is measured against the font's ascent-to-descent span, not
    // against its em square — for this face that is 1.32 em — so the size that
    // makes one advance a third of a cell is measured off the font rather than
    // derived from a nominal em size.
    let unit = font.as_scaled(PxScale::from(1.0));
    let unit_advance = unit.h_advance(unit.font.glyph_id('0'));
    let px = PxScale::from(if unit_advance > 0.0 {
        pitch / unit_advance
    } else {
        pitch
    });
    let scaled = font.as_scaled(px);
    let baseline_y =
        scaled.ascent() + (FACE_TEX_CELL_PX as f32 - (scaled.ascent() - scaled.descent())) * 0.5;
    // The body is a whole number of cells, so the room the name does not fill
    // is split between its two ends rather than hung off one of them.
    let used = text.chars().count() as f32 * pitch;
    let mut pen_x = (w as f32 - used) * 0.5;

    for c in text.chars() {
        let glyph = scaled
            .font
            .glyph_id(c)
            .with_scale_and_position(px, ab_glyph::point(pen_x, baseline_y));
        if let Some(outline) = font.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            outline.draw(|gx, gy, coverage| {
                let px_x = gx as i32 + bounds.min.x as i32;
                let px_y = gy as i32 + bounds.min.y as i32;
                if px_x < 0 || px_y < 0 || px_x as usize >= w || px_y as usize >= h {
                    return;
                }
                let idx = (px_y as usize * w + px_x as usize) * 4;
                for channel in 0..3 {
                    let lit = bg[channel] as f32 + (255.0 - bg[channel] as f32) * coverage;
                    buf[idx + channel] = buf[idx + channel].max(lit as u8);
                }
            });
        }
        // Fixed step: no accumulated advance, no stretch.
        pen_x += pitch;
    }

    let mut image = Image::new(
        Extent3d {
            width: w as u32,
            height: h as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        buf,
        TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        // Clamped on both axes: this texture is a single printed face, so a
        // repeat would bleed one end of the name into the other.
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    images.add(image)
}
