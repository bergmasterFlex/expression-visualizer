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

/// Mesh height of a ribbon drawn as a hairline. The band itself is invisible in
/// that mode (see `edge_band.wgsl`), so this is not a height that is *seen* —
/// it is the height the visible line is measured against, since
/// `RIBBON_LINE_HALF_THICKNESS_UV` is a fraction of it.
///
/// A ribbon of no height would have no triangles to rasterise and a `uv.y`
/// spanning nothing, so the carrier has to be taller than the line it carries.
/// Four times over is comfortable; nothing else depends on the factor.
pub const RIBBON_LINE_HEIGHT: f32 = crate::render::CELL / 4.0;
/// Half-thickness of the hairline in `uv.y` space, i.e. as a fraction of
/// `RIBBON_LINE_HEIGHT`.
///
/// Derived rather than stated, so the line leaving an anchor is the same line
/// that was inside it. These two used to carry separate numbers and had drifted
/// apart by a factor of two and a half, with the seam falling exactly on the
/// anchor's outward face.
pub const RIBBON_LINE_HALF_THICKNESS_UV: f32 =
    crate::render::STRAND_LINE_THICKNESS * 0.5 / RIBBON_LINE_HEIGHT;
/// Coverage below which a ribbon fragment is cut away rather than drawn.
///
/// Half, so that the cut lands exactly where the shapes say it should: the
/// hairline's smoothstep crosses 0.5 at `line_half_thickness`, and a dash flank
/// at its true boundary. `edge_band.wgsl` repeats this number — the two have to
/// agree, and a uniform for a constant would be a slot spent on nothing.
pub const COVERAGE_CUTOFF: f32 = 0.5;
pub const RIBBON_SEGMENTS: usize = 40;

/// Length of one dash-plus-gap of a pending edge's band, in world units along
/// the ribbon's arc.
///
/// Measured in cells, because the band is read against the grid it lies over:
/// a period that fits the grid's beat keeps the gaps from reading as an
/// accident of where a cell line happened to fall.
pub const RIBBON_DASH_PERIOD: f32 = crate::render::CELL * 0.6;
/// Share of a period that is dash: 0.36 of a cell drawn, 0.24 left out.
pub const RIBBON_DASH_DUTY: f32 = 0.6;

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
/// a band — ask `render::leaf_is_drawn_as_line`, which is what the strand stack
/// asks.
///
/// Where a hairline sits depends on *why* it is one, and the two reasons are
/// not the same:
///
/// - the row is drawn as a line, so there is no band for the span to take a
///   share of. The ribbon meets the line where the strand stack draws it, on
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
/// curve is asked only where to be in X and Z. But *how far along* the curve a
/// sample sits decides how far Y has fallen by then — the ribbon climbs by
/// distance travelled, not by curve parameter.
///
/// The two are not the same, because the cubic covers its own length unevenly:
/// the tangents drive it hardest along ±Z at either end, while its X eases in
/// and out as a smoothstep. Which of the two dominates depends on the edge, so
/// a Y ramp taken in the parameter runs ahead of the curve on some stretches
/// and behind on others, and the drop comes out as a shallow-steep-shallow S
/// rather than a slope.
///
/// That S matters more than it looks. The bound camera draws a world
/// displacement at `(−Δz + 0.296·Δx, Δy − 0.634·Δx)`, so a sideways run and a
/// drop are only 25° apart on screen and one cell of Y aliases with 1.6 cells
/// of X. The one cue left is the *shape* of the movement, and it only works
/// while the two shapes differ: X eases, Y climbs evenly.
///
/// UV.x is arc length in world units — the shader normalises it against the
/// returned total to know how far along the ribbon a fragment sits, and
/// interpolates `line_mode` with it. UV.y is 0 at the top edge and 1 at the
/// bottom.
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

    // Where the curve goes, and how far it has come by each sample. Two passes,
    // because the total has to be known before the first Y can be placed.
    //
    // The distance is measured in X and Z alone. That is the whole of the curve
    // the ribbon follows, and it is what the grid shows; the sampled Y is
    // dropped, so taking it along would measure a path nothing is drawn on —
    // the curve runs between the anchors' own rows while the ribbon runs
    // between the two `RibbonEnd`s, and the two agree only by coincidence.
    let mut samples: Vec<(Vec3, f32)> = Vec::with_capacity(n + 1);
    let mut arc = 0.0_f32;
    let mut prev = curve.sample(0.0);
    for i in 0..=n {
        let p = curve.sample(i as f32 / n as f32);
        if i > 0 {
            arc += Vec2::new(p.x - prev.x, p.z - prev.z).length();
        }
        prev = p;
        samples.push((p, arc));
    }
    let arc_total = arc;

    for (p, travelled) in samples {
        // A curve of no length leaves nothing to travel along, so everything
        // collapses onto `start` rather than dividing by zero.
        let s = if arc_total > 0.0 {
            travelled / arc_total
        } else {
            0.0
        };
        let y_bottom = start.y_bottom + (end.y_bottom - start.y_bottom) * s;
        let y_top = start.y_top + (end.y_top - start.y_top) * s;
        // Bottom vertex (uv.y = 1), top vertex (uv.y = 0). Order chosen so
        // the strip winds consistently.
        positions.push([p.x, y_bottom, p.z]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([travelled, 1.0]);
        positions.push([p.x, y_top, p.z]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([travelled, 0.0]);
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
    (mesh, arc_total)
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
    /// The strand's colour, from `render::strand_color`, so it matches the
    /// stretch of itself inside the anchors it joins. Only the RGB is read —
    /// the material cuts fragments away rather than blending them, so there is
    /// no alpha left for the shader to do anything with.
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
    /// World length of one dash-plus-gap along the ribbon's arc. `0.0` means a
    /// solid band, which is what every edge whose type is decided carries. Only
    /// a pending edge sets it, and the gaps are the whole of what it says: a
    /// band drawn through would claim the type is settled.
    #[uniform(0)]
    pub dash_period: f32,
    /// Share of a period that is dash. Read only where `dash_period` stands.
    #[uniform(0)]
    pub dash_duty: f32,
}

impl Material for EdgeMaterial {
    fn fragment_shader() -> ShaderRef {
        EDGE_SHADER_HANDLE.into()
    }

    /// Masked rather than blended, so a strand writes depth and the depth cue
    /// can see it — a strand's height above the plane is the hardest thing in
    /// the picture to judge, and a blended one got no help with it.
    ///
    /// Masked rather than plainly opaque, because the ribbon has real cut-outs
    /// to make: a hairline's mesh is `RIBBON_LINE_HEIGHT` tall while the line
    /// on it is `STRAND_LINE_THICKNESS`, so painting the carrier solid would
    /// draw every hairline five times too thick, and a pending strand's dashes
    /// would fill in.
    ///
    /// The cutting is the shader's own job — Bevy's automatic one lives in
    /// `pbr_functions.wgsl`, which `edge_band.wgsl` does not import — so this
    /// threshold only selects the pass. `edge_band.wgsl` cuts at the same 0.5.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Mask(COVERAGE_CUTOFF)
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
