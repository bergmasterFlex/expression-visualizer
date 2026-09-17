use bevy::prelude::*;

/// Edge length of one grid cell in world units.
///
/// Nodes and anchors span several cells each — a function call is as wide as
/// its inputs and four cells deep — so the cell itself is kept small. Every
/// size that means "a fraction of a cell" is derived from this rather than
/// hard-coded, so retuning the density stays a one-line change.
pub const CELL: f32 = 1.0;

/// Scale factor from layout coordinates to world coordinates.
///
/// Layout space is the pure non-negative address space: every cell address is
/// >= 0 on all three axes, and each scope volume has its origin corner at its
/// own (0,0,0). The axis *orientation* lives here and nowhere else, which is
/// why Y and Z are negated:
///
/// - layout `+X` -> world `+X`, toward the viewer in the default perspective
/// - layout `+Y` -> world `-Y`, downward on screen
/// - layout `+Z` -> world `-Z`, the causal direction `source -> sink`
///
/// Everything downstream of the conversion (meshes, edge tangents, anchor
/// offsets, camera, lights, base grid) stays in world space and is unaffected
/// by the layout-space sign convention.
pub const LAYOUT_SCALE: Vec3 = Vec3::new(CELL, -CELL, -CELL);

/// Convert a layout *address* to its world-space position.
///
/// A cell is anchored at its address by the corner facing the origin: cell
/// `N` occupies `[N, N+1)` on every axis. So this returns the cell's origin
/// corner, not its centre — use `cell_center_world` to place something inside
/// the cell.
pub fn layout_to_world(pos: Vec3) -> Vec3 {
    pos * LAYOUT_SCALE
}

/// World-space centre of the cell at `cell`. Node meshes and anchors sit here.
pub fn cell_center_world(cell: Vec3) -> Vec3 {
    layout_to_world(cell + Vec3::splat(0.5))
}

/// Inverse of `layout_to_world`. Turns a world-space point (e.g. a grid
/// raycast hit) back into layout space; `floor` it to get the containing cell
/// address.
pub fn world_to_layout(world: Vec3) -> Vec3 {
    world / LAYOUT_SCALE
}

/// World-space AABB covering the inclusive cell range `min..=max`, i.e. the
/// volume from the `min` corner to the far corner of `max`. `pad` widens it by
/// that many cells per side.
///
/// `LAYOUT_SCALE` negates Y and Z, so a layout `min` maps to a world `max` on
/// those axes; the result is re-normalised. Callers needing a world rect (grid
/// borders, the footprint uniforms the grid shader compares against) must go
/// through this rather than scaling `min`/`max` individually — otherwise the
/// rect comes out inverted, i.e. empty.
pub fn layout_range_to_world(min: Vec3, max: Vec3, pad: f32) -> (Vec3, Vec3) {
    let a = layout_to_world(min - Vec3::splat(pad));
    let b = layout_to_world(max + Vec3::splat(1.0 + pad));
    (a.min(b), a.max(b))
}

/// Edge thickness of the NORMAL-mode caret's cell outline, in world units.
const CARET_EDGE_THICKNESS: f32 = CELL / 60.0;

/// Alpha of the INSERT-mode caret faces. Nearly solid: the blink is what
/// keeps the cell behind the caret readable, not the paint.
const CARET_FACE_ALPHA: f32 = 0.8;

/// The linear brightness that arrives on screen as #FFFFFF — well past white.
///
/// The camera tonemaps (TonyMcMapface), which maps a linear 1.0 to roughly 0.8
/// on screen: anything painted plain white comes out grey. Overdriving it this
/// far puts it back at #FFFFFF after the curve. It only works because the
/// camera renders HDR; an 8-bit target would clamp this back to 1.0 first.
///
/// A property of the tonemapper rather than of any one thing drawn, so
/// everything that has to read as white goes through it: the caret's faces and
/// the names printed on a body. The floating labels do not — they are UI text
/// and never meet the curve, which is exactly why a name on a body used to
/// look grey beside a label that did not.
pub const DISPLAY_WHITE: f32 = 8.0;

/// The INSERT-mode caret: the two faces of the addressed cell that look back
/// toward the scope origin — the YZ face at the cell's lower X, the XY face at
/// its lower Z. Those are the planes a `Return` resp. `Space` insert opens
/// along, so the caret shows the corner the new layer would appear at.
///
/// `cell` is a layout address, and layout Y and Z are negated on the way to
/// world space — which is why the origin-facing side is the world *maximum* on
/// Z and only X reads the way it looks.
pub fn cell_caret_faces(cell: Vec3) -> Vec<RenderObject> {
    let a = layout_to_world(cell);
    let b = layout_to_world(cell + Vec3::ONE);
    let (lo, hi) = (a.min(b), a.max(b));
    let center = (lo + hi) * 0.5;
    let span = hi - lo;
    let material = || StandardMaterial {
        base_color: Color::LinearRgba(LinearRgba::new(
            DISPLAY_WHITE,
            DISPLAY_WHITE,
            DISPLAY_WHITE,
            CARET_FACE_ALPHA,
        )),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        ..default()
    };
    vec![
        // A Rectangle is built in the XY plane, so the YZ face is that quad
        // turned a quarter turn about Y: its width lands on world Z.
        RenderObject {
            mesh: Rectangle::new(span.z, span.y).mesh().build(),
            material: material(),
            transform: Transform::from_translation(Vec3::new(lo.x, center.y, center.z))
                .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
        },
        RenderObject {
            mesh: Rectangle::new(span.x, span.y).mesh().build(),
            material: material(),
            transform: Transform::from_translation(Vec3::new(center.x, center.y, hi.z)),
        },
    ]
}

/// Wireframe outline of the cell at `cell`: twelve thin cuboids spanning the
/// volume from `cell` to `cell + (1,1,1)`. This is the selection caret — it
/// encloses the addressed cell space itself rather than marking a point, and
/// is drawn whether or not a node occupies the cell.
pub fn cell_caret_edges(cell: Vec3) -> Vec<RenderObject> {
    let a = layout_to_world(cell);
    let b = layout_to_world(cell + Vec3::ONE);
    let (lo, hi) = (a.min(b), a.max(b));
    let center = (lo + hi) * 0.5;
    let span = hi - lo;
    let t = CARET_EDGE_THICKNESS;
    let mut out = Vec::with_capacity(12);
    for axis in 0..3usize {
        // The two axes the edge is offset along; the edge runs along `axis`.
        let (u, v) = match axis {
            0 => (1usize, 2usize),
            1 => (0usize, 2usize),
            _ => (0usize, 1usize),
        };
        let mut size = Vec3::splat(t);
        size[axis] = span[axis] + t;
        for (su, sv) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            let mut pos = center;
            pos[u] = center[u] + su * span[u] * 0.5;
            pos[v] = center[v] + sv * span[v] * 0.5;
            out.push(RenderObject {
                mesh: Cuboid::new(size.x, size.y, size.z).mesh().build(),
                material: StandardMaterial {
                    base_color: Color::srgba(0.85, 0.84, 0.80, 0.7),
                    alpha_mode: AlphaMode::Blend,
                    cull_mode: None,
                    unlit: true,
                    ..default()
                },
                transform: Transform::from_translation(pos),
            });
        }
    }
    out
}

pub struct RenderObject {
    pub mesh: Mesh,
    pub material: StandardMaterial,
    pub transform: Transform,
}

/// A face of a body with text printed on it — the name a Source carries on its
/// top face, rather than a label floating beside the node.
///
/// The texture is the spawner's job, not the renderer's: rasterising one needs
/// `Assets<Image>`, which only the spawner has. So this hands over the string
/// and how many cells it has to fill, and the spawner rasterises, caches and
/// fills in `base_color_texture`. `background` is the body's own colour, baked
/// into the texture so the face and the body under it are indistinguishable.
pub struct RenderTextFace {
    pub mesh: Mesh,
    pub transform: Transform,
    /// Everything but `base_color_texture`, which the spawner supplies.
    pub material: StandardMaterial,
    pub text: String,
    pub cells: u32,
    pub background: Color,
}

/// What a printed face is painted with: the gain that carries its texture up
/// into the range the tonemapper turns back into real colour.
///
/// `base_color` *multiplies* `base_color_texture`, and that multiplication is
/// doing a job here rather than being tolerated. The texture is eight-bit and
/// so cannot hold a value past 1.0, but a name has to reach `DISPLAY_WHITE` to
/// come out white on screen. So `edge::rasterize_face_text` stores everything
/// divided by that gain and this puts it back — the body's colour lands on
/// exactly the body's colour, and the glyphs land past white.
///
/// Painting this in the body's colour, which is what it did first, squared the
/// background and pulled the glyphs down to the body's own colour; painting it
/// plain white fixed the background but left the glyphs at a linear 1.0, which
/// the curve delivers as grey.
fn face_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::LinearRgba(LinearRgba::new(
            DISPLAY_WHITE,
            DISPLAY_WHITE,
            DISPLAY_WHITE,
            1.0,
        )),
        unlit: true,
        ..default()
    }
}

pub struct RenderNode {
    /// `None` for nodes drawn purely as strands or loose objects; the node
    /// entity is still spawned so picking and selection keep working.
    pub node: Option<RenderObject>,
    pub anchors: std::collections::HashMap<crate::model::anchor::Id, RenderAnchor>,
    /// Strands belonging to the node itself rather than to an anchor. A
    /// Pattern uses this: it declares the type its arm matches and is drawn as
    /// that type's band, but owns no anchor to hang it on.
    pub strands: Vec<RenderStrand>,
    /// Meshes that are neither the node's own body nor an anchor's: the line
    /// and the point a Constant is drawn as. They carry no identity — nothing
    /// picks them and no edge ends on them.
    pub objects: Vec<RenderObject>,
    pub labels: Vec<RenderLabel>,
    pub text_faces: Vec<RenderTextFace>,
}

pub struct RenderAnchor {
    /// World point used for screen-space hover picking and as the edge
    /// endpoint. Sits at the centre of the anchor sheet (or sheet stack).
    pub pick_center: Vec3,
    pub strands: Vec<RenderStrand>,
    /// Neutral sheet for anchors that carry no strands (Sink,
    /// Match), so they stay visible and pickable.
    pub plain_body: Option<RenderObject>,
}

/// One leaf's strand where it passes through an anchor.
///
/// A strand is drawn one of two ways — as a band when it carries a type, as a
/// line when it carries a literal — and it wears that shape for its whole
/// length. This is the short stretch of it inside an anchor; the long curved
/// stretch between two anchors is the same strand, built as a ribbon in
/// `edge.rs`. The two ask the same question of the same leaf
/// (`leaf_is_drawn_as_line`) and take their colour from the same place
/// (`strand_color`), so they cannot meet each other wearing different shapes.
///
/// Exactly one of `band` and `line` is present, never both.
pub struct RenderStrand {
    /// The type band. Absent when the leaf carries a literal: the line replaces
    /// it entirely, the same way a value-carrying ribbon drops its band for a
    /// hairline.
    pub band: Option<RenderObject>,
    /// The type's letter, written across the band.
    pub band_label: Option<RenderLabel>,
    /// A thin coloured segment spanning the anchor's full depth, at its
    /// Y-middle. Present iff this leaf's anchor carries a graph-level literal.
    pub line: Option<RenderObject>,
    /// Present alongside `line`. The literal itself, projected into screen
    /// space past the line's tip.
    pub line_label: Option<RenderLabel>,
}

pub struct RenderLabel {
    pub text: String,
    pub color: Color,
    pub font_size: f32,
    pub world_pos: Vec3,
    pub offset: Vec2,
}

/// Height of a strand drawn as a band, and so also the pitch of the anchor
/// rows — a sum type's bands are stacked flush, which is what makes it read as
/// one unbroken band rather than as a pile of separate ones.
///
/// Exactly one cell, since an anchor claims one cell per sum-type member. The
/// band fills the cell in Y but stays slim in X and Z, so the graph keeps its
/// airy look. The connecting strand is built to this same height, which is what
/// lets a band leave an anchor without a step.
pub const STRAND_BAND_HEIGHT: f32 = CELL;
const ANCHOR_HALF_DEPTH: f32 = CELL / 4.0;
/// Full Z-depth of an anchor sheet: half its cell.
///
/// An anchor occupies the half of its cell that faces the node body, so it
/// visibly hangs off the thing it belongs to instead of floating mid-cell. In
/// cell-local terms an input takes `0.5..1` and an output `0..0.5`; since
/// layout +Z is world −Z, that is the −Z half for inputs and the +Z half for
/// outputs. Both meet the cell centre, which is where their edges attach.
const ANCHOR_DEPTH: f32 = 2.0 * ANCHOR_HALF_DEPTH;
/// Y-thickness of a strand drawn as a line — the shape a leaf takes when it
/// carries a literal rather than a type.
///
/// The single source for that thickness. `edge::RIBBON_LINE_HALF_THICKNESS_UV`
/// derives its own fraction from this rather than stating a number of its own,
/// so the segment inside an anchor and the strand leaving it cannot come apart
/// the way they had: this used to be a fiftieth of a cell here against a
/// twentieth out there, and the two met at the anchor's outward face.
///
/// A twentieth of a cell is thin like a grid line, and at the default zoom it
/// is two pixels — which is what a line drawn by an alpha cutout, with no
/// multisampling to fall back on, needs to stay a closed line.
pub const STRAND_LINE_THICKNESS: f32 = CELL / 20.0;
/// The point a Constant's value starts at, sitting at the centre of its body
/// cell. A couple of times the line's own thickness, which is what makes it
/// read as the end of the line rather than as a ball threaded onto it.
const CONSTANT_DOT_RADIUS: f32 = CELL * 0.05;
/// World-space padding between the tip of the gizmo line and the value
/// label's projection point.
const VALUE_LABEL_Z_PADDING: f32 = CELL / 30.0;
/// Screen-space nudge that carries a Source's index label clear of the type
/// letter it hangs off, to its right. Measured from the letter rather than from
/// the cell, because both are glyphs of a fixed pixel size: a gap stated in
/// pixels then holds at every zoom, while one stated in world units would close
/// as the camera pulls back.
const SOURCE_INDEX_LABEL_OFFSET_X: f32 = 18.0;
/// How far a text face floats above the body face it prints on.
///
/// Coplanar is not an option: both surfaces would land in the same depth
/// bucket. It used to be two reasons — a scope's grid plane lay at world Y=0,
/// exactly the plane a body's top face lies in, and washed its 55% veil over
/// every top face in the scene. The plane is the volume's floor now and meets
/// the *bottom* faces of the last row instead, so only the depth argument is
/// left. It carries the constant on its own.
///
/// The bound camera's depth axis is world X and its elevation is a shear, not
/// a tilt, so a lift of `l` in Y buys about `1.6 · l` of depth separation. The
/// depth buffer resolves some `2·10⁻⁵` world units over its range, so a
/// five-hundredth of a cell clears it by two orders of magnitude while staying
/// under a third of a pixel at the tightest zoom.
const BODY_FACE_LIFT: f32 = CELL / 500.0;

/// Sort key that fixes the vertical order of the rows of a sum type. Returns
/// `None` for variants that claim no row.
///
/// `ordered_supported_leaves` sorts by the *reverse* of this, and row 0 is the
/// topmost — so the highest number is drawn at the top and `none`, lowest, at
/// the bottom. That is where it belongs: a sum type reads as "one of these, or
/// else nothing", and the "or else" is the last thing said, not the first.
fn type_order(t: &crate::infer::EType) -> Option<u8> {
    match t {
        crate::infer::EType::None => Some(0),
        crate::infer::EType::Bool(..) => Some(1),
        crate::infer::EType::Char(..) => Some(2),
        crate::infer::EType::Int(..) => Some(3),
        crate::infer::EType::String(..) => Some(4),
        _ => None,
    }
}

/// The letter a type is written with, on the middle of the strand carrying it.
///
/// `none` asks for one too, and its arm is the interesting one. It is the only
/// leaf drawn as a *line* that still wears a letter rather than a value —
/// because its value is its type, so the type is the only thing there is to
/// write. That is what puts `n` flush with `i`, `s`, `c` and `b` instead of
/// spelling the word out past the strand's tip the way a literal does.
fn type_letter(t: &crate::infer::EType) -> &'static str {
    match t {
        crate::infer::EType::Bool(..) => "b",
        crate::infer::EType::Char(..) => "c",
        crate::infer::EType::Int(..) => "i",
        crate::infer::EType::String(..) => "s",
        crate::infer::EType::None => "n",
        _ => "?",
    }
}

/// The colour a strand of type `t` is drawn in, wherever it appears: the
/// segment inside an anchor, the strand running between two of them, and the
/// body of a node that declares it.
///
/// The palette is Okabe–Ito, and it is not a matter of taste. Colour is the
/// only thing that says which type a strand carries — there is no second
/// channel saying it again — so a reader who cannot separate two of these
/// cannot read the graph at all. Okabe–Ito is chosen for exactly that: its
/// hues stay apart under the common forms of colour blindness, which an
/// ad-hoc red/green/blue set does not.
///
/// Written as bytes rather than as floats so that what stands here is the
/// palette as it is published, and a value can be checked against the source
/// without converting anything first.
///
/// Opaque, and that is the whole of the other choice. A strand used to blend
/// at 0.6 so the grid showed through it, but only opaque geometry writes
/// depth, and a strand that writes no depth is invisible to the depth cue —
/// which is exactly the wrong way round, since a strand's height above the
/// plane is the hardest thing in the picture to judge by eye. The node bodies
/// had already worked around the old alpha by forcing it back to 1.0 at the
/// call site.
pub fn strand_color(t: &crate::infer::EType) -> Color {
    match t {
        // reddish purple, #CC79A7
        crate::infer::EType::Bool(..) => Color::srgb_u8(204, 121, 167),
        // bluish green, #009E73
        crate::infer::EType::Char(..) => Color::srgb_u8(0, 158, 115),
        // blue, #0072B2
        crate::infer::EType::Int(..) => Color::srgb_u8(0, 114, 178),
        // orange, #E69F00
        crate::infer::EType::String(..) => Color::srgb_u8(230, 159, 0),
        // vermillion, #D55E00 — the palette's nearest neighbour to the orange
        // above, and deliberately so: `none` is the sad path of a type, not a
        // type of its own, and reads as a darker, redder cousin of one.
        crate::infer::EType::None => Color::srgb_u8(213, 94, 0),
        // Pending, and whatever else has not been decided. Outside the palette
        // on purpose: neutral grey is the absence of a hue, which is the
        // absence of a type.
        _ => Color::srgb(0.5, 0.5, 0.5),
    }
}

/// World-space Y offset of leaf row `index` relative to the anchor's own row.
///
/// Rows run from the anchor's cell along +Y in layout space, which is downward
/// in world space — so the offset is negative and grows with the index. The
/// leaves are no longer centred on the anchor: an anchor's address is its
/// first row.
///
/// Both the strand stack (`build_anchor_strands`) and the edge ribbons
/// (`spawn_graph_nodes`) go through this, so they cannot drift apart.
pub fn leaf_row_offset(index: usize) -> f32 {
    index as f32 * LAYOUT_SCALE.y.signum() * STRAND_BAND_HEIGHT
}
/// The row-claiming leaves of `t`, in the order they are stacked: String at the
/// top, then Int, Char, Bool, and `none` at the bottom (`type_order`).
///
/// Which leaves claim a row is decided by `infer::row_leaves`, so the render
/// stack and the cell addressing can never disagree about how tall an anchor
/// is; this only fixes their order. Both the strand stack and the edge ribbons
/// read it, which is what keeps a strand meeting the row it belongs to.
pub fn ordered_supported_leaves(t: &crate::infer::EType) -> Vec<crate::infer::EType> {
    let mut leaves = crate::infer::row_leaves(t);
    leaves.sort_by_key(|leaf| std::cmp::Reverse(type_order(leaf).unwrap_or(u8::MAX)));
    leaves
}

/// The rows an anchor draws, each paired with the row index it owns.
///
/// Without a run this is `ordered_supported_leaves` and its own indices: every
/// leaf of the declared type, in stacking order.
///
/// A run that has reached this anchor drops the rows it did not take. A
/// `Char|None` that produced `'b'` never went down the `none` row, and drawing
/// that row would say the outcome is still open when it has been settled. What
/// stays is the row the value landed on — narrowed to the value, so it is drawn
/// as a line rather than as a band.
///
/// The index travels with it rather than being recounted, and that is the whole
/// point of returning pairs. The layout reserved cells from the *declared* type
/// (`infer::anchor_rows`, deliberately blind to any run — see `infer::Known`),
/// so a surviving row that slid up to position 0 would leave its strand hanging
/// off the node and miss every ribbon aimed at it.
///
/// Which row the value landed on is asked with `infer::types_match`, the
/// comparison that ignores literals and asks only what *kind* of thing this is
/// — `30` lands on the `Integer` row. A value matching no declared row can only
/// come of a graph edited between two steps; the run is then talking about a
/// node that no longer exists as it was, and the written type is the better
/// answer than an empty cell, so every row is kept.
pub fn drawn_rows(
    declared: &crate::infer::EType,
    taken: Option<&crate::infer::EType>,
) -> Vec<(usize, crate::infer::EType)> {
    let rows = ordered_supported_leaves(declared);
    let Some(taken) = taken else {
        return rows.into_iter().enumerate().collect();
    };
    match rows
        .iter()
        .position(|row| crate::infer::types_match(row, taken))
    {
        Some(index) => vec![(index, taken.clone())],
        None => rows.into_iter().enumerate().collect(),
    }
}

/// World-space Y of the top and bottom edge of `span` within the leaf row
/// centred at `row_center_y`.
///
/// A span is stated in the band's own reading direction — `0.0` is its top edge
/// — while rows are stacked along layout +Y, which is world −Y. The step is
/// therefore taken from `LAYOUT_SCALE` exactly as `leaf_row_offset` takes it,
/// so a flip of the axis convention moves both together instead of leaving this
/// one silently upside down.
///
/// The pair is returned top first, i.e. the larger world Y first.
pub fn row_span_world_y(row_center_y: f32, span: &crate::infer::RowSpan) -> (f32, f32) {
    let step = LAYOUT_SCALE.y.signum() * STRAND_BAND_HEIGHT;
    let row_top = row_center_y - step * 0.5;
    (row_top + span.top * step, row_top + span.bottom * step)
}

/// The literal a leaf carries, if it carries one: the one the leaf itself is
/// pinned to, or the one pinned to its anchor.
///
/// The leaf's own literal comes first and the anchor's is the fallback. A leaf
/// carries a literal when the type says so — one row of a `1|2` — and the
/// anchor carries one when the *node* says so, which a Constant does for a type
/// that names no value of its own. Where both speak they agree; where only one
/// does, it is the one that knows.
///
/// `none` is answered `None` outright, before either of them is asked, and
/// that is not a gap: it carries no literal because its *value is its type*.
/// It is still drawn as a line — see `leaf_is_drawn_as_line` — but a line
/// labelled with a type letter rather than with a word, which is the whole of
/// what makes it read as the type it is.
///
/// Asked outright rather than left to `leaf_literal`, which answers `None` for
/// it too and would then hand the question straight to the fallback. The
/// fallback is the *node's* word, and `none` is not a value of the node: a
/// partial cast to `42` has an output of `42|none` and an anchor that carries
/// `42`, so its sad row wore the target's literal instead of its own letter —
/// the one row in the picture that says the cast failed, spelled as the value
/// it failed to produce.
fn leaf_value_text(leaf: &crate::infer::EType, graph_value: Option<&str>) -> Option<String> {
    if matches!(leaf, crate::infer::EType::None) {
        return None;
    }
    crate::infer::leaf_literal(leaf)
        .or(graph_value)
        .map(str::to_string)
}

/// True when a leaf is drawn as a thin line rather than as a band: it is
/// `none`, whose value is its type, or a literal is pinned to it.
///
/// Both the strand stack and every ribbon that meets it ask this, of the same
/// leaf, so the two cannot disagree about what shape they are joining.
pub fn leaf_is_drawn_as_line(leaf: &crate::infer::EType, graph_value: Option<&str>) -> bool {
    matches!(leaf, crate::infer::EType::None) || leaf_value_text(leaf, graph_value).is_some()
}

/// The face of an anchor's band that faces the node body — the far face of an
/// input, the near face of an output.
///
/// An anchor's *cell centre* is its outward face, which is where an ordinary
/// edge arrives (see `ANCHOR_DEPTH`). The connections a Match is made of leave
/// and arrive on the other side, so they need this one instead: a link aimed at
/// the cell centre of an output would run through the whole band before
/// stopping at its back.
pub fn anchor_body_face_world(anchor_world_pos: Vec3, is_input: bool) -> Vec3 {
    let sign = if is_input { -1.0 } else { 1.0 };
    Vec3::new(
        anchor_world_pos.x,
        anchor_world_pos.y,
        anchor_world_pos.z + sign * ANCHOR_DEPTH,
    )
}

/// World centre of the cell a Pattern names its arm's type on.
///
/// A Pattern owns no anchor, so this address exists nowhere else — and it is
/// needed twice, by the node pass that draws the band and by the link pass that
/// has to meet it. Written once here so the two cannot drift.
pub fn pattern_band_world(layout_node: &crate::layout::LayoutNode, extra_offset: Vec3) -> Vec3 {
    cell_center_world(
        layout_node.pos
            + extra_offset
            + Vec3::new(0.0, 0.0, crate::layout::PATTERN_TYPE_LOCAL_Z as f32),
    )
}

/// World centre of the cell a TypeCast names its target type on. The same
/// address a Pattern's arm has, one node-local hop instead of two: a cast
/// borrows the Match's rhythm but has no Pattern node to go through.
pub fn cast_band_world(layout_node: &crate::layout::LayoutNode, extra_offset: Vec3) -> Vec3 {
    cell_center_world(
        layout_node.pos + extra_offset + Vec3::new(0.0, 0.0, crate::layout::CAST_TYPE_Z as f32),
    )
}

/// Whether a strand stack writes in words what its shape already says.
///
/// The type is in the colour and the value is in the shape — a band for a type,
/// a line for a value — so the writing is a convenience, not the statement. Once
/// several bands of the same type stand within a cell or two of each other, the
/// convenience turns into noise: the same word five times over two cells is
/// harder to read past than no word at all.
#[derive(Clone, Copy)]
enum Lettering {
    /// The type's letter across a band, the literal beside a line.
    Spelled,
    /// Neither. Something adjacent has already spelled it.
    Silent,
}

/// Build the stack of translucent type rectangles at an anchor.
///
/// `anchor_world_pos` is the world centre of the anchor's **first row** cell.
/// Each further leaf sits one cell further along +Y in layout space (see
/// `leaf_row_offset`), so the stack grows downward from the anchor's address
/// rather than being centred on it.
///
/// In Z each rect fills the half of its cell facing the node body — the far
/// half for an input, the near half for an output — so the stack hangs off the
/// node rather than floating mid-cell. `is_input` picks the side.
///
/// `graph_value` is the graph-level literal on the anchor's type, if any. When
/// present the band is dropped entirely and the leaf is drawn as a single
/// thin line plus the literal — a value is shown as the value, not as its
/// type. In practice value-carrying nodes have a single leaf, so this only
/// fires on one strand per anchor.
///
/// `none` takes that same line whether or not anything was pinned to the
/// anchor, and it is the one leaf that does: it is the type whose value *is*
/// the type, so there is no band it could honestly wear. That is why the
/// choice is made per leaf rather than per anchor — in `Char|None` the `Char`
/// keeps its band and only the `none` row becomes a line.
///
/// `lettering` says whether the stack writes any of that down. Where several
/// bands for the same type stand within a cell or two of each other — a Match's
/// input, its arm, the branch source behind it — only one of them needs the
/// word, and the rest read better without it.
///
/// `taken` is the row a run settled on, `None` where none has. It drops the
/// rows the run did not take while leaving the taken one on the index it always
/// had — see `drawn_rows`, which both this and the ribbons meeting it go
/// through.
fn build_anchor_strands(
    t: &crate::infer::EType,
    graph_value: Option<&str>,
    taken: Option<&crate::infer::EType>,
    anchor_world_pos: Vec3,
    is_input: bool,
    lettering: Lettering,
) -> Vec<RenderStrand> {
    let rows = drawn_rows(t, taken);
    if rows.is_empty() {
        return vec![];
    }
    // Direction the anchor body extends from the cell centre: toward the node,
    // i.e. −Z for an input (cell-local 0.5..1) and +Z for an output (0..0.5).
    let sign = if is_input { -1.0 } else { 1.0 };
    let full_depth = ANCHOR_DEPTH;
    // The cell centre is the anchor's outward face — the point its edge meets —
    // so every span is measured from there into the anchor's own half.
    let full_rect_z_center = anchor_world_pos.z + sign * ANCHOR_HALF_DEPTH;
    let line_tip_z = anchor_world_pos.z + sign * full_depth;

    rows
        .into_iter()
        .map(|(k, leaf)| {
            let y_center = anchor_world_pos.y + leaf_row_offset(k);
            let color = strand_color(&leaf);
            let letter = type_letter(&leaf).to_string();
            let center = Vec3::new(anchor_world_pos.x, y_center, full_rect_z_center);
            // Whether this row is drawn as a line at all — asked through the
            // one function every ribbon meeting this row asks, so a strand
            // joins the shape it actually finds.
            if leaf_is_drawn_as_line(&leaf, graph_value) {
                // A line across the anchor's full depth, its colour carrying
                // the type. No band — the same choice the edge shader makes for
                // value-carrying strands.
                //
                // What is written on it depends on *why* it is a line, and the
                // two reasons want different places. A literal writes its value
                // out past the tip, where a word of any length has room. `none`
                // carries no literal — its value is its type — so it writes the
                // type's own letter on the strand's middle, at the very
                // `center` the band branch below gives `i`, `s`, `c` and `b`.
                // That is the whole of what puts it flush with them: the same
                // position, not a correction applied to a different one.
                let (label_text, label_world) = match leaf_value_text(&leaf, graph_value) {
                    Some(value) => (
                        value,
                        Vec3::new(
                            anchor_world_pos.x,
                            y_center,
                            line_tip_z + sign * VALUE_LABEL_Z_PADDING,
                        ),
                    ),
                    None => (letter, center),
                };
                RenderStrand {
                    band: None,
                    band_label: None,
                    line: Some(RenderObject {
                        mesh: Cuboid::new(0.0, STRAND_LINE_THICKNESS, full_depth)
                            .mesh()
                            .build(),
                        material: StandardMaterial {
                            base_color: color,
                            cull_mode: None,
                            unlit: true,
                            ..default()
                        },
                        transform: Transform::from_translation(center),
                    }),
                    line_label: match lettering {
                        Lettering::Spelled => Some(RenderLabel {
                            text: label_text,
                            color: Color::WHITE,
                            font_size: 14.0,
                            world_pos: label_world,
                            offset: Vec2::ZERO,
                        }),
                        Lettering::Silent => None,
                    },
                }
            } else {
                RenderStrand {
                    band: Some(RenderObject {
                        // Flat in X, exactly as the ribbon leaving it is.
                        // An anchor's segment and the strand that continues
                        // it are one surface running along Z, and a segment
                        // carrying a thickness the strand has no way to match
                        // reads as a second thing wearing the same colour.
                        mesh: Cuboid::new(0.0, STRAND_BAND_HEIGHT, full_depth)
                            .mesh()
                            .build(),
                        material: StandardMaterial {
                            base_color: color,
                            cull_mode: None,
                            unlit: true,
                            ..default()
                        },
                        transform: Transform::from_translation(center),
                    }),
                    band_label: match lettering {
                        Lettering::Spelled => Some(RenderLabel {
                            text: letter,
                            color: Color::WHITE,
                            font_size: 14.0,
                            world_pos: center,
                            offset: Vec2::ZERO,
                        }),
                        Lettering::Silent => None,
                    },
                    line: None,
                    line_label: None,
                }
            }
        })
        .collect()
}

/// Anchor rendered from an inferred type: a strand stack when the type
/// has renderable leaves, otherwise the neutral grey body. `Pending` has no
/// leaves, so an output whose type the inferer cannot decide yet reads exactly
/// like the typeless (unconstrained) inputs.
fn typed_anchor(
    t: &crate::infer::EType,
    graph_value: Option<&str>,
    taken: Option<&crate::infer::EType>,
    cell_center: Vec3,
    is_input: bool,
    lettering: Lettering,
) -> RenderAnchor {
    let strands = build_anchor_strands(t, graph_value, taken, cell_center, is_input, lettering);
    RenderAnchor {
        // The cell centre is the anchor's outward face, so edges meet it there
        // no matter how many rows the anchor spans.
        pick_center: cell_center,
        plain_body: strands
            .is_empty()
            .then(|| plain_anchor_body(cell_center, is_input)),
        strands,
    }
}

/// A declared type drawn on a cell the way an input anchor is drawn — the
/// type's band, or its literal's line — and the neutral grey where no type has
/// been chosen yet.
///
/// Two kinds hang a type on a cell of their own rather than on an anchor: a
/// Pattern, whose cell is the arm it matches, and a TypeCast, whose cell is what
/// it converts to. Both are drawn input-side — the cell centre is the near face
/// the incoming strands meet — and there the likeness ends. A Pattern's cell
/// names what may *arrive* at it; a cast's names what *leaves* it, which is a
/// different sentence about the same shape.
///
/// Usually neither writes anything: a declared type stands between an anchor
/// that names the same type and one that names what it narrows to, so it is
/// never the only place the word could be read. `lettering` is how a caller says
/// that this time it is — which happens to a cast with nothing wired into it,
/// whose input and output are both grey and whose target cell is then the only
/// thing on screen that knows the type.
///
/// It returns two vectors because `RenderNode::strands` has no `plain_body` slot
/// the way `RenderAnchor` does — an unchosen type has no leaves, so
/// `build_anchor_strands` yields nothing and the grey goes into `objects`.
fn declared_type_cell(
    r#type: Option<&crate::model::r#type::EType>,
    cell_center: Vec3,
    lettering: Lettering,
) -> (Vec<RenderStrand>, Vec<RenderObject>) {
    let eval_type = r#type
        .map(crate::infer::graph_type_to_eval_type)
        .unwrap_or(crate::infer::EType::Pending);
    let literal = r#type.and_then(crate::layout::value_of_etype);
    // No `taken`: a declared cell says what the program asks for, and a run
    // narrows what *arrives*, never what was asked.
    let strands = build_anchor_strands(
        &eval_type,
        literal.as_deref(),
        None,
        cell_center,
        true,
        lettering,
    );
    let objects = strands
        .is_empty()
        .then(|| plain_anchor_body(cell_center, true))
        .into_iter()
        .collect();
    (strands, objects)
}

/// A neutral grey anchor sheet for anchors that carry no strands
/// (unconstrained inputs, pending outputs), so they stay visible and pickable.
/// `cell_center` is the anchor cell's centre; the sheet fills that cell's
/// body-facing half, like a strand would.
fn plain_anchor_body(cell_center: Vec3, is_input: bool) -> RenderObject {
    // Same half of the cell a strand would occupy, so a typeless anchor
    // hangs off its node exactly like a typed one.
    let sign = if is_input { -1.0 } else { 1.0 };
    let center = cell_center + Vec3::new(0.0, 0.0, sign * ANCHOR_DEPTH * 0.5);
    RenderObject {
        // Flat in X for the reason the typed bands are — see
        // `build_anchor_strands`. A grey anchor is the same shape as a
        // coloured one, undecided rather than different.
        mesh: Cuboid::new(0.0, STRAND_BAND_HEIGHT, ANCHOR_DEPTH)
            .mesh()
            .build(),
        material: StandardMaterial {
            // Through the same call the grey strands go through, so a typeless
            // anchor and the undecided strand leaving it cannot end up wearing
            // two different greys.
            base_color: strand_color(&crate::infer::EType::Pending),
            cull_mode: None,
            unlit: true,
            ..default()
        },
        transform: Transform::from_translation(center),
    }
}

/// Spawn the graph node meshes.
///
/// `extra_offset` (grid units) is added to `layout_node.pos` before the
/// grid→world conversion; used for pattern sub-graph nodes whose positions are
/// relative to the containing pattern.
///
/// `flat_graph` is the program's flattened graph. Type inference needs it because
/// every edge — including those inside Pattern branches — lives in the
/// program-level edge table, while `layout_graph` may be a sub-layout that holds
/// only nodes.
pub fn layoutnode_to_rendernode(
    layout_node: &crate::layout::LayoutNode,
    layout_graph: &crate::layout::LayoutGraph,
    flat_graph: &crate::model::term_graph::TermGraph,
    function_declarations: &std::collections::HashMap<
        crate::model::function_declaration::FunctionDeclarationId,
        crate::model::function_declaration::FunctionDeclaration,
    >,
    // What a run has narrowed, `Known::nothing()` while none is going. Only
    // the literals are read from it: an anchor's rows are the ones its declared
    // type reserved, evaluated or not, so the node keeps its footprint.
    known: &crate::infer::Known,
    extra_offset: Vec3,
) -> RenderNode {
    let graph = &layout_graph.graph;
    // World centre of a node-local cell. Every part of a node — each anchor
    // row, the body — lives in its own cell, so placement goes through this
    // rather than nudging sub-meshes around inside a single cell.
    let cell = |x: i32, y: i32, z: i32| {
        cell_center_world(layout_node.pos + extra_offset + Vec3::new(x as f32, y as f32, z as f32))
    };
    let node = graph.nodes.get(&layout_node.node_id).unwrap();
    match node {
        // A named declaration: an opaque body in the colour of its type, as
        // long as its name needs, with the name printed on the top face and
        // the output anchor one cell behind the body's end.
        crate::model::node::ENode::Source {
            name,
            r#type,
            output_anchor,
        } => {
            let depth = crate::layout::name_body_cells(name);
            let output_world = cell(0, 0, depth);
            // Base type, literal stripped: what a Source produces comes from
            // the evaluation prompt, so a literal written on it types nothing
            // (see `infer::anchor_type`). The literal is still *shown* — it
            // travels beside the type as `output_value` and the output anchor
            // draws it as a line, the way any pinned value is drawn.
            //
            // Asked of the anchor rather than read off the declaration,
            // because this is the one node whose written literal was never the
            // value: what a run hands on is the prompt's answer, and that is
            // what a Source shows once it has actually handed it on. Not
            // before — the cell in front of the body carries what is waiting,
            // and this one carries what left.
            let output_eval_type =
                crate::infer::base_type_of(&crate::infer::graph_type_to_eval_type(r#type));
            let output_value = crate::infer::anchor_literal(flat_graph, output_anchor, known);
            // A Source has no input in the graph — it *is* where a value comes
            // in from outside — but the value has to be seen arriving
            // somewhere, so one is drawn all the same, one cell in front of the
            // body. It says the type the Source declares and carries the index
            // the evaluation prompt asks by, which is otherwise written down
            // nowhere: the index is the lateral order, so it is read off the
            // node's place rather than off the node.
            //
            // Visual only. It is not in `anchors`, so no edge can end here and
            // the pointer cannot pick it, and it claims no cell: layout Z=0 is
            // the scope's first row, so this hangs outside the volume, against
            // its front face.
            let input_world = cell(0, 0, -1);
            // The declared type and nothing else while the program is only
            // written: a literal on a Source is not the value it is evaluated
            // with — that one comes from the prompt — so what arrives here is
            // a value *of this type* and nothing narrower can be said.
            //
            // A run answered at the prompt says something narrower, and says it
            // *here* before it says it anywhere. This cell is outside the
            // volume, in front of the node: what stands on it is the world's
            // and not the program's, so it may carry the answer from the moment
            // it is typed — which is the whole of what step 0 shows. The output
            // anchor a body's length behind stays a band until the Source is
            // actually evaluated, and the distance between the two is the step
            // that carries the value across.
            //
            // Read off the node, `offered_at`, and not off an anchor: this cell
            // has none. That is why a value may stand on it without the program
            // having read it — there is nothing here for an edge to reach.
            let offered = known.offered_at(&layout_node.node_id);
            let input_strands = build_anchor_strands(
                &output_eval_type,
                offered.and_then(crate::infer::leaf_literal),
                offered,
                input_world,
                true,
                Lettering::Spelled,
            );
            // Where `build_anchor_strands` puts the first row's type letter: the
            // label leans into the anchor's own half, toward the body.
            let index_label_world = Vec3::new(
                input_world.x,
                input_world.y,
                input_world.z - ANCHOR_HALF_DEPTH,
            );
            let index_label = layout_graph
                .source_index(&layout_node.node_id)
                .map(|index| RenderLabel {
                    text: format!("[{}]", index),
                    // Neutral grey at the small size, the way a FunctionCall's
                    // input names are drawn: same kind of label, naming an
                    // anchor beside the letter that types it.
                    color: Color::srgb(0.5, 0.5, 0.5),
                    font_size: 12.0,
                    world_pos: index_label_world,
                    offset: Vec2::new(SOURCE_INDEX_LABEL_OFFSET_X, 0.0),
                });
            // The body wears the type it declares, in the same colour every
            // strand of that type wears. It used to have to force the alpha
            // back to 1.0 here, because strands were translucent and a body is
            // not; strands are opaque now, so the two agree by construction.
            let body_color = strand_color(&output_eval_type);
            let cell_x = LAYOUT_SCALE.x.abs();
            let cell_y = LAYOUT_SCALE.y.abs();
            let cell_z = LAYOUT_SCALE.z.abs();
            // Midpoint of the first and last body cell — no sign juggling, and
            // it stays right whichever way `LAYOUT_SCALE` points.
            let body_center = (cell(0, 0, 0) + cell(0, 0, depth - 1)) * 0.5;
            // One cell tall: a Source declares a leaf type, never a sum, so
            // its output is always a single row. Were that ever to change, the
            // body would follow `anchor_rows` the way a FunctionCall's far
            // face does.
            let body_size = Vec3::new(cell_x, cell_y, depth as f32 * cell_z);
            // Layout Y=0 is the body's upper bound and layout +Y is world −Y,
            // so the top face lies half a cell *above* the centre in world
            // terms.
            let top_y = body_center.y + cell_y * 0.5;
            // `emissive` would be dead weight here: for an unlit material the
            // shader skips the lighting pass that would add it. Unlit also
            // keeps the body and the name face lying on it identical by
            // construction — a lit body would shade with its orientation while
            // the face, being its own entity at its own angle, would not.
            let body_material = StandardMaterial {
                base_color: body_color,
                unlit: true,
                ..default()
            };
            let name_face = RenderTextFace {
                // Width runs along local +X, height along local +Y; the
                // rotation below maps those onto world −Z and −X.
                mesh: Rectangle::new(depth as f32 * cell_z, cell_x).mesh().build(),
                transform: Transform {
                    translation: Vec3::new(body_center.x, top_y + BODY_FACE_LIFT, body_center.z),
                    // A `Rectangle` is built in the XY plane. Laying it flat
                    // alone (the `x` term) would run the text along world +X,
                    // which slants away from the viewer; the `y` term turns it
                    // so the text reads along world −Z, i.e. layout +Z, which
                    // the projection draws exactly left to right. What is left
                    // — local +Y onto world −X — keeps the basis right-handed,
                    // which is what stops the glyphs coming out mirrored.
                    rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)
                        * Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                    scale: Vec3::ONE,
                },
                material: face_material(),
                text: name.clone(),
                cells: depth as u32,
                background: body_color,
            };
            RenderNode {
                node: Some(RenderObject {
                    mesh: Cuboid::new(body_size.x, body_size.y, body_size.z)
                        .mesh()
                        .build(),
                    material: body_material,
                    transform: Transform::from_translation(body_center),
                }),
                anchors: std::collections::HashMap::from([(
                    output_anchor.clone(),
                    RenderAnchor {
                        pick_center: output_world,
                        strands: build_anchor_strands(
                            &output_eval_type,
                            output_value.as_deref(),
                            // A Source declares one leaf, so there is no second
                            // row a run could drop — and the literal beside it
                            // already turns the one it has into a line.
                            None,
                            output_world,
                            false,
                            Lettering::Spelled,
                        ),
                        plain_body: None,
                    },
                )]),
                // The drawn-only input anchor belongs to the node itself, not
                // to an anchor of it — the same place a Pattern's band lives.
                strands: input_strands,
                objects: vec![],
                // The name is on the body now, and the type is in the body's
                // colour and at the output anchor — the index is the one thing
                // left that has to be written beside the node.
                labels: index_label.into_iter().collect(),
                text_faces: vec![name_face],
            }
        }
        // A literal: the line its value travels on, starting at a point in the
        // body cell `0|0` and leaving through the output anchor at `0|1`.
        crate::model::node::ENode::Constant {
            r#type,
            output_anchor,
        } => {
            let body_world = cell(0, 0, 0);
            let output_world = cell(0, 0, 1);
            let output_eval_type = crate::infer::graph_type_to_eval_type(r#type);
            let output_value = crate::layout::value_of_etype(r#type);
            let color = strand_color(&output_eval_type);
            // The half of the body cell that faces the anchor, so the segment
            // continues the value line the anchor draws into its own half
            // (`build_anchor_strands`) and the two meet exactly at the cell
            // boundary. Every constant has that line now, `none` included —
            // its value is its type, so the anchor draws it as a value like
            // any other.
            let half_depth = LAYOUT_SCALE.z.abs() * 0.5;
            let line_center = Vec3::new(
                body_world.x,
                body_world.y,
                // Layout +Z is world −Z, so the anchor side of the cell is the
                // one at the lower world Z.
                body_world.z - half_depth * 0.5,
            );
            let line = RenderObject {
                mesh: Cuboid::new(0.0, STRAND_LINE_THICKNESS, half_depth)
                    .mesh()
                    .build(),
                material: StandardMaterial {
                    base_color: color,
                    cull_mode: None,
                    unlit: true,
                    ..default()
                },
                transform: Transform::from_translation(line_center),
            };
            // Where the value begins. White rather than the type's colour: the
            // line already carries the type along its whole length, and what
            // the point says is that this is the end of it — nothing is behind
            // a constant.
            let dot = RenderObject {
                mesh: Sphere::new(CONSTANT_DOT_RADIUS).mesh().build(),
                material: StandardMaterial {
                    base_color: Color::WHITE,
                    unlit: true,
                    ..default()
                },
                transform: Transform::from_translation(body_world),
            };
            RenderNode {
                node: None,
                anchors: std::collections::HashMap::from([(
                    output_anchor.clone(),
                    RenderAnchor {
                        pick_center: output_world,
                        strands: build_anchor_strands(
                            &output_eval_type,
                            output_value.as_deref(),
                            // A Constant declares one leaf and is its own
                            // value, so a run neither drops a row here nor
                            // narrows one: it has always been the line it is.
                            None,
                            output_world,
                            false,
                            Lettering::Spelled,
                        ),
                        plain_body: None,
                    },
                )]),
                strands: vec![],
                objects: vec![line, dot],
                // A literal draws itself: the value hangs off the output
                // anchor, so there is nothing for a body label to add.
                labels: vec![],
                text_faces: vec![],
            }
        }
        // Input anchor at `0|0`, a gap at `0|1`, the target type at `0|2`,
        // output at `0|3` — a Match's rhythm with one arm, which is how a cast
        // reads. Reads, and no more than that: an arm takes a share of the
        // band and leaves the rest, a cast turns the whole band into one of
        // two things.
        crate::model::node::ENode::TypeCast {
            r#type,
            input_anchor,
            output_anchor,
        } => {
            let input_world = cell(0, 0, 0);
            let body_world = cast_band_world(layout_node, extra_offset);
            let output_world = cell(0, 0, crate::layout::CAST_OUTPUT_Z);
            // The output reflects a cast that can fail as `Sum(target, none)`,
            // and stays `Pending` while nothing flows in at all, or while
            // nothing has been cast *to*; that logic lives in
            // `infer::type_cast_output_type`, which counts up `cast_kind` row
            // by row — the same call the strands are drawn from, so the row
            // this anchor grows is the row a strand is aimed at. Nothing falls back to
            // the declared type here: what the cast will produce is a question
            // about what arrives, not about what it aims at.
            let output_eval_type =
                crate::infer::anchor_type(flat_graph, output_anchor, function_declarations)
                    .unwrap_or(crate::infer::EType::Pending);
            let input_eval_type =
                crate::infer::incoming_anchor_type(flat_graph, input_anchor, function_declarations);
            let elim_value = r#type.as_ref().and_then(crate::layout::value_of_etype);
            // Built before the target cell, because whether that cell speaks
            // depends on whether this one does.
            let output_anchor_render = typed_anchor(
                &output_eval_type,
                elim_value.as_deref(),
                known.at_output(flat_graph, output_anchor),
                output_world,
                false,
                Lettering::Spelled,
            );
            // Drawn as its arm's band, exactly as a Pattern is — the shape is
            // shared, the statement is not. A pattern names what it takes out
            // of the band; a cast names what it turns the whole band into.
            //
            // A Pattern can stay silent because its BranchSource stands right
            // behind it wearing the same type and saying so. A cast has no such
            // neighbour: with nothing wired in, its input is grey and its output
            // is `Pending`, which claims no row and so draws grey too — and the
            // one thing on screen that knows the type would be the only thing
            // not saying it. So the target cell speaks exactly as long as
            // nothing downstream of it does.
            let (strands, objects) = declared_type_cell(
                r#type.as_ref(),
                body_world,
                if output_anchor_render.strands.is_empty() {
                    Lettering::Spelled
                } else {
                    Lettering::Silent
                },
            );
            RenderNode {
                // No body of its own — the band *is* the node, the way a
                // Pattern's is. The node entity is still spawned for picking.
                node: None,
                anchors: std::collections::HashMap::from([
                    (
                        input_anchor.clone(),
                        match input_eval_type {
                            // A typecast constrains nothing, so its input shows
                            // whatever arrives — and a neutral body when idle.
                            Some(t) => {
                                // The row a run settled on, and its word read
                                // off that row rather than fetched upstream: an
                                // unevaluated cast input shows the type that
                                // arrives and never the literal behind it, and
                                // that is not a run's business to change.
                                let taken = known.at_input(flat_graph, input_anchor);
                                typed_anchor(
                                    &t,
                                    taken.and_then(crate::infer::leaf_literal),
                                    taken,
                                    input_world,
                                    true,
                                    Lettering::Spelled,
                                )
                            }
                            None => RenderAnchor {
                                pick_center: input_world,
                                strands: vec![],
                                plain_body: Some(plain_anchor_body(input_world, true)),
                            },
                        },
                    ),
                    (output_anchor.clone(), output_anchor_render),
                ]),
                strands,
                objects,
                labels: vec![],
                text_faces: vec![],
            }
        }
        // Input anchors along `i|0`, body spanning the full width from `z=1` to
        // as deep as the function's name needs, output one cell behind it.
        crate::model::node::ENode::FunctionCall {
            function_declaration_id,
            input_anchors,
            output_anchor,
        } => {
            let function_declaration = function_declarations
                .get(function_declaration_id)
                .expect("function call refers to unknown function declaration");
            let width = input_anchors.len().max(1) as i32;
            // The same rule the shape uses (`layout::node_shape`), read off the
            // same name — the two are hand-written copies of one cell map and
            // must not drift.
            let depth = crate::layout::name_body_cells(&function_declaration.name);
            let cell_x = LAYOUT_SCALE.x.abs();
            let cell_y = LAYOUT_SCALE.y.abs();
            let cell_z = LAYOUT_SCALE.z.abs();
            // The body is a frustum spanning what goes in to what comes out:
            // the near face covers the whole input anchor block, the far face
            // matches the output anchor. Both anchor stacks grow downward from
            // row 0, so the two faces are top aligned, not centred on one
            // another — which is why this builds explicit corners.
            let input_rows = input_anchors
                .iter()
                .map(|a| crate::infer::anchor_rows(flat_graph, a, function_declarations))
                .max()
                .unwrap_or(1);
            let output_rows =
                crate::infer::anchor_rows(flat_graph, output_anchor, function_declarations);
            let first_col = cell(0, 0, 0);
            let last_col = cell(width - 1, 0, 0);
            // Upper edge of row 0, shared by both faces.
            let top_y = first_col.y + cell_y * 0.5;
            // Front face of the first body cell and back face of the last.
            let near_z = cell(0, 0, 1).z + cell_z * 0.5;
            let far_z = cell(0, 0, depth).z - cell_z * 0.5;
            let body_center = Vec3::new(
                (first_col.x + last_col.x) * 0.5,
                top_y - input_rows.max(output_rows) as f32 * cell_y * 0.5,
                (near_z + far_z) * 0.5,
            );
            let quad = |x_min: f32, x_max: f32, y_top: f32, y_bottom: f32, z: f32| {
                // CCW seen from +Z, i.e. from in front of the near face.
                [
                    Vec3::new(x_min, y_bottom, z) - body_center,
                    Vec3::new(x_max, y_bottom, z) - body_center,
                    Vec3::new(x_max, y_top, z) - body_center,
                    Vec3::new(x_min, y_top, z) - body_center,
                ]
                .map(|v| v.to_array())
            };
            let base_quad = quad(
                first_col.x - cell_x * 0.5,
                last_col.x + cell_x * 0.5,
                top_y,
                top_y - input_rows as f32 * cell_y,
                near_z,
            );
            let top_quad = quad(
                first_col.x - cell_x * 0.5,
                first_col.x + cell_x * 0.5,
                top_y,
                top_y - output_rows as f32 * cell_y,
                far_z,
            );
            let output_world = cell(0, 0, depth + 1);
            let body_color = Color::srgb(0.5, 0.9, 1.0);
            // The function's name, printed along the body the way a Source's is
            // — same rectangle, same rotation, same lift.
            //
            // Over column x=0 alone, and that is not only what "along the x=0
            // row" asks for: the body is a frustum that tapers in X toward the
            // output, so x=0 is the one column whose roof line both faces share
            // at `top_y`. A face spanning the full input width would float over
            // a sloped roof everywhere else.
            let name_face = RenderTextFace {
                mesh: Rectangle::new(depth as f32 * cell_z, cell_x).mesh().build(),
                transform: Transform {
                    translation: Vec3::new(
                        first_col.x,
                        top_y + BODY_FACE_LIFT,
                        (near_z + far_z) * 0.5,
                    ),
                    rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)
                        * Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                    scale: Vec3::ONE,
                },
                material: face_material(),
                text: function_declaration.name.clone(),
                cells: depth as u32,
                background: body_color,
            };
            RenderNode {
                node: Some(RenderObject {
                    mesh: crate::mesh::frustum_8pt_mesh(base_quad, top_quad),
                    material: StandardMaterial {
                        base_color: body_color,
                        unlit: true,
                        ..default()
                    },
                    transform: Transform::from_translation(body_center),
                }),
                anchors: input_anchors
                    .iter()
                    .enumerate()
                    .map(|(i_anchor, anchor_id)| {
                        let input_world = cell(i_anchor as i32, 0, 0);
                        // A parameter that constrains nothing — `=` and `!=`
                        // take any two values — reads like a TypeCast input:
                        // whatever arrives, and a neutral body when idle.
                        let declared = function_declaration
                            .inputs
                            .get(i_anchor)
                            .and_then(|param| param.r#type.clone());
                        let shown = declared.or_else(|| {
                            crate::infer::incoming_anchor_type(
                                flat_graph,
                                anchor_id,
                                function_declarations,
                            )
                        });
                        // A parameter owns no literal — a declaration says
                        // what may arrive, never what does — and a written
                        // graph leaves it at that. A run does not: what arrived
                        // is settled, so the row it arrived on carries its
                        // word, read off that row and fetched from nowhere
                        // else.
                        let arriving = known.at_input(flat_graph, anchor_id);
                        (
                            anchor_id.clone(),
                            match shown {
                                Some(t) => typed_anchor(
                                    &t,
                                    arriving.and_then(crate::infer::leaf_literal),
                                    arriving,
                                    input_world,
                                    true,
                                    Lettering::Spelled,
                                ),
                                None => RenderAnchor {
                                    pick_center: input_world,
                                    strands: vec![],
                                    plain_body: Some(plain_anchor_body(input_world, true)),
                                },
                            },
                        )
                    })
                    .chain([(
                        output_anchor.clone(),
                        typed_anchor(
                            &function_declaration.output_type,
                            // A call as written carries no literal: its output
                            // is whatever the declaration promises. A call that
                            // has *run* carries one, and then the promise is
                            // beside the point — `*` that produced `30` is a
                            // `30` and draws as the line one is.
                            crate::infer::anchor_literal(flat_graph, output_anchor, known)
                                .as_deref(),
                            known.at_output(flat_graph, output_anchor),
                            output_world,
                            false,
                            Lettering::Spelled,
                        ),
                    )])
                    .collect(),
                strands: vec![],
                objects: vec![],
                // The function's name is printed on the body now, not floated
                // over its centre — what is left beside the node is the name of
                // each parameter, at the anchor it belongs to.
                labels: input_anchors
                    .iter()
                    .enumerate()
                    .filter_map(|(i_anchor, _)| {
                        let name = function_declaration.inputs.get(i_anchor)?.name.clone();
                        Some(RenderLabel {
                            text: name,
                            // Neutral grey, subordinate to the name on the body
                            // and to the centred type letter.
                            color: Color::srgb(0.5, 0.5, 0.5),
                            font_size: 12.0,
                            world_pos: cell(i_anchor as i32, 0, 0),
                            // Nudge down so the type letter stays free.
                            offset: Vec2::new(0.0, 14.0),
                        })
                    })
                    .collect(),
                text_faces: vec![name_face],
            }
        }
        // Nothing but an input anchor, sitting alone on the scope's last Z row.
        // It constrains nothing, so it takes the shape of whatever type arrives
        // — a neutral body when idle — but it takes it silently, and the
        // program's Sink alone speaks again on the far side of the back wall.
        crate::model::node::ENode::Sink { input_anchor } => {
            let input_world = cell(0, 0, 0);
            let incoming =
                crate::infer::incoming_anchor_type(flat_graph, input_anchor, function_declarations);
            // A Sink constrains nothing, so it owns no type to pin a literal
            // to — but it is where a branch's value comes to rest, and a value
            // is drawn as the value. So the literal is read off the anchor
            // upstream, the one that does own it, and the Sink shows what
            // actually arrived rather than the shape of what might have.
            let incoming_value =
                crate::infer::incoming_anchor_literal(flat_graph, input_anchor, known);
            // The Sink's own anchor says nothing in words. Whatever reaches it
            // was already named by the anchor it left — a Sink adds no step, it
            // only ends one — and inside a branch it stands two cells from the
            // Match's output, which names the same thing again.
            //
            // The program's Sink is the exception, and it says its piece on the
            // far side instead. See `outgoing_strands`.
            let anchor = match incoming.as_ref() {
                Some(t) => typed_anchor(
                    t,
                    incoming_value.as_deref(),
                    known.at_input(flat_graph, input_anchor),
                    input_world,
                    true,
                    Lettering::Silent,
                ),
                None => RenderAnchor {
                    pick_center: input_world,
                    strands: vec![],
                    plain_body: Some(plain_anchor_body(input_world, true)),
                },
            };
            // Mirror of the drawn-only input a Source hangs against the front
            // face: the program's result, named once, where the program ends.
            //
            // Visual only, like that one. It is not in `anchors`, so no edge can
            // end here and the pointer cannot pick it, and it claims no cell —
            // `grid_bounds` stops at the Sink's own row, so this hangs outside
            // the volume against its back face. Drawn as an *output* so the band
            // fills the half of its cell that faces the volume and sits flush
            // against that face, the way the Source's sits flush against the
            // front one.
            //
            // Only the program's Sink. A branch's Sink is followed by its
            // Match's output two cells on, which already names the same value,
            // and the cell right behind it belongs to that Match's envelope.
            let outgoing_world = cell(0, 0, 1);
            let is_program_sink = layout_node.node_id == flat_graph.sink_node_id;
            let outgoing_strands = if is_program_sink {
                incoming
                    .as_ref()
                    .map(|t| {
                        build_anchor_strands(
                            t,
                            incoming_value.as_deref(),
                            known.at_input(flat_graph, input_anchor),
                            outgoing_world,
                            false,
                            Lettering::Spelled,
                        )
                    })
                    .unwrap_or_default()
            } else {
                vec![]
            };
            // The exit is always drawn, grey until there is something to say —
            // it belongs to the program and not to its state. A program has an
            // end whether or not anything reaches it yet, and a wall with
            // nothing on it reads as a program with nothing wired rather than as
            // a program without an end.
            //
            // Grey covers both ways of having nothing: no edge into the Sink,
            // and an edge whose type is still `Pending` — which claims no row,
            // so it builds no strands either. Asking the strands rather than the
            // type is what folds the two together, the same way
            // `declared_type_cell` folds them.
            let outgoing_objects: Vec<RenderObject> = (is_program_sink
                && outgoing_strands.is_empty())
            .then(|| plain_anchor_body(outgoing_world, false))
            .into_iter()
            .collect();
            RenderNode {
                node: None,
                anchors: std::collections::HashMap::from([(input_anchor.clone(), anchor)]),
                strands: outgoing_strands,
                objects: outgoing_objects,
                labels: vec![],
                text_faces: vec![],
            }
        }
        // A Pattern declares the type its arm matches and fixes its branch's
        // row, but owns no anchor — the branch draws its value from its own
        // BranchSource, behind it in the branch volume. It is drawn as that
        // type's band, like an input anchor: the value it accepts is what the
        // band names. Its first cell is the gap it holds open and stays empty.
        //
        // Silent unconditionally, unlike a cast's target: the BranchSource is
        // one cell behind it wearing the same type and naming it, whatever the
        // Match's input happens to be.
        crate::model::node::ENode::Pattern { r#type, .. } => {
            let (strands, objects) = declared_type_cell(
                r#type.as_ref(),
                pattern_band_world(layout_node, extra_offset),
                Lettering::Silent,
            );
            RenderNode {
                node: None,
                anchors: std::collections::HashMap::new(),
                strands,
                objects,
                labels: vec![],
                text_faces: vec![],
            }
        }
        // Mirror of the Sink: a single output anchor at the branch origin. Its
        // type is the owning Pattern's, resolved through `infer::anchor_type`.
        crate::model::node::ENode::BranchSource { output_anchor, .. } => {
            let output_world = cell(0, 0, 0);
            // Both type and literal come from the owning Pattern, so the
            // source shows exactly what its arm matched.
            let output_eval_type =
                crate::infer::anchor_type(flat_graph, output_anchor, function_declarations)
                    .unwrap_or(crate::infer::EType::Pending);
            let output_value = crate::infer::anchor_literal(flat_graph, output_anchor, known);
            RenderNode {
                node: None,
                anchors: std::collections::HashMap::from([(
                    output_anchor.clone(),
                    typed_anchor(
                        &output_eval_type,
                        output_value.as_deref(),
                        known.at_output(flat_graph, output_anchor),
                        output_world,
                        false,
                        Lettering::Spelled,
                    ),
                )]),
                strands: vec![],
                objects: vec![],
                labels: vec![],
                text_faces: vec![],
            }
        }
        // The branch source's twin, and drawn like it — one anchor on the
        // entry row, no body, because a Tunnel declares nothing either.
        //
        // What it has and the branch source has not is a second anchor, and it
        // sits *outside*: one cell in front of the entry row, against the
        // scope's front face, exactly where a Source draws the arrival it only
        // mimes. Here it is the real thing — in `anchors`, so the pointer
        // picks it and an edge from the enclosing graph ends on it. That is
        // the whole difference between the two, and it is the whole point of
        // the node.
        crate::model::node::ENode::Tunnel {
            input_anchor,
            output_anchor,
        } => {
            let input_world = cell(0, 0, -1);
            let output_world = cell(0, 0, 0);
            // One type for both ends: a Tunnel borrows what arrives and hands
            // it on unchanged, so drawing them from the same lookup is what
            // makes the pass-through visible rather than merely true.
            let eval_type =
                crate::infer::anchor_type(flat_graph, output_anchor, function_declarations)
                    .unwrap_or(crate::infer::EType::Pending);
            let value = crate::infer::anchor_literal(flat_graph, output_anchor, known);
            RenderNode {
                node: None,
                anchors: std::collections::HashMap::from([
                    (
                        input_anchor.clone(),
                        typed_anchor(
                            &eval_type,
                            value.as_deref(),
                            known.at_output(flat_graph, output_anchor),
                            input_world,
                            true,
                            Lettering::Spelled,
                        ),
                    ),
                    (
                        output_anchor.clone(),
                        typed_anchor(
                            &eval_type,
                            value.as_deref(),
                            known.at_output(flat_graph, output_anchor),
                            output_world,
                            false,
                            Lettering::Spelled,
                        ),
                    ),
                ]),
                strands: vec![],
                objects: vec![],
                labels: vec![],
                text_faces: vec![],
            }
        }
        crate::model::node::ENode::Match {
            patterns,
            input_anchor,
            output_anchor,
        } => {
            // The Match draws no body of its own: each Pattern is its own type
            // band, and the branches speak for themselves. All it contributes
            // are its two anchors.
            //
            // Input anchor owns the Match's own cell at local 0|0.
            let input_world = cell(0, 0, 0);
            let incoming =
                crate::infer::incoming_anchor_type(flat_graph, input_anchor, function_declarations);
            // A Match constrains its input no more than a Sink does, so the
            // literal comes from upstream for the same reason — and here it
            // decides the shape the arms are joined to: a value arriving at a
            // Match is a line, and an arm that accepts a whole type is a band,
            // which is what makes the link between them widen along its length.
            let incoming_value =
                crate::infer::incoming_anchor_literal(flat_graph, input_anchor, known);
            // The output owns its own cell directly behind the deepest branch;
            // `match_output_z` decides which one. Its type is the union of the
            // branch types, or `Pending` while the inferer cannot decide it.
            let out_world = cell(0, 0, layout_graph.match_output_z(patterns));
            let output_eval_type =
                crate::infer::anchor_type(flat_graph, output_anchor, function_declarations)
                    .unwrap_or(crate::infer::EType::Pending);
            RenderNode {
                node: None,
                anchors: std::collections::HashMap::from([
                    (
                        input_anchor.clone(),
                        match incoming {
                            Some(t) => typed_anchor(
                                &t,
                                incoming_value.as_deref(),
                                known.at_input(flat_graph, input_anchor),
                                input_world,
                                true,
                                Lettering::Spelled,
                            ),
                            None => RenderAnchor {
                                pick_center: input_world,
                                strands: vec![],
                                plain_body: Some(plain_anchor_body(input_world, true)),
                            },
                        },
                    ),
                    (
                        output_anchor.clone(),
                        typed_anchor(
                            &output_eval_type,
                            crate::infer::anchor_literal(flat_graph, output_anchor, known)
                                .as_deref(),
                            known.at_output(flat_graph, output_anchor),
                            out_world,
                            false,
                            Lettering::Spelled,
                        ),
                    ),
                ]),
                strands: vec![],
                objects: vec![],
                labels: vec![],
                text_faces: vec![],
            }
        }
        crate::model::node::ENode::Root { .. } => {
            unreachable!("Root node has no layout position and is never rendered directly")
        }
    }
}

pub fn emissive_color(color: Color) -> LinearRgba {
    let c = color.to_linear();
    LinearRgba::new(c.red * 0.15, c.green * 0.15, c.blue * 0.15, 1.0)
}

pub fn label_for_node(
    node: &crate::model::node::ENode,
    function_declarations: &std::collections::HashMap<
        crate::model::function_declaration::FunctionDeclarationId,
        crate::model::function_declaration::FunctionDeclaration,
    >,
) -> String {
    match node {
        crate::model::node::ENode::Sink { .. } => "sink".to_string(),
        crate::model::node::ENode::FunctionCall {
            function_declaration_id,
            ..
        } => function_declarations
            .get(&function_declaration_id)
            .unwrap()
            .name
            .to_string(),
        crate::model::node::ENode::Constant { r#type, .. } => r#type.to_string(),
        crate::model::node::ENode::Source { name, r#type, .. } => {
            format!("{}: {}", name, r#type.to_string())
        }
        crate::model::node::ENode::Match { .. } => "match".to_string(),
        // The two that may not have been typed yet. A question mark rather than
        // a type name, because there is no type to name.
        crate::model::node::ENode::TypeCast { r#type, .. }
        | crate::model::node::ENode::Pattern { r#type, .. } => r#type
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| "?".to_string()),
        crate::model::node::ENode::BranchSource { .. } => "branch source".to_string(),
        crate::model::node::ENode::Tunnel { .. } => "tunnel".to_string(),
        crate::model::node::ENode::Root { .. } => "root".to_string(),
    }
}
