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

/// Edge thickness of the selection caret's cell outline, in world units.
const CARET_EDGE_THICKNESS: f32 = CELL / 60.0;

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

/// A wire-like band standing in for a node body. Carries an `EdgeMaterial`
/// instead of a `StandardMaterial`, so the spawner handles it separately from
/// `RenderObject`.
pub struct RenderBand {
    pub mesh: Mesh,
    pub color: Color,
    /// Which prebuilt marquee texture names the type (`edge::EdgeLabelTextures`).
    pub kind: crate::edge::LeafKind,
}

pub struct RenderNode {
    /// `None` for nodes drawn purely as bands or markers; the node entity is
    /// still spawned so picking and selection keep working.
    pub node: Option<RenderObject>,
    pub anchors: std::collections::HashMap<crate::model::anchor::Id, RenderAnchor>,
    /// Type markers belonging to the node itself rather than to an anchor. A
    /// Pattern uses this: it declares the type its arm matches and is drawn as
    /// that type's band, but owns no anchor to hang it on.
    pub markers: Vec<RenderTypeMarker>,
    pub bands: Vec<RenderBand>,
    pub labels: Vec<RenderLabel>,
}

pub struct RenderAnchor {
    /// World point used for screen-space hover picking and as the edge
    /// endpoint. Sits at the centre of the anchor cuboid (or cuboid stack).
    pub pick_center: Vec3,
    pub type_markers: Vec<RenderTypeMarker>,
    /// Neutral cuboid for anchors that carry no type markers (Sink,
    /// Match), so they stay visible and pickable.
    pub plain_body: Option<RenderObject>,
}

pub struct RenderTypeMarker {
    /// The type band and its letter. Absent when the leaf carries a literal:
    /// the value line then replaces the band entirely, matching how a
    /// value-carrying edge drops its band for a hairline.
    pub rect: Option<RenderObject>,
    pub label: Option<RenderLabel>,
    /// Present iff this leaf's anchor carries an AST-level literal. A thin
    /// coloured segment spanning the anchor's full depth, at its Y-middle.
    pub value_line: Option<RenderObject>,
    /// Present alongside `value_line`. Text is the literal itself, projected
    /// into screen space past the line's tip.
    pub value_label: Option<RenderLabel>,
}

pub struct RenderLabel {
    pub text: String,
    pub color: Color,
    pub font_size: f32,
    pub world_pos: Vec3,
    pub offset: Vec2,
}

const TYPE_MARKER_ALPHA: f32 = 0.6;
/// Height of one anchor row: exactly one cell, since an anchor claims one
/// cell per sum-type member. The marker mesh fills the cell in Y but stays
/// slim in X and Z, so a sum type reads as an unbroken band while the graph
/// keeps its airy look.
pub const TYPE_MARKER_Y_STEP: f32 = CELL;
const TYPE_MARKER_HALF_DEPTH: f32 = CELL / 4.0;
/// Full Z-depth of an anchor cuboid: half its cell.
///
/// An anchor occupies the half of its cell that faces the node body, so it
/// visibly hangs off the thing it belongs to instead of floating mid-cell. In
/// cell-local terms an input takes `0.5..1` and an output `0..0.5`; since
/// layout +Z is world −Z, that is the −Z half for inputs and the +Z half for
/// outputs. Both meet the cell centre, which is where their edges attach.
const ANCHOR_DEPTH: f32 = 2.0 * TYPE_MARKER_HALF_DEPTH;
/// X-width of an anchor cuboid. Shared with the slab nodes so anchor and node
/// line up exactly in X.
pub const ANCHOR_X: f32 = CELL * 0.075;
/// Y-thickness of the gizmo line drawn in the far 2/3 of a value-carrying
/// type marker.
const VALUE_LINE_THICKNESS: f32 = CELL * 0.02;
/// World-space padding between the tip of the gizmo line and the value
/// label's projection point.
const VALUE_LABEL_Z_PADDING: f32 = CELL / 30.0;

/// Sort key that fixes the vertical order of type-marker rectangles.
/// Returns `None` for variants that should not render a rectangle.
fn type_marker_order(t: &crate::infer::EType) -> Option<u8> {
    match t {
        crate::infer::EType::Bool(..) => Some(0),
        crate::infer::EType::Char(..) => Some(1),
        crate::infer::EType::Int(..) => Some(2),
        crate::infer::EType::String(..) => Some(3),
        crate::infer::EType::None => Some(4),
        _ => None,
    }
}

fn type_marker_letter(t: &crate::infer::EType) -> &'static str {
    match t {
        crate::infer::EType::Bool(..) => "b",
        crate::infer::EType::Char(..) => "c",
        crate::infer::EType::Int(..) => "i",
        crate::infer::EType::String(..) => "s",
        crate::infer::EType::None => "n",
        _ => "?",
    }
}

pub fn type_marker_color(t: &crate::infer::EType) -> Color {
    match t {
        crate::infer::EType::Bool(..) => Color::srgba(0.65, 0.30, 0.95, TYPE_MARKER_ALPHA),
        crate::infer::EType::Char(..) => Color::srgba(0.30, 0.90, 0.40, TYPE_MARKER_ALPHA),
        crate::infer::EType::Int(..) => Color::srgba(0.40, 0.70, 1.00, TYPE_MARKER_ALPHA),
        crate::infer::EType::String(..) => Color::srgba(1.00, 0.90, 0.30, TYPE_MARKER_ALPHA),
        crate::infer::EType::None => Color::srgba(0.95, 0.30, 0.30, TYPE_MARKER_ALPHA),
        _ => Color::srgba(0.5, 0.5, 0.5, TYPE_MARKER_ALPHA),
    }
}

/// World-space Y offset of leaf row `index` relative to the anchor's own row.
///
/// Rows run from the anchor's cell along +Y in layout space, which is downward
/// in world space — so the offset is negative and grows with the index. The
/// leaves are no longer centred on the anchor: an anchor's address is its
/// first row.
///
/// Both the marker stack (`build_type_markers`) and the edge ribbons
/// (`spawn_ast_nodes`) go through this, so they cannot drift apart.
pub fn leaf_row_offset(index: usize) -> f32 {
    index as f32 * LAYOUT_SCALE.y.signum() * TYPE_MARKER_Y_STEP
}
/// The row-claiming leaves of `t`, sorted into the fixed bottom-to-top stack
/// order (none at bottom → bool at top).
///
/// Which leaves claim a row is decided by `infer::row_leaves`, so the render
/// stack and the cell addressing can never disagree about how tall an anchor
/// is; this only fixes their order.
pub fn ordered_supported_leaves(t: &crate::infer::EType) -> Vec<crate::infer::EType> {
    let mut leaves = crate::infer::row_leaves(t);
    leaves.sort_by_key(|leaf| std::cmp::Reverse(type_marker_order(leaf).unwrap_or(u8::MAX)));
    leaves
}

/// Body of a source node, rendered as a wire band rather than a solid slab:
/// it *is* the start of the value's path, and the marquee names its type.
///
/// Reuses the edge ribbon (`edge::build_ribbon_mesh`) with a hand-built
/// straight curve through the body cell — `EdgeCurve`'s control points are
/// public precisely so a caller can bypass `from_endpoints`, whose ≥1.5-unit
/// handles would bulge a same-cell span.
pub fn source_body_curve(cell_center: Vec3) -> crate::edge::EdgeCurve {
    let half = LAYOUT_SCALE.z.abs() * 0.5;
    let front = cell_center + Vec3::new(0.0, 0.0, half);
    let back = cell_center + Vec3::new(0.0, 0.0, -half);
    crate::edge::EdgeCurve {
        p0: front,
        p1: front.lerp(back, 1.0 / 3.0),
        p2: front.lerp(back, 2.0 / 3.0),
        p3: back,
    }
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
/// `ast_value` is the AST-level literal on the anchor's type, if any. When
/// present the band is dropped entirely and the leaf is drawn as a single
/// thin line plus the literal — a value is shown as the value, not as its
/// type. In practice value-carrying nodes have a single leaf, so this only
/// fires on one marker per anchor.
fn build_type_markers(
    t: &crate::infer::EType,
    ast_value: Option<&str>,
    anchor_world_pos: Vec3,
    is_input: bool,
) -> Vec<RenderTypeMarker> {
    let leaves = ordered_supported_leaves(t);
    if leaves.is_empty() {
        return vec![];
    }
    // Direction the anchor body extends from the cell centre: toward the node,
    // i.e. −Z for an input (cell-local 0.5..1) and +Z for an output (0..0.5).
    let sign = if is_input { -1.0 } else { 1.0 };
    let full_depth = ANCHOR_DEPTH;
    // The cell centre is the anchor's outward face — the point its edge meets —
    // so every span is measured from there into the anchor's own half.
    let full_rect_z_center = anchor_world_pos.z + sign * TYPE_MARKER_HALF_DEPTH;
    let line_tip_z = anchor_world_pos.z + sign * full_depth;

    leaves
        .into_iter()
        .enumerate()
        .map(|(k, leaf)| {
            let y_center = anchor_world_pos.y + leaf_row_offset(k);
            let color = type_marker_color(&leaf);
            let letter = type_marker_letter(&leaf).to_string();
            let center = Vec3::new(anchor_world_pos.x, y_center, full_rect_z_center);

            if let Some(value) = ast_value {
                // A leaf pinned to a literal is drawn as that literal and
                // nothing else: one thin line across the anchor's full depth,
                // its colour carrying the type. No band, no type letter — the
                // same choice the edge shader makes for value-carrying edges.
                let label_world = Vec3::new(
                    anchor_world_pos.x,
                    y_center,
                    line_tip_z + sign * VALUE_LABEL_Z_PADDING,
                );
                RenderTypeMarker {
                    rect: None,
                    label: None,
                    value_line: Some(RenderObject {
                        mesh: Cuboid::new(0.0, VALUE_LINE_THICKNESS, full_depth)
                            .mesh()
                            .build(),
                        material: StandardMaterial {
                            base_color: color,
                            alpha_mode: AlphaMode::Blend,
                            cull_mode: None,
                            unlit: true,
                            ..default()
                        },
                        transform: Transform::from_translation(center),
                    }),
                    value_label: Some(RenderLabel {
                        text: value.to_string(),
                        color: Color::WHITE,
                        font_size: 14.0,
                        world_pos: label_world,
                        offset: Vec2::ZERO,
                    }),
                }
            } else {
                RenderTypeMarker {
                    rect: Some(RenderObject {
                        mesh: Cuboid::new(ANCHOR_X, TYPE_MARKER_Y_STEP, full_depth)
                            .mesh()
                            .build(),
                        material: StandardMaterial {
                            base_color: color,
                            alpha_mode: AlphaMode::Blend,
                            cull_mode: None,
                            unlit: true,
                            ..default()
                        },
                        transform: Transform::from_translation(center),
                    }),
                    label: Some(RenderLabel {
                        text: letter,
                        color: Color::WHITE,
                        font_size: 14.0,
                        world_pos: center,
                        offset: Vec2::ZERO,
                    }),
                    value_line: None,
                    value_label: None,
                }
            }
        })
        .collect()
}

/// Anchor rendered from an inferred type: a type-marker stack when the type
/// has renderable leaves, otherwise the neutral grey body. `Pending` has no
/// leaves, so an output whose type the inferer cannot decide yet reads exactly
/// like the typeless (unconstrained) inputs.
fn typed_anchor(
    t: &crate::infer::EType,
    ast_value: Option<&str>,
    cell_center: Vec3,
    is_input: bool,
) -> RenderAnchor {
    let type_markers = build_type_markers(t, ast_value, cell_center, is_input);
    RenderAnchor {
        // The cell centre is the anchor's outward face, so edges meet it there
        // no matter how many rows the anchor spans.
        pick_center: cell_center,
        plain_body: type_markers
            .is_empty()
            .then(|| plain_anchor_body(cell_center, is_input)),
        type_markers,
    }
}

/// A neutral grey anchor cuboid for anchors that carry no type markers
/// (unconstrained inputs, pending outputs), so they stay visible and pickable.
/// `cell_center` is the anchor cell's centre; the cuboid fills that cell's
/// body-facing half, like a type marker would.
fn plain_anchor_body(cell_center: Vec3, is_input: bool) -> RenderObject {
    // Same half of the cell a type marker would occupy, so a typeless anchor
    // hangs off its node exactly like a typed one.
    let sign = if is_input { -1.0 } else { 1.0 };
    let center = cell_center + Vec3::new(0.0, 0.0, sign * ANCHOR_DEPTH * 0.5);
    RenderObject {
        mesh: Cuboid::new(ANCHOR_X, TYPE_MARKER_Y_STEP, ANCHOR_DEPTH)
            .mesh()
            .build(),
        material: StandardMaterial {
            base_color: Color::srgba(0.5, 0.5, 0.5, TYPE_MARKER_ALPHA),
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            unlit: true,
            ..default()
        },
        transform: Transform::from_translation(center),
    }
}

/// Spawn the AST node meshes.
///
/// `extra_offset` (grid units) is added to `layout_node.pos` before the
/// grid→world conversion; used for pattern sub-AST nodes whose positions are
/// relative to the containing pattern.
///
/// `flat_ast` is the program's flattened AST. Type inference needs it because
/// every edge — including those inside Pattern branches — lives in the
/// program-level edge table, while `layout_ast` may be a sub-layout that holds
/// only nodes.
pub fn layoutnode_to_rendernode(
    layout_node: &crate::layout::LayoutNode,
    layout_ast: &crate::layout::LayoutAst,
    flat_ast: &crate::model::ast::Ast,
    function_declarations: &std::collections::HashMap<
        crate::model::function_declaration::FunctionDeclarationId,
        crate::model::function_declaration::FunctionDeclaration,
    >,
    extra_offset: Vec3,
) -> RenderNode {
    let ast = &layout_ast.ast;
    let node_pos = cell_center_world(layout_node.pos + extra_offset);
    // World centre of a node-local cell. Every part of a node — each anchor
    // row, the body — lives in its own cell, so placement goes through this
    // rather than nudging sub-meshes around inside a single cell.
    let cell = |x: i32, y: i32, z: i32| {
        cell_center_world(layout_node.pos + extra_offset + Vec3::new(x as f32, y as f32, z as f32))
    };
    let node = ast.nodes.get(&layout_node.node_id).unwrap();
    match node {
        // Source nodes (a declared constant, a named variable): body band at
        // `0|0` naming the type, output anchor at `0|1`.
        crate::model::node::ENode::ConstDecl {
            r#type,
            output_anchor,
        }
        | crate::model::node::ENode::VarDecl {
            r#type,
            output_anchor,
            ..
        } => {
            let body_world = cell(0, 0, 0);
            let output_world = cell(0, 0, 1);
            let output_eval_type = crate::infer::ast_type_to_eval_type(r#type);
            let output_value = crate::layout::value_of_etype(r#type);
            let bands = crate::edge::leaf_kind_of(&output_eval_type)
                .map(|kind| RenderBand {
                    mesh: crate::edge::build_ribbon_mesh(
                        &source_body_curve(body_world),
                        body_world.y,
                        body_world.y,
                        crate::edge::RIBBON_HEIGHT,
                    ),
                    color: type_marker_color(&output_eval_type),
                    kind,
                })
                .into_iter()
                .collect();
            RenderNode {
                node: None,
                anchors: std::collections::HashMap::from([(
                    output_anchor.clone(),
                    RenderAnchor {
                        pick_center: output_world,
                        type_markers: build_type_markers(
                            &output_eval_type,
                            output_value.as_deref(),
                            output_world,
                            false,
                        ),
                        plain_body: None,
                    },
                )]),
                markers: vec![],
                bands,
                labels: match node {
                    crate::model::node::ENode::VarDecl { .. } => vec![RenderLabel {
                        text: label_for_node(node, function_declarations),
                        color: Color::WHITE,
                        font_size: 18.0,
                        world_pos: body_world,
                        offset: Vec2::ZERO,
                    }],
                    _ => vec![],
                },
            }
        }
        // Input anchor at `0|0`, body (the target type) at `0|1`, output at
        // `0|2`.
        crate::model::node::ENode::TypeCast {
            r#type,
            input_anchor,
            output_anchor,
        } => {
            let color = Color::srgb(0.9, 0.0, 0.0);
            let input_world = cell(0, 0, 0);
            let body_world = cell(0, 0, 1);
            let output_world = cell(0, 0, 2);
            // The output reflects a possibly failed cast as `Sum(target, none)`
            // when a mismatched type flows in, and stays `Pending` while
            // nothing flows in at all; that logic lives in `infer::anchor_type`.
            let output_eval_type =
                crate::infer::anchor_type(flat_ast, output_anchor, function_declarations)
                    .unwrap_or_else(|| crate::infer::ast_type_to_eval_type(r#type));
            let input_eval_type =
                crate::infer::incoming_anchor_type(flat_ast, input_anchor, function_declarations);
            let elim_value = crate::layout::value_of_etype(r#type);
            RenderNode {
                node: Some(RenderObject {
                    mesh: Cuboid::new(ANCHOR_X, TYPE_MARKER_Y_STEP, ANCHOR_X)
                        .mesh()
                        .build(),
                    material: StandardMaterial {
                        base_color: color,
                        emissive: emissive_color(color),
                        metallic: 0.3,
                        perceptual_roughness: 0.6,
                        ..default()
                    },
                    transform: Transform::from_translation(body_world),
                }),
                anchors: std::collections::HashMap::from([
                    (
                        input_anchor.clone(),
                        match input_eval_type {
                            // A typecast constrains nothing, so its input shows
                            // whatever arrives — and a neutral body when idle.
                            Some(t) => typed_anchor(&t, None, input_world, true),
                            None => RenderAnchor {
                                pick_center: input_world,
                                type_markers: vec![],
                                plain_body: Some(plain_anchor_body(input_world, true)),
                            },
                        },
                    ),
                    (
                        output_anchor.clone(),
                        typed_anchor(
                            &output_eval_type,
                            elim_value.as_deref(),
                            output_world,
                            false,
                        ),
                    ),
                ]),
                markers: vec![],
                bands: vec![],
                labels: vec![],
            }
        }
        // Input anchors along `i|0`, body spanning the full width at `z=1..2`,
        // output at `0|3`.
        crate::model::node::ENode::FunctionCall {
            function_declaration_id,
            input_anchors,
            output_anchor,
        } => {
            let function_declaration = function_declarations
                .get(function_declaration_id)
                .expect("function call refers to unknown function declaration");
            let width = input_anchors.len().max(1) as i32;
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
                .map(|a| crate::infer::anchor_rows(flat_ast, a, function_declarations))
                .max()
                .unwrap_or(1);
            let output_rows =
                crate::infer::anchor_rows(flat_ast, output_anchor, function_declarations);
            let first_col = cell(0, 0, 0);
            let last_col = cell(width - 1, 0, 0);
            // Upper edge of row 0, shared by both faces.
            let top_y = first_col.y + cell_y * 0.5;
            // Front face of body cell z=1 and back face of body cell z=2.
            let near_z = cell(0, 0, 1).z + cell_z * 0.5;
            let far_z = cell(0, 0, 2).z - cell_z * 0.5;
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
            let output_world = cell(0, 0, 3);
            RenderNode {
                node: Some(RenderObject {
                    mesh: crate::mesh::frustum_8pt_mesh(base_quad, top_quad),
                    material: StandardMaterial {
                        base_color: Color::srgb(0.5, 0.9, 1.0),
                        emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
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
                        let input_type = function_declaration
                            .inputs
                            .get(i_anchor)
                            .map(|param| param.r#type.clone())
                            .unwrap_or(crate::infer::EType::None);
                        (
                            anchor_id.clone(),
                            typed_anchor(&input_type, None, input_world, true),
                        )
                    })
                    .chain([(
                        output_anchor.clone(),
                        typed_anchor(&function_declaration.output_type, None, output_world, false),
                    )])
                    .collect(),
                markers: vec![],
                bands: vec![],
                labels: std::iter::once(RenderLabel {
                    text: label_for_node(node, function_declarations),
                    color: Color::WHITE,
                    font_size: 18.0,
                    world_pos: body_center,
                    offset: Vec2::ZERO,
                })
                .chain(
                    input_anchors
                        .iter()
                        .enumerate()
                        .filter_map(|(i_anchor, _)| {
                            let name = function_declaration.inputs.get(i_anchor)?.name.clone();
                            Some(RenderLabel {
                                text: name,
                                // Neutral grey, subordinate to the node name and
                                // the centred type-marker letter.
                                color: Color::srgb(0.5, 0.5, 0.5),
                                font_size: 12.0,
                                world_pos: cell(i_anchor as i32, 0, 0),
                                // Nudge down so the type-marker letter stays free.
                                offset: Vec2::new(0.0, 14.0),
                            })
                        }),
                )
                .collect(),
            }
        }
        // Nothing but an input anchor, sitting alone on the scope's last Z row.
        // It constrains nothing, so it renders exactly like a Match or
        // TypeCast input: whatever type arrives, a neutral body when idle.
        crate::model::node::ENode::Sink { input_anchor } => {
            let input_world = cell(0, 0, 0);
            let incoming =
                crate::infer::incoming_anchor_type(flat_ast, input_anchor, function_declarations);
            RenderNode {
                node: None,
                anchors: std::collections::HashMap::from([(
                    input_anchor.clone(),
                    match incoming {
                        Some(t) => typed_anchor(&t, None, input_world, true),
                        None => RenderAnchor {
                            pick_center: input_world,
                            type_markers: vec![],
                            plain_body: Some(plain_anchor_body(input_world, true)),
                        },
                    },
                )]),
                markers: vec![],
                bands: vec![],
                labels: vec![],
            }
        }
        // A Pattern declares the type its arm matches and fixes its branch's
        // row, but owns no anchor — the branch draws its value from its own
        // BranchSource, one cell behind in the branch volume. It is drawn as
        // that type's band, like an input anchor: the value it accepts is what
        // the band names.
        crate::model::node::ENode::Pattern { r#type, .. } => RenderNode {
            node: None,
            anchors: std::collections::HashMap::new(),
            markers: build_type_markers(
                &crate::infer::ast_type_to_eval_type(r#type),
                crate::layout::value_of_etype(r#type).as_deref(),
                node_pos,
                true,
            ),
            bands: vec![],
            labels: vec![],
        },
        // Mirror of the Sink: a single output anchor at the branch origin. Its
        // type is the owning Pattern's, resolved through `infer::anchor_type`.
        crate::model::node::ENode::BranchSource { output_anchor, .. } => {
            let output_world = cell(0, 0, 0);
            // Both type and literal come from the owning Pattern, so the
            // source shows exactly what its arm matched.
            let output_eval_type =
                crate::infer::anchor_type(flat_ast, output_anchor, function_declarations)
                    .unwrap_or(crate::infer::EType::Pending);
            let output_value = crate::infer::anchor_literal(flat_ast, output_anchor);
            RenderNode {
                node: None,
                anchors: std::collections::HashMap::from([(
                    output_anchor.clone(),
                    typed_anchor(
                        &output_eval_type,
                        output_value.as_deref(),
                        output_world,
                        false,
                    ),
                )]),
                markers: vec![],
                bands: vec![],
                labels: vec![],
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
                crate::infer::incoming_anchor_type(flat_ast, input_anchor, function_declarations);
            // The output owns its own cell directly behind the deepest branch;
            // `match_output_z` decides which one. Its type is the union of the
            // branch types, or `Pending` while the inferer cannot decide it.
            let out_world = cell(0, 0, layout_ast.match_output_z(patterns));
            let output_eval_type =
                crate::infer::anchor_type(flat_ast, output_anchor, function_declarations)
                    .unwrap_or(crate::infer::EType::Pending);
            RenderNode {
                node: None,
                anchors: std::collections::HashMap::from([
                    (
                        input_anchor.clone(),
                        match incoming {
                            Some(t) => typed_anchor(&t, None, input_world, true),
                            None => RenderAnchor {
                                pick_center: input_world,
                                type_markers: vec![],
                                plain_body: Some(plain_anchor_body(input_world, true)),
                            },
                        },
                    ),
                    (
                        output_anchor.clone(),
                        typed_anchor(&output_eval_type, None, out_world, false),
                    ),
                ]),
                markers: vec![],
                bands: vec![],
                labels: vec![],
            }
        }
        crate::model::node::ENode::Program { .. } => {
            unreachable!("Program node has no layout position and is never rendered directly")
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
        crate::model::node::ENode::ConstDecl { r#type, .. }
        | crate::model::node::ENode::TypeCast { r#type, .. } => r#type.to_string(),
        crate::model::node::ENode::VarDecl { name, r#type, .. } => {
            format!("{}: {}", name, r#type.to_string())
        }
        crate::model::node::ENode::Match { .. } => "match".to_string(),
        crate::model::node::ENode::Pattern { r#type, .. } => r#type.to_string(),
        crate::model::node::ENode::BranchSource { .. } => "branch source".to_string(),
        crate::model::node::ENode::Program { .. } => "program".to_string(),
    }
}
