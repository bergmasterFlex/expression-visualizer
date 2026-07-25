use bevy::prelude::*;

use crate::ast::FunctionDeclaration;

/// Scale factor from layout coordinates to world coordinates.
pub const LAYOUT_SCALE: Vec3 = Vec3::new(3.0, 3.0, 3.0);

/// Convert a layout position to a world-space position.
pub fn layout_to_world(pos: Vec3) -> Vec3 {
    pos * LAYOUT_SCALE
}

pub struct RenderObject {
    pub mesh: Mesh,
    pub material: StandardMaterial,
    pub transform: Transform,
}

pub struct RenderNode {
    pub node: RenderObject,
    pub anchors: std::collections::HashMap<crate::ast::AnchorId, RenderAnchor>,
    pub labels: Vec<RenderLabel>,
    /// Extra decorative meshes with no associated anchor (e.g. the grey
    /// sink-tip hull of a Match). Spawned alongside `node` by the renderer.
    pub decorations: Vec<RenderObject>,
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
    pub rect: RenderObject,
    pub label: RenderLabel,
    /// Present iff this leaf's anchor carries an AST-level literal. Rendered
    /// as a thin coloured segment along Z in the far 2/3 of the original
    /// rect footprint, at the rect's Y-middle.
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
pub const TYPE_MARKER_Y_STEP: f32 = 1.0;
const TYPE_MARKER_HALF_DEPTH: f32 = 0.25;
/// Full Z-depth of an anchor cuboid (half the old 1.0 type-marker depth).
const ANCHOR_DEPTH: f32 = 2.0 * TYPE_MARKER_HALF_DEPTH;
/// X-width of an anchor cuboid (half the old 0.45 node width). Shared with the
/// three slab nodes so anchor and node line up exactly in X.
pub const ANCHOR_X: f32 = 0.225;
/// Half Z-depth (and half X-width) of the three slab nodes — the anchor cuboid
/// sits flush against this face.
const NODE_HALF_DEPTH: f32 = ANCHOR_X / 2.0;
/// Y-thickness of the gizmo line drawn in the far 2/3 of a value-carrying
/// type marker. Matches the anchor-sphere radius so lines and spheres read as
/// one visual family.
const VALUE_LINE_THICKNESS: f32 = 0.06;
/// World-space padding between the tip of the gizmo line and the value
/// label's projection point.
const VALUE_LABEL_Z_PADDING: f32 = 0.1;

/// Sort key that fixes the vertical order of type-marker rectangles.
/// Returns `None` for variants that should not render a rectangle.
fn type_marker_order(t: &crate::eval::EType) -> Option<u8> {
    match t {
        crate::eval::EType::Bool(..) => Some(0),
        crate::eval::EType::Char(..) => Some(1),
        crate::eval::EType::Int(..) => Some(2),
        crate::eval::EType::Float(..) => Some(3),
        crate::eval::EType::String(..) => Some(4),
        crate::eval::EType::Undefined => Some(5),
        _ => None,
    }
}

fn type_marker_letter(t: &crate::eval::EType) -> &'static str {
    match t {
        crate::eval::EType::Bool(..) => "b",
        crate::eval::EType::Char(..) => "c",
        crate::eval::EType::Int(..) => "i",
        crate::eval::EType::Float(..) => "f",
        crate::eval::EType::String(..) => "s",
        crate::eval::EType::Undefined => "u",
        _ => "?",
    }
}

pub fn type_marker_color(t: &crate::eval::EType) -> Color {
    match t {
        crate::eval::EType::Bool(..) => Color::srgba(0.65, 0.30, 0.95, TYPE_MARKER_ALPHA),
        crate::eval::EType::Char(..) => Color::srgba(0.30, 0.90, 0.40, TYPE_MARKER_ALPHA),
        crate::eval::EType::Int(..) => Color::srgba(0.40, 0.70, 1.00, TYPE_MARKER_ALPHA),
        crate::eval::EType::Float(..) => Color::srgba(0.10, 0.25, 0.75, TYPE_MARKER_ALPHA),
        crate::eval::EType::String(..) => Color::srgba(1.00, 0.90, 0.30, TYPE_MARKER_ALPHA),
        crate::eval::EType::Undefined => Color::srgba(0.95, 0.30, 0.30, TYPE_MARKER_ALPHA),
        _ => Color::srgba(0.5, 0.5, 0.5, TYPE_MARKER_ALPHA),
    }
}

/// Flatten `t` (expanding sum types), filter to the six supported leaves, and
/// sort into the fixed bottom-to-top stack order (undefined at bottom → bool
/// at top).
pub fn ordered_supported_leaves(t: &crate::eval::EType) -> Vec<crate::eval::EType> {
    let mut leaves: Vec<_> = crate::eval::flatten_type(t)
        .into_iter()
        .filter_map(|leaf| type_marker_order(&leaf).map(|k| (k, leaf)))
        .collect();
    leaves.sort_by_key(|(k, _)| std::cmp::Reverse(*k));
    leaves.into_iter().map(|(_, leaf)| leaf).collect()
}

/// Build the stack of translucent type rectangles at an anchor.
///
/// `anchor_world_pos` is the world-space anchor position; `is_input` decides
/// which face of each rectangle touches the anchor (input: lower-Z face at
/// anchor, stack extends +Z; output: upper-Z face at anchor, stack extends −Z).
///
/// `ast_value` is the AST-level literal on the anchor's type, if any. When
/// present, every rendered leaf collapses its rect to the third nearest the
/// anchor and grows a horizontal gizmo line + value label in the far 2/3.
/// In practice value-carrying nodes have a single leaf, so this only fires
/// on one marker per anchor.
fn build_type_markers(
    t: &crate::eval::EType,
    ast_value: Option<&str>,
    anchor_world_pos: Vec3,
    is_input: bool,
) -> Vec<RenderTypeMarker> {
    let leaves = ordered_supported_leaves(t);
    let n = leaves.len();
    if n == 0 {
        return vec![];
    }
    // Sign: +1 for inputs (stack extends +Z from the anchor), -1 for outputs.
    let sign = if is_input { 1.0 } else { -1.0 };
    let full_depth = 2.0 * TYPE_MARKER_HALF_DEPTH;
    let stub_depth = full_depth / 3.0;
    let line_depth = full_depth - stub_depth;
    // Rect (full-value case) is centred at ±HALF_DEPTH from the anchor;
    // the stub rect is centred at ±stub_depth/2 so its near face still
    // touches the anchor at anchor.z.
    let full_rect_z_center = anchor_world_pos.z + sign * TYPE_MARKER_HALF_DEPTH;
    let stub_rect_z_center = anchor_world_pos.z + sign * (stub_depth * 0.5);
    let line_z_center = anchor_world_pos.z + sign * (stub_depth + line_depth * 0.5);
    let line_tip_z = anchor_world_pos.z + sign * full_depth;

    leaves
        .into_iter()
        .enumerate()
        .map(|(k, leaf)| {
            let y_center =
                anchor_world_pos.y + (k as f32 - (n as f32 - 1.0) / 2.0) * TYPE_MARKER_Y_STEP;
            let color = type_marker_color(&leaf);
            let letter = type_marker_letter(&leaf).to_string();

            if let Some(value) = ast_value {
                let stub_center = Vec3::new(anchor_world_pos.x, y_center, stub_rect_z_center);
                let line_center = Vec3::new(anchor_world_pos.x, y_center, line_z_center);
                let label_world = Vec3::new(
                    anchor_world_pos.x,
                    y_center,
                    line_tip_z + sign * VALUE_LABEL_Z_PADDING,
                );
                RenderTypeMarker {
                    rect: RenderObject {
                        mesh: Cuboid::new(ANCHOR_X, TYPE_MARKER_Y_STEP, stub_depth)
                            .mesh()
                            .build(),
                        material: StandardMaterial {
                            base_color: color,
                            alpha_mode: AlphaMode::Blend,
                            cull_mode: None,
                            unlit: true,
                            ..default()
                        },
                        transform: Transform::from_translation(stub_center),
                    },
                    label: RenderLabel {
                        text: letter,
                        color: Color::WHITE,
                        font_size: 14.0,
                        world_pos: stub_center,
                        offset: Vec2::ZERO,
                    },
                    value_line: Some(RenderObject {
                        mesh: Cuboid::new(0.0, VALUE_LINE_THICKNESS, line_depth)
                            .mesh()
                            .build(),
                        material: StandardMaterial {
                            base_color: color,
                            alpha_mode: AlphaMode::Blend,
                            cull_mode: None,
                            unlit: true,
                            ..default()
                        },
                        transform: Transform::from_translation(line_center),
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
                let center = Vec3::new(anchor_world_pos.x, y_center, full_rect_z_center);
                RenderTypeMarker {
                    rect: RenderObject {
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
                    },
                    label: RenderLabel {
                        text: letter,
                        color: Color::WHITE,
                        font_size: 14.0,
                        world_pos: center,
                        offset: Vec2::ZERO,
                    },
                    value_line: None,
                    value_label: None,
                }
            }
        })
        .collect()
}

/// Centre of an anchor cuboid whose near face sits at `face` (world space).
/// Inputs extend the cuboid +Z from the face, outputs extend −Z.
fn anchor_pick_center(face: Vec3, is_input: bool) -> Vec3 {
    let sign = if is_input { 1.0 } else { -1.0 };
    face + Vec3::new(0.0, 0.0, sign * ANCHOR_DEPTH * 0.5)
}

/// A neutral grey anchor cuboid centred at `center`. Used for anchors that
/// carry no type markers (Sink, Match) so they stay visible
/// and pickable once the spheres are gone.
fn plain_anchor_body(center: Vec3) -> RenderObject {
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
/// relative to the containing pattern. `sink_scale` shrinks a Sink's mesh
/// (anchor sphere and its transform stay unchanged so it remains clickable);
/// non-Sink node kinds ignore it.
pub fn layoutnode_to_rendernode(
    layout_node: &crate::layout::LayoutNode,
    layout_ast: &crate::layout::LayoutAst,
    function_declarations: &std::collections::HashMap<
        crate::ast::FunctionDeclarationId,
        crate::ast::FunctionDeclaration,
    >,
    extra_offset: Vec3,
    sink_scale: f32,
) -> RenderNode {
    let ast = &layout_ast.ast;
    let node_pos = layout_to_world(layout_node.pos + extra_offset);
    let node_pos_tf = Transform::from_translation(node_pos);
    let node = ast.nodes.get(&layout_node.node_id).unwrap();
    return match node {
        crate::ast::node::ENode::ConstDecl {
            r#type,
            output_anchor,
        } => {
            let color = Color::srgb(0.0, 0.9, 0.0);
            let output_local = Vec3::new(0.0, 0.0, -NODE_HALF_DEPTH);
            let output_world = node_pos + output_local;
            let output_eval_type = crate::eval::ast_type_to_eval_type(r#type);
            let output_value = crate::layout::value_of_etype(r#type);
            RenderNode {
                node: RenderObject {
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
                    transform: node_pos_tf,
                },
                anchors: std::collections::HashMap::from([(
                    output_anchor.clone(),
                    RenderAnchor {
                        pick_center: anchor_pick_center(output_world, false),
                        type_markers: build_type_markers(
                            &output_eval_type,
                            output_value.as_deref(),
                            output_world,
                            false,
                        ),
                        plain_body: None,
                    },
                )]),
                labels: vec![],
                decorations: vec![],
            }
        }
        crate::ast::node::ENode::TypeCast {
            r#type,
            input_anchor,
            output_anchor,
        } => {
            let color = Color::srgb(0.9, 0.0, 0.0);
            let input_local = Vec3::new(0.0, 0.0, NODE_HALF_DEPTH);
            let output_local = Vec3::new(0.0, 0.0, -NODE_HALF_DEPTH);
            let input_world = node_pos + input_local;
            let output_world = node_pos + output_local;
            let eval_type = crate::eval::ast_type_to_eval_type(r#type);
            let elim_value = crate::layout::value_of_etype(r#type);
            RenderNode {
                node: RenderObject {
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
                    transform: node_pos_tf,
                },
                anchors: std::collections::HashMap::from([
                    (
                        input_anchor.clone(),
                        RenderAnchor {
                            pick_center: anchor_pick_center(input_world, true),
                            type_markers: build_type_markers(
                                &eval_type,
                                elim_value.as_deref(),
                                input_world,
                                true,
                            ),
                            plain_body: None,
                        },
                    ),
                    (
                        output_anchor.clone(),
                        RenderAnchor {
                            pick_center: anchor_pick_center(output_world, false),
                            type_markers: build_type_markers(
                                &eval_type,
                                elim_value.as_deref(),
                                output_world,
                                false,
                            ),
                            plain_body: None,
                        },
                    ),
                ]),
                labels: vec![],
                decorations: vec![],
            }
        }
        crate::ast::node::ENode::FunctionCall {
            function_declaration_id,
            input_anchors,
            output_anchor,
        } => {
            let function_declaration = function_declarations
                .get(function_declaration_id)
                .expect("function call refers to unknown function declaration");
            // Footprint-driven placement: the mesh centres on the AABB of the
            // node's footprint (min-extent rule in layout::node_footprint), so
            // 1–2-input calls sit between two cells along Z and 3+-input calls
            // sit between four cells (X+0.5, Z−0.5 from the anchor cell).
            // The pyramid itself keeps its original small dimensions — only
            // the footprint (grid-line suppression + displacement) grows.
            let footprint = layout_ast
                .node_footprint(&layout_node.node_id)
                .expect("function call must have a footprint");
            let fp_center_x_grid = (footprint.min.x + footprint.max.x) as f32 * 0.5;
            let fp_center_z_grid = (footprint.min.z + footprint.max.z) as f32 * 0.5;
            let n = input_anchors.len() as f32;
            let mesh_base_x_world = n * 0.5;
            let mesh_depth_world = 1.0;
            let mesh_height_world = std::f32::consts::FRAC_1_SQRT_2;
            // The rect_pyramid_z_mesh has its origin at the base centre (base
            // at local Z=0, tip at local Z=−depth). Shift the transform by
            // +depth/2 so the mesh's Z-bbox centres on the footprint centre.
            let fp_center_world = layout_to_world(
                Vec3::new(fp_center_x_grid, layout_node.pos.y, fp_center_z_grid) + extra_offset,
            );
            let mesh_center_world = fp_center_world + Vec3::new(0.0, 0.0, mesh_depth_world * 0.5);
            let mesh_tf = Transform::from_translation(mesh_center_world);
            let node_center_world = fp_center_world;
            let spread = 1.0_f32;
            let output_local = Vec3::new(0.0, 0.0, -mesh_depth_world);
            let output_world = mesh_center_world + output_local;
            RenderNode {
                node: RenderObject {
                    mesh: crate::mesh::rect_pyramid_z_mesh(
                        mesh_base_x_world,
                        mesh_height_world,
                        mesh_depth_world,
                    ),
                    material: StandardMaterial {
                        base_color: Color::srgb(0.5, 0.9, 1.0),
                        emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                        unlit: true,
                        ..default()
                    },
                    transform: mesh_tf,
                },
                anchors: input_anchors
                    .iter()
                    .enumerate()
                    .map(|(i_anchor, anchor_id)| {
                        let start_x = -(n - 1.0) * spread / 2.0;
                        let x = start_x + i_anchor as f32 * spread;
                        let input_local = Vec3::new(x, 0.0, 0.0);
                        let input_world = mesh_center_world + input_local;
                        let input_type = function_declaration
                            .inputs
                            .get(i_anchor)
                            .map(|param| param.r#type.clone())
                            .unwrap_or(crate::eval::EType::Undefined);
                        (
                            anchor_id.clone(),
                            RenderAnchor {
                                pick_center: anchor_pick_center(input_world, true),
                                type_markers: build_type_markers(
                                    &input_type,
                                    None,
                                    input_world,
                                    true,
                                ),
                                plain_body: None,
                            },
                        )
                    })
                    .chain([(
                        output_anchor.clone(),
                        RenderAnchor {
                            pick_center: anchor_pick_center(output_world, false),
                            type_markers: build_type_markers(
                                &function_declaration.output_type,
                                None,
                                output_world,
                                false,
                            ),
                            plain_body: None,
                        },
                    )])
                    .collect(),
                labels: vec![RenderLabel {
                    text: label_for_node(node, function_declarations),
                    color: Color::WHITE,
                    font_size: 18.0,
                    world_pos: node_center_world,
                    offset: Vec2::ZERO,
                }],
                decorations: vec![],
            }
        }
        crate::ast::node::ENode::VarDecl {
            r#type,
            output_anchor,
            ..
        } => {
            let color = Color::srgb(0.0, 0.6, 0.9);
            let output_local = Vec3::new(0.0, 0.0, -NODE_HALF_DEPTH);
            let output_world = node_pos + output_local;
            let output_eval_type = crate::eval::ast_type_to_eval_type(r#type);
            let output_value = crate::layout::value_of_etype(r#type);
            RenderNode {
                node: RenderObject {
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
                    transform: node_pos_tf,
                },
                anchors: std::collections::HashMap::from([(
                    output_anchor.clone(),
                    RenderAnchor {
                        pick_center: anchor_pick_center(output_world, false),
                        type_markers: build_type_markers(
                            &output_eval_type,
                            output_value.as_deref(),
                            output_world,
                            false,
                        ),
                        plain_body: None,
                    },
                )]),
                labels: vec![RenderLabel {
                    text: label_for_node(node, function_declarations),
                    color: Color::WHITE,
                    font_size: 18.0,
                    world_pos: node_pos,
                    offset: Vec2::ZERO,
                }],
                decorations: vec![],
            }
        }
        crate::ast::node::ENode::Sink { input_anchor } => RenderNode {
            node: RenderObject {
                mesh: crate::mesh::square_pyramid_z_mesh(6.0 * sink_scale, 9.0 * sink_scale),
                material: StandardMaterial {
                    base_color: Color::srgba(0.5, 0.5, 0.5, 0.5),
                    alpha_mode: AlphaMode::Blend,
                    cull_mode: None,
                    ..default()
                },
                transform: node_pos_tf,
            },
            anchors: std::collections::HashMap::from([(
                input_anchor.clone(),
                RenderAnchor {
                    pick_center: node_pos,
                    type_markers: vec![],
                    plain_body: Some(plain_anchor_body(node_pos)),
                },
            )]),
            labels: vec![],
            decorations: vec![],
        },
        crate::ast::node::ENode::Pattern {
            r#type,
            output_anchor,
            ..
        } => {
            let color = Color::srgb(0.9, 0.0, 0.0);
            let output_local = Vec3::new(0.0, 0.0, -NODE_HALF_DEPTH);
            let output_world = node_pos + output_local;
            let eval_type = crate::eval::ast_type_to_eval_type(r#type);
            let pattern_value = crate::layout::value_of_etype(r#type);
            RenderNode {
                node: RenderObject {
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
                    transform: node_pos_tf,
                },
                anchors: std::collections::HashMap::from([(
                    output_anchor.clone(),
                    RenderAnchor {
                        pick_center: anchor_pick_center(output_world, false),
                        type_markers: build_type_markers(
                            &eval_type,
                            pattern_value.as_deref(),
                            output_world,
                            false,
                        ),
                        plain_body: None,
                    },
                )]),
                labels: vec![],
                decorations: vec![],
            }
        }
        crate::ast::node::ENode::Match {
            patterns,
            input_anchor,
            output_anchor,
        } => {
            // The Match's LayoutNode sits at the lowest Pattern's grid pos
            // (see LayoutAst::recompute_match_pos). Find the highest sibling
            // to size the envelope.
            let max_y_grid = patterns
                .iter()
                .filter_map(|pid| layout_ast.layout_nodes.get(pid).map(|ln| ln.pos.y))
                .fold(layout_node.pos.y, f32::max);
            let y_diff_grid = max_y_grid - layout_node.pos.y;
            let y_diff_world = y_diff_grid * LAYOUT_SCALE.y;
            // Pad by a full anchor-rect height so the top/bottom patterns (now
            // TYPE_MARKER_Y_STEP tall) stay enclosed by the envelope.
            let height = y_diff_world + TYPE_MARKER_Y_STEP;
            let center_local = Vec3::new(0.0, y_diff_world / 2.0, 0.0);
            // Envelope hugs the slim patterns with the same 0.05 per-side margin
            // it had around the old 0.45 cubes.
            let envelope_xz = ANCHOR_X + 2.0 * 0.05;
            // Input port sits flush against the envelope's +Z front face.
            let front_face = node_pos + Vec3::new(0.0, 0.0, envelope_xz / 2.0);
            let port_center = anchor_pick_center(front_face, true);

            let mut anchors = std::collections::HashMap::from([(
                input_anchor.clone(),
                RenderAnchor {
                    pick_center: port_center,
                    type_markers: vec![],
                    plain_body: Some(plain_anchor_body(port_center)),
                },
            )]);
            let mut decorations = Vec::new();

            // Grey sink-tip hull: the mirror of the red pattern envelope, drawn
            // along the −Z tips of every sibling Pattern's Sink. Each
            // Pattern's sink lives in its sub-layout; `harmonize_match_sinks`
            // aligns all sibling sinks to a common (x, z), so the tips form a
            // vertical Y column. Tip world = layout_to_world(sink_local +
            // pattern_grid + extra_offset) + (0, 0, −SINK_TIP_DEPTH). Pattern
            // sub-content always renders at sink_scale 1/3 (see walk_all_into),
            // so the pyramid depth is 9 · 1/3 = 3.0 world units.
            const SINK_TIP_DEPTH: f32 = 3.0;
            let mut tip_x = node_pos.x;
            let mut tip_z = node_pos.z;
            let mut min_tip_y = f32::MAX;
            let mut max_tip_y = f32::MIN;
            for pid in patterns {
                let Some(pattern_ln) = layout_ast.layout_nodes.get(pid) else {
                    continue;
                };
                let Some(sub) = layout_ast.sub_layouts.get(pid) else {
                    continue;
                };
                let Some(sink_id) = sub.sink_id() else {
                    continue;
                };
                let Some(sink_ln) = sub.layout_nodes.get(&sink_id) else {
                    continue;
                };
                let tip = layout_to_world(sink_ln.pos + pattern_ln.pos + extra_offset)
                    + Vec3::new(0.0, 0.0, -SINK_TIP_DEPTH);
                tip_x = tip.x;
                tip_z = tip.z;
                min_tip_y = min_tip_y.min(tip.y);
                max_tip_y = max_tip_y.max(tip.y);
            }

            if min_tip_y <= max_tip_y {
                let hull_height = (max_tip_y - min_tip_y) + TYPE_MARKER_Y_STEP;
                let hull_center = Vec3::new(tip_x, (min_tip_y + max_tip_y) * 0.5, tip_z);
                decorations.push(RenderObject {
                    mesh: Cuboid::new(envelope_xz, hull_height, envelope_xz)
                        .mesh()
                        .build(),
                    material: StandardMaterial {
                        base_color: Color::srgba(0.5, 0.5, 0.5, 0.35),
                        alpha_mode: AlphaMode::Blend,
                        cull_mode: None,
                        ..default()
                    },
                    transform: Transform::from_translation(hull_center),
                });
                // Output port mirrors the input port: flush against the hull's
                // −Z back face, at the base (lowest) Pattern's Y (node_pos.y).
                let back_face = Vec3::new(tip_x, node_pos.y, tip_z - envelope_xz / 2.0);
                let out_center = anchor_pick_center(back_face, false);
                anchors.insert(
                    output_anchor.clone(),
                    RenderAnchor {
                        pick_center: out_center,
                        type_markers: vec![],
                        plain_body: Some(plain_anchor_body(out_center)),
                    },
                );
            }

            RenderNode {
                node: RenderObject {
                    mesh: Cuboid::new(envelope_xz, height, envelope_xz).mesh().build(),
                    material: StandardMaterial {
                        base_color: Color::srgba(0.9, 0.0, 0.0, 0.35),
                        alpha_mode: AlphaMode::Blend,
                        cull_mode: None,
                        ..default()
                    },
                    transform: node_pos_tf * Transform::from_translation(center_local),
                },
                anchors,
                labels: vec![],
                decorations,
            }
        }
        crate::ast::node::ENode::Program { .. } => {
            unreachable!("Program node has no layout position and is never rendered directly")
        }
    };
}

pub fn emissive_color(color: Color) -> LinearRgba {
    let c = color.to_linear();
    LinearRgba::new(c.red * 0.15, c.green * 0.15, c.blue * 0.15, 1.0)
}

pub fn label_for_node(
    node: &crate::ast::node::ENode,
    function_declarations: &std::collections::HashMap<
        crate::ast::FunctionDeclarationId,
        FunctionDeclaration,
    >,
) -> String {
    use crate::ast::node::ENode;
    match node {
        ENode::Sink { .. } => "sink".to_string(),
        ENode::FunctionCall {
            function_declaration_id,
            ..
        } => function_declarations
            .get(&function_declaration_id)
            .unwrap()
            .name
            .to_string(),
        ENode::ConstDecl { r#type, .. } | ENode::TypeCast { r#type, .. } => r#type.to_string(),
        ENode::VarDecl { name, r#type, .. } => format!("{}: {}", name, r#type.to_string()),
        ENode::Match { .. } => "match".to_string(),
        ENode::Pattern { r#type, .. } => r#type.to_string(),
        ENode::Program { .. } => "program".to_string(),
    }
}
