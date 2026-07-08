use bevy::prelude::*;

use crate::ast::FunctionDeclaration;

/// Scale factor from layout coordinates to world coordinates.
pub const LAYOUT_SCALE: Vec3 = Vec3::new(3.0, 1.5, 3.0);

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
}

pub struct RenderAnchor {
    pub normal: RenderObject,
    pub hovered: RenderObject,
    pub type_markers: Vec<RenderTypeMarker>,
}

pub struct RenderTypeMarker {
    pub rect: RenderObject,
    pub label: RenderLabel,
}

pub struct RenderLabel {
    pub text: String,
    pub color: Color,
    pub font_size: f32,
    pub world_pos: Vec3,
    pub offset: Vec2,
}

const TYPE_MARKER_ALPHA: f32 = 0.6;
const TYPE_MARKER_Y_STEP: f32 = 1.0;
const TYPE_MARKER_HALF_DEPTH: f32 = 0.5;

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

fn type_marker_color(t: &crate::eval::EType) -> Color {
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
fn ordered_supported_leaves(t: &crate::eval::EType) -> Vec<crate::eval::EType> {
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
fn build_type_markers(
    t: &crate::eval::EType,
    anchor_world_pos: Vec3,
    is_input: bool,
) -> Vec<RenderTypeMarker> {
    let leaves = ordered_supported_leaves(t);
    let n = leaves.len();
    if n == 0 {
        return vec![];
    }
    let z_center = if is_input {
        anchor_world_pos.z + TYPE_MARKER_HALF_DEPTH
    } else {
        anchor_world_pos.z - TYPE_MARKER_HALF_DEPTH
    };
    leaves
        .into_iter()
        .enumerate()
        .map(|(k, leaf)| {
            let y_center =
                anchor_world_pos.y + (k as f32 - (n as f32 - 1.0) / 2.0) * TYPE_MARKER_Y_STEP;
            let center = Vec3::new(anchor_world_pos.x, y_center, z_center);
            RenderTypeMarker {
                rect: RenderObject {
                    mesh: Cuboid::new(0.0, TYPE_MARKER_Y_STEP, 2.0 * TYPE_MARKER_HALF_DEPTH)
                        .mesh()
                        .build(),
                    material: StandardMaterial {
                        base_color: type_marker_color(&leaf),
                        alpha_mode: AlphaMode::Blend,
                        cull_mode: None,
                        unlit: true,
                        ..default()
                    },
                    transform: Transform::from_translation(center),
                },
                label: RenderLabel {
                    text: type_marker_letter(&leaf).to_string(),
                    color: Color::WHITE,
                    font_size: 14.0,
                    world_pos: center,
                    offset: Vec2::ZERO,
                },
            }
        })
        .collect()
}

/// Spawn the AST node meshes.
pub fn layoutnode_to_rendernode(
    layout_node: &crate::layout::LayoutNode,
    ast: &crate::ast::Ast,
    function_declarations: &std::collections::HashMap<
        crate::ast::FunctionDeclarationId,
        crate::ast::FunctionDeclaration,
    >,
) -> RenderNode {
    let node_pos = layout_node.pos * Vec3::new(3.0, 1.5, 3.0);
    let node_pos_tf = Transform::from_translation(node_pos);
    let node = ast.nodes.get(&layout_node.node_id).unwrap();
    return match node {
        crate::ast::node::ENode::TypeIntroduction {
            r#type,
            output_anchor,
        } => {
            let color = Color::srgb(0.0, 0.9, 0.0);
            let output_local = Vec3::new(0.0, 0.0, -0.225);
            let output_world = node_pos + output_local;
            let output_eval_type = crate::eval::ast_type_to_eval_type(r#type);
            RenderNode {
                node: RenderObject {
                    mesh: match r#type {
                        /*
                        let sphere_mesh = meshes.add(Sphere::new(0.32));
                        let octa_mesh = meshes.add(mesh::octahedron_mesh(0.38));
                        let ring_mesh = meshes.add(Torus::new(0.225, 0.38));
                        let ring_big_mesh = meshes.add(Torus::new(0.025, 0.48));
                        let cone_mesh = meshes.add(mesh::create_cone_mesh(0.5, 1.0, 16));
                        let pyramide_mesh = meshes.add(mesh::create_cone_mesh(0.5, 1.0, 4));
                        let bool_mesh = meshes.add(mesh::create_bool_mesh(0.5, 1.0, 16));
                        */
                        crate::ast::node::EType::Bool { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                        crate::ast::node::EType::Int { .. } => Cuboid::new(0.45, 0.45, 0.45).mesh(),
                        crate::ast::node::EType::Float { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                        crate::ast::node::EType::String { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                        crate::ast::node::EType::Char { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                        crate::ast::node::EType::Any { .. } => Cuboid::new(0.45, 0.45, 0.45).mesh(),
                        crate::ast::node::EType::Undefined { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                        crate::ast::node::EType::Exception { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                    }
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
                        normal: RenderObject {
                            mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                            material: StandardMaterial {
                                base_color: Color::srgb(0.3, 0.6, 1.0),
                                emissive: LinearRgba::new(0.05, 0.1, 0.2, 1.0),
                                unlit: true,
                                ..default()
                            },
                            transform: node_pos_tf * Transform::from_translation(output_local),
                        },
                        hovered: RenderObject {
                            mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                            material: StandardMaterial {
                                base_color: Color::srgb(0.5, 0.9, 1.0),
                                emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                                unlit: true,
                                ..default()
                            },
                            transform: node_pos_tf
                                * Transform::from_translation(output_local)
                                * Transform::from_scale(Vec3::splat(1.8)),
                        },
                        type_markers: build_type_markers(&output_eval_type, output_world, false),
                    },
                )]),
                labels: vec![RenderLabel {
                    text: node.label(function_declarations),
                    color: Color::WHITE,
                    font_size: 18.0,
                    world_pos: node_pos,
                    offset: Vec2::ZERO,
                }],
            }
        }
        crate::ast::node::ENode::TypeElimination {
            r#type,
            input_anchor,
            output_anchor,
        } => {
            let color = Color::srgb(0.9, 0.0, 0.0);
            let input_local = Vec3::new(0.0, 0.0, 0.55);
            let output_local = Vec3::new(0.0, 0.0, -0.55);
            let input_world = node_pos + input_local;
            let output_world = node_pos + output_local;
            let eval_type = crate::eval::ast_type_to_eval_type(r#type);
            RenderNode {
                node: RenderObject {
                    mesh: match r#type {
                        crate::ast::node::EType::Bool { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                        crate::ast::node::EType::Int { .. } => Cuboid::new(0.45, 0.45, 0.45).mesh(),
                        crate::ast::node::EType::Float { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                        crate::ast::node::EType::String { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                        crate::ast::node::EType::Char { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                        crate::ast::node::EType::Any { .. } => Cuboid::new(0.45, 0.45, 0.45).mesh(),
                        crate::ast::node::EType::Undefined { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                        crate::ast::node::EType::Exception { .. } => {
                            Cuboid::new(0.45, 0.45, 0.45).mesh()
                        }
                    }
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
                            normal: RenderObject {
                                mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                                material: StandardMaterial {
                                    base_color: Color::srgb(0.3, 0.6, 1.0),
                                    emissive: LinearRgba::new(0.05, 0.1, 0.2, 1.0),
                                    unlit: true,
                                    ..default()
                                },
                                transform: node_pos_tf * Transform::from_translation(input_local),
                            },
                            hovered: RenderObject {
                                mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                                material: StandardMaterial {
                                    base_color: Color::srgb(0.5, 0.9, 1.0),
                                    emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                                    unlit: true,
                                    ..default()
                                },
                                transform: node_pos_tf
                                    * Transform::from_translation(input_local)
                                    * Transform::from_scale(Vec3::splat(1.8)),
                            },
                            type_markers: build_type_markers(&eval_type, input_world, true),
                        },
                    ),
                    (
                        output_anchor.clone(),
                        RenderAnchor {
                            normal: RenderObject {
                                mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                                material: StandardMaterial {
                                    base_color: Color::srgb(0.3, 0.6, 1.0),
                                    emissive: LinearRgba::new(0.05, 0.1, 0.2, 1.0),
                                    unlit: true,
                                    ..default()
                                },
                                transform: node_pos_tf * Transform::from_translation(output_local),
                            },
                            hovered: RenderObject {
                                mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                                material: StandardMaterial {
                                    base_color: Color::srgb(0.5, 0.9, 1.0),
                                    emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                                    unlit: true,
                                    ..default()
                                },
                                transform: node_pos_tf
                                    * Transform::from_translation(output_local)
                                    * Transform::from_scale(Vec3::splat(1.8)),
                            },
                            type_markers: build_type_markers(&eval_type, output_world, false),
                        },
                    ),
                ]),
                labels: vec![RenderLabel {
                    text: node.label(function_declarations),
                    color: Color::WHITE,
                    font_size: 18.0,
                    world_pos: node_pos,
                    offset: Vec2::ZERO,
                }],
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
            let output_local = Vec3::new(0.0, 0.0, -1.0);
            let output_world = node_pos + output_local;
            RenderNode {
                node: RenderObject {
                    mesh: crate::mesh::rect_pyramid_z_mesh(
                        input_anchors.len() as f32 * 0.5,
                        std::f32::consts::FRAC_1_SQRT_2,
                        1.0,
                    ),
                    material: StandardMaterial {
                        base_color: Color::srgb(0.5, 0.9, 1.0),
                        emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                        unlit: true,
                        ..default()
                    },
                    transform: node_pos_tf,
                },
                anchors: input_anchors
                    .iter()
                    .enumerate()
                    .map(|(i_anchor, anchor_id)| {
                        let spread = 1.0;
                        let start_x = -(input_anchors.len() as f32 - 1.0) * spread / 2.0;
                        let x = start_x + i_anchor as f32 * spread;
                        let input_local = Vec3::new(x, 0.0, 0.0);
                        let input_world = node_pos + input_local;
                        let input_type = function_declaration
                            .inputs
                            .get(i_anchor)
                            .map(|param| param.r#type.clone())
                            .unwrap_or(crate::eval::EType::Undefined);
                        (
                            anchor_id.clone(),
                            RenderAnchor {
                                normal: RenderObject {
                                    mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                                    material: StandardMaterial {
                                        base_color: Color::srgb(0.3, 0.6, 1.0),
                                        emissive: LinearRgba::new(0.05, 0.1, 0.2, 1.0),
                                        unlit: true,
                                        ..default()
                                    },
                                    transform: node_pos_tf
                                        * Transform::from_translation(input_local),
                                },
                                hovered: RenderObject {
                                    mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                                    material: StandardMaterial {
                                        base_color: Color::srgb(0.5, 0.9, 1.0),
                                        emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                                        unlit: true,
                                        ..default()
                                    },
                                    transform: node_pos_tf
                                        * Transform::from_translation(input_local)
                                        * Transform::from_scale(Vec3::splat(1.8)),
                                },
                                type_markers: build_type_markers(&input_type, input_world, true),
                            },
                        )
                    })
                    .chain([(
                        output_anchor.clone(),
                        RenderAnchor {
                            normal: RenderObject {
                                mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                                material: StandardMaterial {
                                    base_color: Color::srgb(0.3, 0.6, 1.0),
                                    emissive: LinearRgba::new(0.05, 0.1, 0.2, 1.0),
                                    unlit: true,
                                    ..default()
                                },
                                transform: node_pos_tf * Transform::from_translation(output_local),
                            },
                            hovered: RenderObject {
                                mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                                material: StandardMaterial {
                                    base_color: Color::srgb(0.5, 0.9, 1.0),
                                    emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                                    unlit: true,
                                    ..default()
                                },
                                transform: node_pos_tf
                                    * Transform::from_translation(output_local)
                                    * Transform::from_scale(Vec3::splat(1.8)),
                            },
                            type_markers: build_type_markers(
                                &function_declaration.output_type,
                                output_world,
                                false,
                            ),
                        },
                    )])
                    .collect(),
                labels: vec![RenderLabel {
                    text: node.label(function_declarations),
                    color: Color::WHITE,
                    font_size: 18.0,
                    world_pos: node_pos,
                    offset: Vec2::ZERO,
                }],
            }
        }
        crate::ast::node::ENode::VarDecl {
            r#type,
            output_anchor,
            ..
        } => {
            let color = Color::srgb(0.0, 0.6, 0.9);
            let output_local = Vec3::new(0.0, 0.0, -0.225);
            let output_world = node_pos + output_local;
            let output_eval_type = crate::eval::ast_type_to_eval_type(r#type);
            RenderNode {
                node: RenderObject {
                    mesh: Cuboid::new(0.45, 0.45, 0.45).mesh().build(),
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
                        normal: RenderObject {
                            mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                            material: StandardMaterial {
                                base_color: Color::srgb(0.3, 0.6, 1.0),
                                emissive: LinearRgba::new(0.05, 0.1, 0.2, 1.0),
                                unlit: true,
                                ..default()
                            },
                            transform: node_pos_tf * Transform::from_translation(output_local),
                        },
                        hovered: RenderObject {
                            mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                            material: StandardMaterial {
                                base_color: Color::srgb(0.5, 0.9, 1.0),
                                emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                                unlit: true,
                                ..default()
                            },
                            transform: node_pos_tf
                                * Transform::from_translation(output_local)
                                * Transform::from_scale(Vec3::splat(1.8)),
                        },
                        type_markers: build_type_markers(&output_eval_type, output_world, false),
                    },
                )]),
                labels: vec![RenderLabel {
                    text: node.label(function_declarations),
                    color: Color::WHITE,
                    font_size: 18.0,
                    world_pos: node_pos,
                    offset: Vec2::ZERO,
                }],
            }
        }
        crate::ast::node::ENode::SinkWall { input_anchor } => RenderNode {
            node: RenderObject {
                mesh: crate::mesh::square_pyramid_z_mesh(6.0, 9.0),
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
                    normal: RenderObject {
                        mesh: Sphere::new(0.12).mesh().ico(2).unwrap(),
                        material: StandardMaterial {
                            base_color: Color::srgb(0.3, 0.6, 1.0),
                            emissive: LinearRgba::new(0.05, 0.1, 0.2, 1.0),
                            unlit: true,
                            ..default()
                        },
                        transform: node_pos_tf,
                    },
                    hovered: RenderObject {
                        mesh: Sphere::new(0.12).mesh().ico(2).unwrap(),
                        material: StandardMaterial {
                            base_color: Color::srgb(0.5, 0.9, 1.0),
                            emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                            unlit: true,
                            ..default()
                        },
                        transform: node_pos_tf * Transform::from_scale(Vec3::splat(1.8)),
                    },
                    type_markers: vec![],
                },
            )]),
            labels: vec![],
        },
        crate::ast::node::ENode::MatchFront { levels, width } => {
            let w = *width as f32;
            let l = *levels as f32;
            RenderNode {
                node: RenderObject {
                    mesh: Cuboid::new(w * 6.0 - 3.0, l * 3.0, 0.05).mesh().build(),
                    material: StandardMaterial {
                        base_color: Color::srgba(0.5, 0.5, 0.5, 0.5),
                        alpha_mode: AlphaMode::Blend,
                        cull_mode: None,
                        ..default()
                    },
                    transform: node_pos_tf
                        * Transform::from_translation(Vec3::new(w * 3.0 - 3.0, l * 1.5 - 1.5, 0.0)),
                },
                anchors: std::collections::HashMap::new(),
                labels: vec![],
            }
        }
        crate::ast::node::ENode::MatchBack {
            levels,
            input_anchors,
            output_anchor,
        } => {
            let top_y = *levels as f32 * 3.0 - 1.5;
            // Base CCW viewed from +z (since tip sits at -z).
            let base = [
                [-1.5, -1.5, 0.0],
                [1.5, -1.5, 0.0],
                [1.5, top_y, 0.0],
                [-1.5, top_y, 0.0],
            ];
            let tip = [0.0, 0.0, -3.0];
            let input_spread = 3.0;
            let anchors = input_anchors
                .iter()
                .enumerate()
                .map(|(i, anchor_id)| {
                    let y = i as f32 * input_spread;
                    (
                        anchor_id.clone(),
                        RenderAnchor {
                            normal: RenderObject {
                                mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                                material: StandardMaterial {
                                    base_color: Color::srgb(0.3, 0.6, 1.0),
                                    emissive: LinearRgba::new(0.05, 0.1, 0.2, 1.0),
                                    unlit: true,
                                    ..default()
                                },
                                transform: node_pos_tf
                                    * Transform::from_translation(Vec3::new(0.0, y, 0.0)),
                            },
                            hovered: RenderObject {
                                mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                                material: StandardMaterial {
                                    base_color: Color::srgb(0.5, 0.9, 1.0),
                                    emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                                    unlit: true,
                                    ..default()
                                },
                                transform: node_pos_tf
                                    * Transform::from_translation(Vec3::new(0.0, y, 0.0))
                                    * Transform::from_scale(Vec3::splat(1.8)),
                            },
                            type_markers: vec![],
                        },
                    )
                })
                .chain([(
                    output_anchor.clone(),
                    RenderAnchor {
                        normal: RenderObject {
                            mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                            material: StandardMaterial {
                                base_color: Color::srgb(0.3, 0.6, 1.0),
                                emissive: LinearRgba::new(0.05, 0.1, 0.2, 1.0),
                                unlit: true,
                                ..default()
                            },
                            transform: node_pos_tf
                                * Transform::from_translation(Vec3::new(0.0, 0.0, -3.0)),
                        },
                        hovered: RenderObject {
                            mesh: Sphere::new(0.06).mesh().ico(2).unwrap(),
                            material: StandardMaterial {
                                base_color: Color::srgb(0.5, 0.9, 1.0),
                                emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                                unlit: true,
                                ..default()
                            },
                            transform: node_pos_tf
                                * Transform::from_translation(Vec3::new(0.0, 0.0, -3.0))
                                * Transform::from_scale(Vec3::splat(1.8)),
                        },
                        type_markers: vec![],
                    },
                )])
                .collect();
            RenderNode {
                node: RenderObject {
                    mesh: crate::mesh::pyramid_5pt_mesh(base, tip),
                    material: StandardMaterial {
                        base_color: Color::srgba(0.5, 0.5, 0.5, 0.5),
                        alpha_mode: AlphaMode::Blend,
                        cull_mode: None,
                        ..default()
                    },
                    transform: node_pos_tf,
                },
                anchors,
                labels: vec![],
            }
        }
        crate::ast::node::ENode::MatchGrid { .. } => RenderNode {
            node: RenderObject {
                mesh: Cuboid::new(0.0, 0.0, 0.0).mesh().build(),
                material: StandardMaterial::default(),
                transform: node_pos_tf,
            },
            anchors: std::collections::HashMap::new(),
            labels: vec![],
        },
    };

    /*
    let mut node_entites = std::collections::HashMap::<ast::AstNodeId, Entity>::new();
    let mut anchor_entities = std::collections::HashMap::<ast::AnchorId, Entity>::new();

    for (node_id, node) in &state.layout_ast.ast.nodes {
        let color = node_color(node);
        let emissive = node_emissive(&node);

        let material = materials.add(StandardMaterial {
            base_color: color,
            emissive,
            metallic: 0.3,
            perceptual_roughness: 0.6,
            ..default()
        });

        let node_pos = state.layout_ast.layout_nodes.get(node_id).unwrap().pos;
        let node_pos = node_pos * Vec3::new(3.0, 1.5, 3.0);

        // Pick shape based on AST type


        let anchors = node.anchors();
        let input_anchor_count = anchors
            .iter()
            .filter(|(_, a)| match a {
                ast::EAnchor::Input { .. } => true,
                _ => false,
            })
            .count();

        let node_entity = commands
            .spawn((
                pbr_bundle,
                AstNodeEntity {
                    node_id: node_id.clone(),
                },
                AstSceneEntity,
            ))
            .id();

        commands.entity(node_entity).with_children(|parent| {
            anchors.into_iter().for_each(|(id, anchor)| {
                let (b, a) = spawn_anchor(id.clone(), anchor, input_anchor_count, &anchor_assets);
                anchor_entities.insert(id, parent.spawn((b, a, anchor_assets.clone())).id());
            });
        });

        node_entites.insert(node_id.clone(), node_entity.clone());

        //Value label
        spawn_world_label(
            &mut commands,
            &node.label(&state.function_declarations),
            node_color(node),
            18.0,
            node_pos,
            Vec2::ZERO,
            AstSceneEntity,
        );

        // Type label (smaller, above)
        spawn_world_label(
            &mut commands,
            node.eval_type(&state.layout_ast.ast, &state.function_declarations)
                .to_string()
                .as_ref(),
            Color::srgba(0.3, 0.3, 0.37, 1.0),
            14.0,
            node_pos,
            Vec2::new(0.0, -22.0), // 22px above
            AstSceneEntity,
        );

        spawn_world_label(
            &mut commands,
            "X",
            Color::srgba(1.0, 1.0, 1.0, 1.0),
            18.0,
            Vec3::new(10.0, 0.0, 0.0),
            Vec2::new(0.0, -22.0), // 22px above
            AstSceneEntity,
        );
        spawn_world_label(
            &mut commands,
            "Y",
            Color::srgba(1.0, 1.0, 1.0, 1.0),
            18.0,
            Vec3::new(0.0, 10.0, 0.0),
            Vec2::new(0.0, -22.0), // 22px above
            AstSceneEntity,
        );
        spawn_world_label(
            &mut commands,
            "Z",
            Color::srgba(1.0, 1.0, 1.0, 1.0),
            18.0,
            Vec3::new(0.0, 0.0, 10.0),
            Vec2::new(0.0, -22.0), // 22px above
            AstSceneEntity,
        );
    }

    for e in state.layout_ast.edges() {
        commands.spawn(Edge {
            from_anchor: *anchor_entities.get(&e.from_anchor.anchor_id).unwrap(),
            to_anchor: *anchor_entities.get(&e.to_anchor.anchor_id).unwrap(),
        });
    }
    */

    /*
    // Translucent Z-planes for ternary branches (thin cuboids facing Z)
    let z_levels: std::collections::HashSet<i32> = state
        .nodes
        .iter()
        .map(|n| (n.pos.z * 10.0) as i32)
        .filter(|z| z.abs() > 1)
        .collect();

    let plane_mesh = meshes.add(Cuboid::new(14.0, 16.0, 0.005));
    for z_int in z_levels {
        let z = z_int as f32 / 10.0;
        let color = if z > 0.0 {
            Color::srgba(0.29, 0.87, 0.50, 0.04)
        } else {
            Color::srgba(0.973, 0.443, 0.443, 0.04)
        };
        let mat = materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            ..default()
        });
        commands.spawn((
            Mesh3d(plane_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(0.0, 0.0, z),
            AstSceneEntity,
        ));
    }
    */
}

pub fn emissive_color(color: Color) -> LinearRgba {
    let c = color.to_linear();
    LinearRgba::new(c.red * 0.15, c.green * 0.15, c.blue * 0.15, 1.0)
}
