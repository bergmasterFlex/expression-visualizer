use bevy::prelude::*;

use crate::ast::FunctionDeclaration;

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
}

pub struct RenderLabel {
    pub text: String,
    pub color: Color,
    pub font_size: f32,
    pub world_pos: Vec3,
    pub offset: Vec2,
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
                            transform: node_pos_tf
                                * Transform::from_translation(Vec3::new(0.0, 0.0, 0.55)),
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
                                * Transform::from_translation(Vec3::new(0.0, 0.0, 0.55))
                                * Transform::from_scale(Vec3::splat(1.8)),
                        },
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
                                transform: node_pos_tf
                                    * Transform::from_translation(Vec3::new(0.0, 0.0, -0.55)),
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
                                    * Transform::from_translation(Vec3::new(0.0, 0.0, -0.55))
                                    * Transform::from_scale(Vec3::splat(1.8)),
                            },
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
                                transform: node_pos_tf
                                    * Transform::from_translation(Vec3::new(0.0, 0.0, 0.55)),
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
                                    * Transform::from_translation(Vec3::new(0.0, 0.0, 0.55))
                                    * Transform::from_scale(Vec3::splat(1.8)),
                            },
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
        crate::ast::node::ENode::Match {
            input_anchor,
            output_anchor,
        } => {
            let color = Color::srgb(0.9, 0.0, 0.0);
            RenderNode {
                node: RenderObject {
                    mesh: crate::mesh::create_bool_mesh(0.5, 1.0, 16),
                    material: StandardMaterial {
                        base_color: color,
                        emissive: emissive_color(color),
                        metallic: 0.3,
                        perceptual_roughness: 0.6,
                        ..default()
                    },
                    transform: node_pos_tf
                        * Transform::from_rotation(Quat::from_axis_angle(
                            Vec3::new(1.0, 0.0, 0.0),
                            std::f32::consts::PI,
                        )),
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
                                transform: node_pos_tf
                                    * Transform::from_translation(Vec3::new(0.0, 0.0, -0.55)),
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
                                    * Transform::from_translation(Vec3::new(0.0, 0.0, -0.55))
                                    * Transform::from_scale(Vec3::splat(1.8)),
                            },
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
                                transform: node_pos_tf
                                    * Transform::from_translation(Vec3::new(0.0, 0.0, 0.55)),
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
                                    * Transform::from_translation(Vec3::new(0.0, 0.0, 0.55))
                                    * Transform::from_scale(Vec3::splat(1.8)),
                            },
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
            input_anchors,
            output_anchor,
            ..
        } => RenderNode {
            node: RenderObject {
                mesh: crate::mesh::create_cone_mesh(0.5, 1.0, 4),
                material: StandardMaterial {
                    base_color: Color::srgb(0.5, 0.9, 1.0),
                    emissive: LinearRgba::new(0.2, 0.5, 0.8, 1.0),
                    unlit: true,
                    ..default()
                },
                transform: node_pos_tf
                    * Transform::from_rotation(Quat::from_axis_angle(
                        Vec3::new(1.0, 0.0, 0.0),
                        std::f32::consts::PI * -0.5,
                    ))
                    * Transform::from_rotation(Quat::from_axis_angle(
                        Vec3::new(0.0, 1.0, 0.0),
                        std::f32::consts::PI * 0.25,
                    )),
            },
            anchors: input_anchors
                .iter()
                .enumerate()
                .map(|(i_anchor, anchor_id)| {
                    let spread = 0.3;
                    let start_x = -(input_anchors.len() as f32 - 1.0) * spread / 2.0;
                    let x = start_x + i_anchor as f32 * spread;
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
                                    * Transform::from_translation(Vec3::new(x, 0.0, 0.55)),
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
                                    * Transform::from_translation(Vec3::new(x, 0.0, 0.55))
                                    * Transform::from_scale(Vec3::splat(1.8)),
                            },
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
                                * Transform::from_translation(Vec3::new(0.0, 0.0, -0.55)),
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
                                * Transform::from_translation(Vec3::new(0.0, 0.0, -0.55))
                                * Transform::from_scale(Vec3::splat(1.8)),
                        },
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
        },
        crate::ast::node::ENode::Sink { input_anchor } => {
            let color = Color::srgb(0.9, 0.0, 0.0);
            RenderNode {
                node: RenderObject {
                    mesh: Torus::new(0.225, 0.38).mesh().build(),
                    material: StandardMaterial {
                        base_color: color,
                        emissive: emissive_color(color),
                        metallic: 0.3,
                        perceptual_roughness: 0.6,
                        ..default()
                    },
                    transform: Transform::from_scale(Vec3::new(2.0, 2.0, 2.0))
                        * node_pos_tf
                        * Transform::from_rotation(Quat::from_axis_angle(
                            Vec3::new(1.0, 0.0, 0.0),
                            std::f32::consts::PI * 0.5,
                        )),
                },
                anchors: std::collections::HashMap::from([(
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
                            transform: node_pos_tf
                                * Transform::from_translation(Vec3::new(0.0, 0.0, -0.55)),
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
                                * Transform::from_translation(Vec3::new(0.0, 0.0, -0.55))
                                * Transform::from_scale(Vec3::splat(1.8)),
                        },
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
            PbrBundle {
                mesh: plane_mesh.clone(),
                material: mat,
                transform: Transform::from_xyz(0.0, 0.0, z),
                ..default()
            },
            AstSceneEntity,
        ));
    }
    */
}

pub fn emissive_color(color: Color) -> LinearRgba {
    let c = color.to_linear();
    LinearRgba::new(c.red * 0.15, c.green * 0.15, c.blue * 0.15, 1.0)
}
