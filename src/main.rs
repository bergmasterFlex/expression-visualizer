mod ast;
mod camera;
mod layout;

use bevy::{input::keyboard::KeyboardInput, prelude::*};

// ── WASM bridge ─────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn read_expression_from_js() -> Option<String> {
    use wasm_bindgen::JsValue;
    let window = web_sys::window()?;
    let val = js_sys::Reflect::get(&window, &JsValue::from_str("astExpression")).ok()?;
    val.as_string()
}

// ── Resources ───────────────────────────────────────────────

#[derive(Resource)]
struct AstState {
    layout_ast: layout::LayoutAst,
    function_declarations:
        std::collections::HashMap<ast::FunctionDeclarationId, ast::FunctionDeclaration>,
}

impl Default for AstState {
    fn default() -> Self {
        Self {
            layout_ast: layout::LayoutAst::empty().plus_sink(),
            function_declarations: std::collections::HashMap::from([]),
        }
    }
}

/// Marker for AST node mesh entities (so we can despawn them on rebuild).
#[derive(Component)]
struct AstNodeEntity {
    node_id: ast::AstNodeId,
}

/// Flag resource that signals the scene needs rebuilding.
#[derive(Resource, Default)]
struct NeedsRebuild(bool);

/// Marker for any spawned scene entity (cleaned on rebuild).
#[derive(Component)]
struct AstSceneEntity;

#[derive(Component)]
struct ResetButton;

#[derive(Component)]
struct AddNumberLiteralButton;

/// Stores which node is hovered / selected.
#[derive(Resource, Default)]
struct PickState {
    hovered: Option<ast::AstNodeId>,  // node_id
    selected: Option<ast::AstNodeId>, // node_id
}

/// UI text showing the selected node's info.
#[derive(Component)]
struct SelectionDisplay;

#[derive(Component)]
struct TextInput {
    value: String,
    focused: bool,
    cursor: usize,
}

#[derive(Component)]
struct TextInputDisplay;

#[derive(Component)]
struct TextInputBox;

// ── Colors ──────────────────────────────────────────────────

fn node_color(node: &ast::EAstNode) -> Color {
    match node {
        ast::EAstNode::Sink { .. } => Color::srgb(1.0, 0.0, 0.0), // #FF6B6B
        ast::EAstNode::FunctionCall { .. } => Color::srgb(0.306, 0.804, 0.769), // #4ECDC4
        ast::EAstNode::MatchTrue { .. } => Color::srgb(0.984, 0.573, 0.235), // #FB923C
        ast::EAstNode::MatchFalse { .. } => Color::srgb(0.133, 0.827, 0.933), // #22D3EE
        ast::EAstNode::BoolLiteral(true) => Color::srgb(0.29, 0.87, 0.50), // #4ADE80
        ast::EAstNode::BoolLiteral(false) => Color::srgb(0.973, 0.443, 0.443), // #F87171
        ast::EAstNode::NumLiteral(_) => Color::srgb(0.91, 0.894, 0.871), // #E8E4DE
    }
}

fn node_emissive(node: &ast::EAstNode) -> LinearRgba {
    let c = node_color(node).to_linear();
    LinearRgba::new(c.red * 0.15, c.green * 0.15, c.blue * 0.15, 1.0)
}

// ── Mesh helpers ────────────────────────────────────────────

/// Build an octahedron mesh (6 verts, 8 faces).
fn octahedron_mesh(size: f32) -> Mesh {
    let v = [
        [0.0, size, 0.0],  // 0 top
        [0.0, -size, 0.0], // 1 bottom
        [size, 0.0, 0.0],  // 2 +X
        [-size, 0.0, 0.0], // 3 -X
        [0.0, 0.0, size],  // 4 +Z
        [0.0, 0.0, -size], // 5 -Z
    ];
    let indices: Vec<u32> = vec![
        0, 4, 2, 0, 2, 5, 0, 5, 3, 0, 3, 4, 1, 2, 4, 1, 5, 2, 1, 3, 5, 1, 4, 3,
    ];
    // Compute flat normals per face
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for tri in indices.chunks(3) {
        let a = Vec3::from(v[tri[0] as usize]);
        let b = Vec3::from(v[tri[1] as usize]);
        let c = Vec3::from(v[tri[2] as usize]);
        let normal = (b - a).cross(c - a).normalize();
        for &idx in tri {
            positions.push(v[idx as usize]);
            normals.push(normal.to_array());
        }
    }
    let flat_indices: Vec<u32> = (0..positions.len() as u32).collect();

    Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        bevy::render::render_asset::RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(bevy::render::mesh::Indices::U32(flat_indices))
}

// ── Systems ─────────────────────────────────────────────────

/// Initial scene setup: camera, lights, ambient.
fn setup_scene(mut commands: Commands) {
    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 5.0, 12.0).looking_at(Vec3::ZERO, Vec3::Y),
            camera: Camera {
                clear_color: ClearColorConfig::Custom(Color::srgb(0.031, 0.031, 0.102)),
                ..default()
            },
            ..default()
        },
        camera::OrbitCameraTag,
    ));

    // Directional light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(5.0, 10.0, 7.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.25, 0.25, 0.38),
        brightness: 200.0,
    });

    // Two colored point lights
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            color: Color::srgb(0.133, 0.827, 0.933),
            intensity: 50_000.0,
            range: 30.0,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 5.0, 8.0),
        ..default()
    });
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            color: Color::srgb(1.0, 0.42, 0.42),
            intensity: 30_000.0,
            range: 30.0,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 5.0, -8.0),
        ..default()
    });
}

/// Spawn the AST node meshes.
fn spawn_ast_nodes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<AstState>,
) {
    let sphere_mesh = meshes.add(Sphere::new(0.32));
    let cube_mesh = meshes.add(Cuboid::new(0.45, 0.45, 0.45));
    let octa_mesh = meshes.add(octahedron_mesh(0.38));
    let ring_mesh = meshes.add(Torus::new(0.025, 0.38));
    let ring_big_mesh = meshes.add(Torus::new(0.025, 0.48));

    for (node_id, node) in &state.layout_ast.ast.nodes {
        let color = node_color(node);
        let emissive = node_emissive(&node);

        // Pick shape based on AST type
        let mesh = match node {
            //ast::EAstNode::TernaryExpr { .. } => octa_mesh.clone(),
            //ast::EAstNode::BoolLiteral(_) => cube_mesh.clone(),
            _ => sphere_mesh.clone(),
        };

        let material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.05, 0.05, 0.10),
            emissive,
            metallic: 0.3,
            perceptual_roughness: 0.6,
            ..default()
        });

        let node_pos = state.layout_ast.layout_nodes.get(node_id).unwrap().pos;

        // Node body
        commands.spawn((
            PbrBundle {
                mesh,
                material,
                transform: Transform::from_translation(node_pos),
                ..default()
            },
            AstNodeEntity {
                node_id: node_id.clone(),
            },
            AstSceneEntity,
        ));

        /*
        // Ring
        let is_ternary = matches!(node.ast, ast::AstNode::TernaryExpr { .. });
        let ring = if is_ternary {
            ring_big_mesh.clone()
        } else {
            ring_mesh.clone()
        };
        let ring_mat = materials.add(StandardMaterial {
            base_color: color.with_alpha(0.7),
            emissive: LinearRgba::new(
                emissive.red * 2.0,
                emissive.green * 2.0,
                emissive.blue * 2.0,
                1.0,
            ),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            ..default()
        });

        let mut ring_transform = Transform::from_translation(node.pos);
        if is_ternary {
            ring_transform.rotate_x(std::f32::consts::FRAC_PI_4);
        }
        commands.spawn((
            PbrBundle {
                mesh: ring.clone(),
                material: ring_mat.clone(),
                transform: ring_transform,
                ..default()
            },
            AstSceneEntity,
        ));

        // Second ring for ternary (crossed)
        if is_ternary {
            let mut t2 = Transform::from_translation(node.pos);
            t2.rotate_x(-std::f32::consts::FRAC_PI_4);
            commands.spawn((
                PbrBundle {
                    mesh: ring,
                    material: ring_mat,
                    transform: t2,
                    ..default()
                },
                AstSceneEntity,
            ));
        }
        */

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

        /*
        // Type label (smaller, above)
        spawn_world_label(
            &mut commands,
            node.ast.type_name(),
            Color::srgba(0.3, 0.3, 0.37, 1.0),
            14.0,
            node.pos,
            Vec2::new(0.0, -22.0), // 22px above
            AstSceneEntity,
        );
        */

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

fn spawn_ui(mut commands: Commands) {
    commands
        .spawn((
            ButtonBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    top: Val::Px(12.0),
                    left: Val::Px(12.0),
                    padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
                    ..default()
                },
                background_color: Color::srgba(0.16, 0.16, 0.22, 0.9).into(),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            ResetButton,
        ))
        .with_children(|parent| {
            parent.spawn(TextBundle::from_section(
                "reset",
                TextStyle {
                    font_size: 14.0,
                    color: Color::srgb(0.6, 0.6, 0.7),
                    ..default()
                },
            ));
        });

    commands
        .spawn((
            ButtonBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    top: Val::Px(48.0),
                    left: Val::Px(12.0),
                    padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
                    ..default()
                },
                background_color: Color::srgba(0.16, 0.16, 0.22, 0.9).into(),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            AddNumberLiteralButton,
        ))
        .with_children(|parent| {
            parent.spawn(TextBundle::from_section(
                "Add Number",
                TextStyle {
                    font_size: 14.0,
                    color: Color::srgb(0.6, 0.6, 0.7),
                    ..default()
                },
            ));
        });
}

fn handle_reset_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (With<Interaction>, With<ResetButton>),
    >,
    mut text_q: Query<&mut Text>,
    mut state: ResMut<AstState>,
    mut orbit: ResMut<camera::OrbitCamera>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut commands: Commands,
    scene_entities: Query<Entity, With<AstSceneEntity>>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut text = text_q.get_mut(children[0]).unwrap();

        match *interaction {
            Interaction::Pressed => {
                /*
                // Reset expression
                let expr = "2 + 2".to_string();
                let tree = crate::ast::parse(&expr);
                let (nodes, edges) = crate::layout::compute_layout(&tree);
                state.expression = expr;
                state.nodes = nodes;
                state.edges = edges;

                for entity in scene_entities.iter() {
                    commands.entity(entity).despawn_recursive();
                }
                rebuild.0 = true;
                orbit.auto_rotate = true;
                orbit.theta = 0.6;
                orbit.phi = 1.0;
                orbit.radius = 7.0;

                bg.0 = Color::srgba(0.1, 0.1, 0.2, 0.95);
                text.sections[0].style.color = Color::srgb(0.133, 0.827, 0.933);
                */
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.2, 0.2, 0.3, 0.95);
                text.sections[0].style.color = Color::srgb(0.85, 0.85, 0.9);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.16, 0.16, 0.22, 0.9);
                text.sections[0].style.color = Color::srgb(0.6, 0.6, 0.7);
            }
        }
    }
}

fn handle_add_number_literal_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (With<Interaction>, With<AddNumberLiteralButton>),
    >,
    mut text_q: Query<&mut Text>,
    mut state: ResMut<AstState>,
    mut orbit: ResMut<camera::OrbitCamera>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut commands: Commands,
    scene_entities: Query<Entity, With<AstSceneEntity>>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut text = text_q.get_mut(children[0]).unwrap();

        match *interaction {
            Interaction::Pressed => {
                // Reset expression
                state.layout_ast = state.layout_ast.plus_number_literal(7);

                rebuild.0 = true;
                /*
                orbit.auto_rotate = true;
                orbit.theta = 0.6;
                orbit.phi = 1.0;
                orbit.radius = 7.0;

                bg.0 = Color::srgba(0.1, 0.1, 0.2, 0.95);
                text.sections[0].style.color = Color::srgb(0.133, 0.827, 0.933);
            */
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.2, 0.2, 0.3, 0.95);
                text.sections[0].style.color = Color::srgb(0.85, 0.85, 0.9);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.16, 0.16, 0.22, 0.9);
                text.sections[0].style.color = Color::srgb(0.6, 0.6, 0.7);
            }
        }
    }
}

/// Draw edges using Gizmos (called every frame).
fn draw_edges(mut gizmos: Gizmos, state: Res<AstState>) {
    let edges = &state.layout_ast.edges();
    /*
    let edges = edges
        .into_iter()
        .chain(
            vec![
                layout::LayoutEdge {
                    from_id: 0,
                    to_id: 0,
                    from_pos: Vec3::new(0.0, 0.0, 0.0),
                    to_pos: Vec3::new(10.0, 0.0, 0.0),
                    label: "X",
                    dir: layout::EdgeDir::Up,
                },
                layout::LayoutEdge {
                    from_id: 0,
                    to_id: 0,
                    from_pos: Vec3::new(0.0, 0.0, 0.0),
                    to_pos: Vec3::new(0.0, 10.0, 0.0),
                    label: "Y",
                    dir: layout::EdgeDir::Up,
                },
                layout::LayoutEdge {
                    from_id: 0,
                    to_id: 0,
                    from_pos: Vec3::new(0.0, 0.0, 0.0),
                    to_pos: Vec3::new(0.0, 0.0, 10.0),
                    label: "Z",
                    dir: layout::EdgeDir::Up,
                },
            ]
            .into_iter(),
        )
        .collect::<Vec<layout::LayoutEdge>>();
    */
    for edge in edges {
        let from = Vec3::from(edge.from_pos);
        let to = edge.to_pos;

        // Determine start/end offsets along Y
        let node_radius = 0.4;
        let start = from;
        let end = to;

        // Determine color
        //let from_node = state.nodes.iter().find(|n| n.id == edge.from_id);
        let color = Color::srgba(0.29, 0.87, 0.50, 0.55);

        // Sample a cubic bezier for a smooth curve
        let mid_y = (start.y + end.y) / 2.0;
        let mid_z = (start.z + end.z) / 2.0;
        let cp1 = Vec3::new(start.x, mid_y, mid_z);
        let cp2 = Vec3::new(end.x, mid_y, mid_z);

        let segments = 20;
        let mut prev = start;
        for i in 1..=segments {
            let t = i as f32 / segments as f32;
            let it = 1.0 - t;
            let p = start * it * it * it
                + cp1 * 3.0 * it * it * t
                + cp2 * 3.0 * it * t * t
                + end * t * t * t;
            gizmos.line(prev, p, color);
            prev = p;
        }

        // Small arrow cone at the end (approximate with short lines)
        let dir = (end - cp2).normalize();
        let perp1 = dir.cross(Vec3::Z).normalize_or_zero() * 0.08;
        let perp2 = dir.cross(Vec3::X).normalize_or_zero() * 0.08;
        let arrow_base = end - dir * 0.2;
        gizmos.line(end, arrow_base + perp1, color);
        gizmos.line(end, arrow_base - perp1, color);
        gizmos.line(end, arrow_base + perp2, color);
        gizmos.line(end, arrow_base - perp2, color);
    }
}

/// Gentle pulsing animation for nodes.
fn animate_nodes(time: Res<Time>, mut query: Query<(&AstNodeEntity, &mut Transform)>) {
    /*
    let t = time.elapsed_seconds();
    for (node_ent, mut transform) in query.iter_mut() {
        let pulse = 1.0 + 0.04 * (t * 2.0 + node_ent.node_id as f32 * 1.5).sin();
        transform.scale = Vec3::splat(pulse);
    }
    */
}

/// If rebuild was requested, respawn the AST scene.
fn rebuild_scene(
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<StandardMaterial>>,
    state: Res<AstState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    if rebuild.0 {
        spawn_ast_nodes(commands, meshes, materials, state);
        rebuild.0 = false;
    }
}

#[derive(Component)]
pub struct WorldLabel {
    pub world_pos: Vec3,
    pub offset: Vec2, // screen-space pixel offset
}

/// Spawn a UI text label that tracks a world position.
fn spawn_world_label(
    commands: &mut Commands,
    text: &str,
    color: Color,
    font_size: f32,
    world_pos: Vec3,
    offset: Vec2,
    marker: impl Bundle,
) -> Entity {
    commands
        .spawn((
            TextBundle {
                text: Text::from_section(
                    text,
                    TextStyle {
                        font_size,
                        color,
                        ..default()
                    },
                ),
                style: Style {
                    position_type: PositionType::Absolute,
                    ..default()
                },
                visibility: Visibility::Hidden,
                ..default()
            },
            WorldLabel { world_pos, offset },
            marker,
        ))
        .id()
}

/// Each frame, project world positions → screen and reposition the text.
fn update_world_labels(
    camera_q: Query<(&Camera, &GlobalTransform), With<camera::OrbitCameraTag>>,
    mut label_q: Query<(&WorldLabel, &mut Style, &mut Visibility, &Node)>,
) {
    let Ok((camera, cam_gt)) = camera_q.get_single() else {
        return;
    };

    for (label, mut style, mut vis, node) in label_q.iter_mut() {
        if let Some(screen_pos) = camera.world_to_viewport(cam_gt, label.world_pos) {
            let size = node.size();
            style.left = Val::Px(screen_pos.x - size.x / 2.0 + label.offset.x);
            style.top = Val::Px(screen_pos.y - size.y / 2.0 + label.offset.y);
            *vis = Visibility::Visible;
        } else {
            // Behind camera
            *vis = Visibility::Hidden;
        }
    }
}

fn spawn_selection_display(mut commands: Commands) {
    commands.spawn((
        TextBundle {
            text: Text::from_section(
                "",
                TextStyle {
                    font_size: 16.0,
                    color: Color::srgb(0.85, 0.85, 0.9),
                    ..default()
                },
            ),
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Px(14.0),
                right: Val::Px(14.0),
                ..default()
            },
            ..default()
        },
        SelectionDisplay,
    ));
}

fn pick_nodes(
    camera_q: Query<(&Camera, &GlobalTransform), With<camera::OrbitCameraTag>>,
    windows: Query<&Window>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut pick: ResMut<PickState>,
    node_q: Query<(&AstNodeEntity, &Transform)>,
) {
    let Ok((camera, cam_gt)) = camera_q.get_single() else {
        return;
    };
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        pick.hovered = None;
        return;
    };

    // Build ray from camera through cursor
    let Some(ray) = camera.viewport_to_world(cam_gt, cursor) else {
        pick.hovered = None;
        return;
    };

    // Test intersection with each node (sphere test, radius 0.35)
    let radius = 0.35_f32;
    let mut closest: Option<(ast::AstNodeId, f32)> = None;

    for (node_ent, transform) in node_q.iter() {
        let center = transform.translation;
        let oc = ray.origin - center;
        let b = oc.dot(*ray.direction);
        let c = oc.dot(oc) - radius * radius;
        let disc = b * b - c;

        if disc >= 0.0 {
            let t = -b - disc.sqrt();
            if t > 0.0 {
                if closest.is_none() || t < closest.clone().unwrap().1 {
                    closest = Some((node_ent.node_id.clone(), t));
                }
            }
        }
    }

    pick.hovered = closest.clone().map(|(id, _)| id);

    if mouse.just_pressed(MouseButton::Left) {
        pick.selected = closest.map(|(id, _)| id);
    }
}

fn highlight_hovered(
    pick: Res<PickState>,
    node_q: Query<(&AstNodeEntity, &Handle<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<AstState>,
) {
    for (node_ent, mat_handle) in node_q.iter() {
        let Some(mat) = materials.get_mut(mat_handle) else {
            continue;
        };
        let Some((_, layout_node)) = state
            .layout_ast
            .ast
            .nodes
            .iter()
            .find(|(n_id, _)| **n_id == node_ent.node_id)
        else {
            continue;
        };

        let base = node_emissive(layout_node);
        let is_hovered = pick.hovered == Some(node_ent.node_id.clone());
        let is_selected = pick.selected == Some(node_ent.node_id.clone());

        let intensity = if is_hovered {
            4.0
        } else if is_selected {
            2.5
        } else {
            1.0
        };

        mat.emissive = LinearRgba::new(
            base.red * intensity,
            base.green * intensity,
            base.blue * intensity,
            1.0,
        );
    }
}

fn update_selection_display(
    pick: Res<PickState>,
    state: Res<AstState>,
    mut display_q: Query<&mut Text, With<SelectionDisplay>>,
) {
    let Ok(mut text) = display_q.get_single_mut() else {
        return;
    };
    if let Some(id) = &pick.selected {
        if let Some(node) = state.layout_ast.ast.nodes.get(&id) {
            text.sections[0].value = format!(
                "{} : <unknown type>",
                node.label(&state.function_declarations)
            );
            text.sections[0].style.color = node_color(&node);
        }
    } else {
        text.sections[0].value.clear();
    }
}

fn update_cursor(pick: Res<PickState>, mut windows: Query<&mut Window>) {
    let Ok(mut window) = windows.get_single_mut() else {
        return;
    };
    window.cursor.icon = if pick.hovered.is_some() {
        CursorIcon::Pointer
    } else {
        CursorIcon::Default
    };
}

fn spawn_text_input(commands: &mut Commands, initial: &str) {
    // Outer container (clickable background)
    commands
        .spawn((
            ButtonBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    top: Val::Px(12.0),
                    left: Val::Px(90.0), // next to reset button
                    padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                    min_width: Val::Px(220.0),
                    border: UiRect::all(Val::Px(1.5)),
                    ..default()
                },
                background_color: Color::srgba(0.06, 0.06, 0.12, 0.9).into(),
                border_color: Color::srgb(0.12, 0.12, 0.24).into(),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            TextInputBox,
            TextInput {
                value: initial.to_string(),
                focused: false,
                cursor: initial.len(),
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                TextBundle::from_section(
                    initial,
                    TextStyle {
                        font_size: 14.0,
                        color: Color::srgb(0.91, 0.89, 0.87),
                        ..default()
                    },
                ),
                TextInputDisplay,
            ));
        });
}

fn text_input_focus(
    mut input_q: Query<(&Interaction, &mut TextInput, &mut BorderColor), With<TextInputBox>>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    let clicked_outside = mouse.just_pressed(MouseButton::Left);

    for (interaction, mut input, mut border) in input_q.iter_mut() {
        if *interaction == Interaction::Pressed {
            input.focused = true;
        } else if clicked_outside && *interaction == Interaction::None {
            input.focused = false;
        }

        if keys.just_pressed(KeyCode::Escape) {
            input.focused = false;
        }

        // Visual feedback
        border.0 = if input.focused {
            Color::srgb(0.133, 0.827, 0.933) // cyan when focused
        } else {
            Color::srgb(0.12, 0.12, 0.24)
        };
    }
}

fn text_input_keyboard(
    mut input_q: Query<(&mut TextInput, &Children), With<TextInputBox>>,
    mut text_q: Query<&mut Text, With<TextInputDisplay>>,
    mut key_events: EventReader<KeyboardInput>,
    mut state: ResMut<AstState>,
    mut orbit: ResMut<camera::OrbitCamera>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut commands: Commands,
    scene_entities: Query<Entity, With<AstSceneEntity>>,
) {
    for (mut input, children) in input_q.iter_mut() {
        if !input.focused {
            continue;
        }

        let mut changed = false;

        for ev in key_events.read() {
            if ev.state != bevy::input::ButtonState::Pressed {
                continue;
            }

            let input_cursor = input.cursor;

            match &ev.logical_key {
                bevy::input::keyboard::Key::Character(s) => {
                    input.value.insert_str(input_cursor, s.as_str());
                    input.cursor += s.len();
                    changed = true;
                }
                bevy::input::keyboard::Key::Space => {
                    input.value.insert(input_cursor, ' ');
                    input.cursor += 1;
                    changed = true;
                }
                bevy::input::keyboard::Key::Backspace => {
                    if input.cursor > 0 {
                        let prev = input.value[..input.cursor]
                            .char_indices()
                            .last()
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        input.value.remove(prev);
                        input.cursor = prev;
                        changed = true;
                    }
                }
                bevy::input::keyboard::Key::Delete => {
                    if input.cursor < input.value.len() {
                        input.value.remove(input_cursor);
                        changed = true;
                    }
                }
                bevy::input::keyboard::Key::ArrowLeft => {
                    if input.cursor > 0 {
                        input.cursor = input.value[..input.cursor]
                            .char_indices()
                            .last()
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                    }
                }
                bevy::input::keyboard::Key::ArrowRight => {
                    if input.cursor < input.value.len() {
                        input.cursor += input.value[input.cursor..]
                            .chars()
                            .next()
                            .map(|c| c.len_utf8())
                            .unwrap_or(0);
                    }
                }
                bevy::input::keyboard::Key::Home => {
                    input.cursor = 0;
                }
                bevy::input::keyboard::Key::End => {
                    input.cursor = input.value.len();
                }
                bevy::input::keyboard::Key::Escape => {
                    input.focused = false;
                }
                _ => {}
            }
        }

        // Update display text with blinking cursor
        if let Ok(mut text) = text_q.get_mut(children[0]) {
            let (before, after) = input.value.split_at(input.cursor);
            text.sections[0].value = if input.focused {
                format!("{}|{}", before, after)
            } else {
                input.value.clone()
            };
        }

        // Rebuild AST on change
        if changed && !input.value.is_empty() {
            /*
            let tree = crate::ast::parse(&input.value);
            let (nodes, edges) = crate::layout::compute_layout(&tree);
            state.expression = input.value.clone();
            state.nodes = nodes;
            state.edges = edges;

            for entity in scene_entities.iter() {
                commands.entity(entity).despawn_recursive();
            }
            */
            rebuild.0 = true;
            //orbit.auto_rotate = true;
            orbit.theta = 0.6;
            orbit.phi = 1.0;
        }
    }
}

// ── App entry ───────────────────────────────────────────────

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "AST Visualizer 3D — Bevy + WebGL".into(),
                    canvas: Some("#bevy-canvas".into()),
                    fit_canvas_to_parent: true,
                    prevent_default_event_handling: true,
                    present_mode: bevy::window::PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
            camera::OrbitCameraPlugin,
        ))
        .init_resource::<AstState>()
        .init_resource::<NeedsRebuild>()
        .init_resource::<PickState>()
        .add_systems(
            Startup,
            (
                setup_scene,
                spawn_ast_nodes,
                spawn_ui,
                spawn_selection_display,
            )
                .chain(),
        )
        .add_systems(Startup, |mut commands: Commands| {
            spawn_text_input(&mut commands, "(5 > 3) ? 10 + 1 : 20 - 5")
        })
        .add_systems(
            Update,
            (
                draw_edges,
                animate_nodes,
                handle_reset_button,
                handle_add_number_literal_button,
                pick_nodes,
                highlight_hovered,
                update_selection_display,
                update_cursor,
                text_input_focus,
                text_input_keyboard,
                (rebuild_scene).chain(),
            ),
        )
        .add_systems(Update, update_world_labels)
        .run();
}
