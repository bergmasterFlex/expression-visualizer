mod ast;
mod camera;
mod layout;

use bevy::prelude::*;
use camera::{OrbitCamera, OrbitCameraPlugin, OrbitCameraTag};
use layout::{EdgeDir, LayoutEdge, LayoutNode};

// ── WASM bridge ─────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn read_expression_from_js() -> Option<String> {
    use wasm_bindgen::JsValue;
    let window = web_sys::window()?;
    let val = js_sys::Reflect::get(&window, &JsValue::from_str("astExpression")).ok()?;
    val.as_string()
}

#[cfg(not(target_arch = "wasm32"))]
fn read_expression_from_js() -> Option<String> {
    None // desktop: use default
}

// ── Resources ───────────────────────────────────────────────

#[derive(Resource)]
struct AstState {
    expression: String,
    nodes: Vec<LayoutNode>,
    edges: Vec<LayoutEdge>,
}

impl Default for AstState {
    fn default() -> Self {
        let expr = "(5 > 3) ? 10 + 1 : 20 - 5".to_string();
        let tree = ast::parse(&expr);
        let (nodes, edges) = layout::compute_layout(&tree);
        Self { expression: expr, nodes, edges }
    }
}

/// Marker for AST node mesh entities (so we can despawn them on rebuild).
#[derive(Component)]
struct AstNodeEntity {
    node_id: usize,
}

/// Flag resource that signals the scene needs rebuilding.
#[derive(Resource, Default)]
struct NeedsRebuild(bool);

/// Marker for any spawned scene entity (cleaned on rebuild).
#[derive(Component)]
struct AstSceneEntity;

// ── Colors ──────────────────────────────────────────────────

fn node_color(node: &ast::AstNode) -> Color {
    match node {
        ast::AstNode::BinaryExpr { op: '+', .. } => Color::srgb(1.0, 0.42, 0.42),     // #FF6B6B
        ast::AstNode::BinaryExpr { op: '-', .. } => Color::srgb(0.306, 0.804, 0.769),  // #4ECDC4
        ast::AstNode::BinaryExpr { op: '*', .. } => Color::srgb(1.0, 0.9, 0.427),      // #FFE66D
        ast::AstNode::BinaryExpr { op: '/', .. } => Color::srgb(0.655, 0.545, 0.98),   // #A78BFA
        ast::AstNode::BinaryExpr { .. }          => Color::srgb(1.0, 0.42, 0.42),
        ast::AstNode::ComparisonExpr { .. }      => Color::srgb(0.984, 0.573, 0.235),   // #FB923C
        ast::AstNode::TernaryExpr { .. }         => Color::srgb(0.133, 0.827, 0.933),   // #22D3EE
        ast::AstNode::BoolLiteral(true)          => Color::srgb(0.29, 0.87, 0.50),      // #4ADE80
        ast::AstNode::BoolLiteral(false)         => Color::srgb(0.973, 0.443, 0.443),   // #F87171
        ast::AstNode::NumLiteral(_)              => Color::srgb(0.91, 0.894, 0.871),    // #E8E4DE
    }
}

fn node_emissive(node: &ast::AstNode) -> LinearRgba {
    let c = node_color(node).to_linear();
    LinearRgba::new(c.red * 0.15, c.green * 0.15, c.blue * 0.15, 1.0)
}

// ── Mesh helpers ────────────────────────────────────────────

/// Build an octahedron mesh (6 verts, 8 faces).
fn octahedron_mesh(size: f32) -> Mesh {
    let v = [
        [0.0, size, 0.0],     // 0 top
        [0.0, -size, 0.0],    // 1 bottom
        [size, 0.0, 0.0],     // 2 +X
        [-size, 0.0, 0.0],    // 3 -X
        [0.0, 0.0, size],     // 4 +Z
        [0.0, 0.0, -size],    // 5 -Z
    ];
    let indices: Vec<u32> = vec![
        0, 4, 2,  0, 2, 5,  0, 5, 3,  0, 3, 4,
        1, 2, 4,  1, 5, 2,  1, 3, 5,  1, 4, 3,
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
        OrbitCameraTag,
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

    for node in &state.nodes {
        let color = node_color(&node.ast);
        let emissive = node_emissive(&node.ast);

        // Pick shape based on AST type
        let mesh = match &node.ast {
            ast::AstNode::TernaryExpr { .. } => octa_mesh.clone(),
            ast::AstNode::BoolLiteral(_) => cube_mesh.clone(),
            _ => sphere_mesh.clone(),
        };

        let material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.05, 0.05, 0.10),
            emissive,
            metallic: 0.3,
            perceptual_roughness: 0.6,
            ..default()
        });

        // Node body
        commands.spawn((
            PbrBundle {
                mesh,
                material,
                transform: Transform::from_translation(node.pos),
                ..default()
            },
            AstNodeEntity { node_id: node.id },
            AstSceneEntity,
        ));

        // Ring
        let is_ternary = matches!(node.ast, ast::AstNode::TernaryExpr { .. });
        let ring = if is_ternary { ring_big_mesh.clone() } else { ring_mesh.clone() };
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
    }

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
}

/// Draw edges using Gizmos (called every frame).
fn draw_edges(mut gizmos: Gizmos, state: Res<AstState>) {
    for edge in &state.edges {
        let from = edge.from_pos;
        let to = edge.to_pos;
        let is_ternary_edge = matches!(edge.label, "then" | "else" | "cond");

        // Determine start/end offsets along Y
        let node_radius = 0.4;
        let start = if edge.dir == EdgeDir::Up {
            from + Vec3::Y * node_radius
        } else {
            from - Vec3::Y * node_radius
        };
        let end = if edge.dir == EdgeDir::Up {
            to - Vec3::Y * node_radius
        } else {
            to + Vec3::Y * node_radius
        };

        // Determine color
        let from_node = state.nodes.iter().find(|n| n.id == edge.from_id);
        let color = match edge.label {
            "then" => Color::srgba(0.29, 0.87, 0.50, 0.55),
            "else" => Color::srgba(0.973, 0.443, 0.443, 0.55),
            "cond" => Color::srgba(0.133, 0.827, 0.933, 0.55),
            _ => {
                if let Some(n) = from_node {
                    node_color(&n.ast).with_alpha(0.3)
                } else {
                    Color::srgba(0.3, 0.3, 0.4, 0.3)
                }
            }
        };

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
fn animate_nodes(
    time: Res<Time>,
    mut query: Query<(&AstNodeEntity, &mut Transform)>,
) {
    let t = time.elapsed_seconds();
    for (node_ent, mut transform) in query.iter_mut() {
        let pulse = 1.0 + 0.04 * (t * 2.0 + node_ent.node_id as f32 * 1.5).sin();
        transform.scale = Vec3::splat(pulse);
    }
}

/// Poll the JS bridge for expression changes and rebuild the tree if needed.
fn poll_expression(
    mut commands: Commands,
    mut state: ResMut<AstState>,
    mut orbit: ResMut<OrbitCamera>,
    mut rebuild: ResMut<NeedsRebuild>,
    scene_entities: Query<Entity, With<AstSceneEntity>>,
) {
    if let Some(new_expr) = read_expression_from_js() {
        if new_expr != state.expression && !new_expr.is_empty() {
            state.expression = new_expr.clone();
            let tree = ast::parse(&new_expr);
            let (nodes, edges) = layout::compute_layout(&tree);
            state.nodes = nodes;
            state.edges = edges;

            // Despawn old scene entities
            for entity in scene_entities.iter() {
                commands.entity(entity).despawn_recursive();
            }

            // Reset camera
            orbit.auto_rotate = true;
            orbit.theta = 0.6;
            orbit.phi = 1.0;

            // Compute new radius
            let max_spread = state.nodes.iter()
                .map(|n| n.pos.length())
                .fold(0.0f32, f32::max);
            orbit.radius = (max_spread * 1.3 + 4.0).max(7.0);

            rebuild.0 = true;
        }
    }
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
                    ..default()
                }),
                ..default()
            }),
            OrbitCameraPlugin,
        ))
        .init_resource::<AstState>()
        .init_resource::<NeedsRebuild>()
        .add_systems(Startup, (setup_scene, spawn_ast_nodes).chain())
        .add_systems(Update, (
            draw_edges,
            animate_nodes,
            (poll_expression, rebuild_scene).chain(),
        ))
        .run();
}
