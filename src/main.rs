mod ast;
mod camera;
mod eval;
mod grid;
mod layout;
mod mesh;
mod render;

use std::{collections::hash_map, f32::consts::PI};

use ast::FunctionParameterDeclaration;
use bevy::{input::keyboard::KeyboardInput, math::VectorSpace, prelude::*};

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

#[derive(Resource, Default)]
struct CurrentInputString(String);

impl Default for AstState {
    fn default() -> Self {
        Self {
            layout_ast: layout::LayoutAst::empty().plus_sink(),
            function_declarations: std::collections::HashMap::from([
                (
                    ast::FunctionDeclarationId(0),
                    ast::FunctionDeclaration {
                        name: "+".to_string(),
                        inputs: vec![
                            FunctionParameterDeclaration {
                                name: "summand1".to_string(),
                                r#type: eval::EType::Int(None),
                            },
                            FunctionParameterDeclaration {
                                name: "summand2".to_string(),
                                r#type: eval::EType::Int(None),
                            },
                        ],
                        output_type: eval::EType::Int(None),
                    },
                ),
                (
                    ast::FunctionDeclarationId(1),
                    ast::FunctionDeclaration {
                        name: "/".to_string(),
                        inputs: vec![
                            FunctionParameterDeclaration {
                                name: "dividend".to_string(),
                                r#type: eval::EType::Int(None),
                            },
                            FunctionParameterDeclaration {
                                name: "divisor".to_string(),
                                r#type: eval::EType::Int(None),
                            },
                        ],
                        output_type: eval::EType::SumType(vec![
                            eval::EType::Float(None),
                            eval::EType::Undefined,
                        ]),
                    },
                ),
            ]),
        }
    }
}

#[derive(Component)]
pub struct AnchorHovered;

#[derive(Component)]
pub enum EAnchor {
    Input {
        id: ast::AnchorId,
        render_objects: render::RenderAnchor,
    },
    Output {
        id: ast::AnchorId,
        render_objects: render::RenderAnchor,
    },
}

impl EAnchor {
    pub fn id(&self) -> ast::AnchorId {
        match self {
            EAnchor::Input { id, .. } | EAnchor::Output { id, .. } => id.clone(),
        }
    }
}

#[derive(Component)]
pub struct Edge {
    pub from_anchor: Entity,
    pub to_anchor: Entity,
}

pub struct DragInfo {
    pub source_anchor: Entity,
    pub source_anchor_id: ast::AnchorId,
    pub source_pos: Vec3,
    pub current_end: Vec3,
    pub target_anchor: Option<Entity>,
    pub target_anchor_id: Option<ast::AnchorId>,
}

#[derive(Resource, Default)]
pub struct DragState {
    pub active: Option<DragInfo>,
}

/// Vorberechnete Materials für Anchors
#[derive(Component, Clone)]
pub struct AnchorAssets {
    pub mesh: Handle<Mesh>,
    pub tf_normal_pre: Transform,
    pub tf_normal_post: Transform,
    pub tf_hovered_pre: Transform,
    pub tf_hovered_post: Transform,
    pub mat_normal: Handle<StandardMaterial>,
    pub mat_hovered: Handle<StandardMaterial>,
}

/// Marker for AST node mesh entities (so we can despawn them on rebuild).
#[derive(Component)]
struct AstNodeEntity {
    node_id: ast::node::Id,
}

/// Flag resource that signals the scene needs rebuilding.
#[derive(Resource, Default)]
struct NeedsRebuild(bool);

/// Marker for any spawned scene entity (cleaned on rebuild).
#[derive(Component)]
struct AstSceneEntity;

//Buttons
#[derive(Component)]
struct DeleteNodeButton;
#[derive(Component)]
struct ResetCameraButton;
#[derive(Component, Clone)]
enum EAstActionButton {
    AddIntIntroductionButton,
    AddBoolIntroductionButton,
    AddIntEliminationButton,
    AddFunctionCallButton,
    AddMatchButton,
}

/// Stores which node is hovered / selected.
#[derive(Resource, Default)]
struct PickState {
    hovered: Option<ast::node::Id>,  // node_id
    selected: Option<ast::node::Id>, // node_id
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
        FogSettings {
            color: Color::srgba(0.02, 0.02, 0.36, 1.0), // very dark blue-ish
            falloff: FogFalloff::Exponential { density: 0.03 },
            ..default()
        },
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
    let mut node_entites = std::collections::HashMap::<ast::node::Id, Entity>::new();
    let mut anchor_entities = std::collections::HashMap::<ast::AnchorId, Entity>::new();
    for (node_id, layout_node) in &state.layout_ast.layout_nodes {
        let node = state
            .layout_ast
            .ast
            .nodes
            .get(&layout_node.node_id)
            .unwrap();
        let render_node = render::layoutnode_to_rendernode(
            &layout_node,
            &state.layout_ast.ast,
            &state.function_declarations,
        );
        let node_entity = commands
            .spawn((
                PbrBundle {
                    mesh: meshes.add(render_node.node.mesh),
                    material: materials.add(render_node.node.material),
                    transform: render_node.node.transform,
                    ..default()
                },
                AstNodeEntity {
                    node_id: node_id.clone(),
                },
                AstSceneEntity,
            ))
            .id();
        render_node
            .anchors
            .into_iter()
            .for_each(|(anchor_id, render_anchor)| {
                let layout_anchor = state.layout_ast.layout_anchor(anchor_id.clone());
                anchor_entities.insert(
                    anchor_id.clone(),
                    commands
                        .spawn((
                            PbrBundle {
                                mesh: meshes.add(render_anchor.normal.mesh.clone()),
                                material: materials.add(render_anchor.normal.material.clone()),
                                transform: render_node.node.transform,
                                ..default()
                            },
                            match layout_anchor.anchor {
                                ast::EAnchor::Input { .. } => EAnchor::Input {
                                    id: anchor_id,
                                    render_objects: render_anchor,
                                },
                                ast::EAnchor::Output => EAnchor::Output {
                                    id: anchor_id,
                                    render_objects: render_anchor,
                                },
                            },
                        ))
                        .id(),
                );
            });

        node_entites.insert(node_id.clone(), node_entity.clone());

        render_node.labels.into_iter().for_each(|l| {
            spawn_world_label(&mut commands, l, AstSceneEntity);
        });
    }

    for e in state.layout_ast.edges() {
        commands.spawn(Edge {
            from_anchor: *anchor_entities.get(&e.from_anchor.anchor_id).unwrap(),
            to_anchor: *anchor_entities.get(&e.to_anchor.anchor_id).unwrap(),
        });
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
    let y_offset = 12.0;
    spawn_ui_button(
        &mut commands,
        "Reset Camera",
        ResetCameraButton,
        Vec2::new(12.0, y_offset),
    );
    let y_offset = y_offset + 36.0;
    // Outer container (clickable background)
    let initial = "";
    commands
        .spawn((
            ButtonBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    top: Val::Px(y_offset),
                    left: Val::Px(12.0), // next to reset button
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
    let y_offset = y_offset + 36.0;
    spawn_ui_button(
        &mut commands,
        "Delete Node",
        DeleteNodeButton,
        Vec2::new(12.0, y_offset),
    );
    let y_offset = y_offset + 36.0;
    spawn_ui_button(
        &mut commands,
        "Add Int Introduction",
        EAstActionButton::AddIntIntroductionButton,
        Vec2::new(12.0, y_offset),
    );
    let y_offset = y_offset + 36.0;
    spawn_ui_button(
        &mut commands,
        "Add Bool Introduction",
        EAstActionButton::AddBoolIntroductionButton,
        Vec2::new(12.0, y_offset),
    );
    let y_offset = y_offset + 36.0;
    spawn_ui_button(
        &mut commands,
        "Add Function Call",
        EAstActionButton::AddFunctionCallButton,
        Vec2::new(12.0, y_offset),
    );
    let y_offset = y_offset + 36.0;
    spawn_ui_button(
        &mut commands,
        "Add Match",
        EAstActionButton::AddMatchButton,
        Vec2::new(12.0, y_offset),
    );
    let y_offset = y_offset + 36.0;
    spawn_ui_button(
        &mut commands,
        "Add Int Elimination",
        EAstActionButton::AddIntEliminationButton,
        Vec2::new(12.0, y_offset),
    );
}

fn spawn_ui_button<C: Component>(commands: &mut Commands, label: &str, component: C, pos: Vec2) {
    commands
        .spawn((
            ButtonBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    top: Val::Px(pos.y),
                    left: Val::Px(pos.x),
                    padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
                    ..default()
                },
                background_color: Color::srgba(0.16, 0.16, 0.22, 0.9).into(),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            component,
        ))
        .with_children(|parent| {
            parent.spawn(TextBundle::from_section(
                label,
                TextStyle {
                    font_size: 14.0,
                    color: Color::srgb(0.6, 0.6, 0.7),
                    ..default()
                },
            ));
        });
}

fn handle_reset_camera_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (With<Interaction>, With<ResetCameraButton>),
    >,
    mut text_q: Query<&mut Text>,
    mut orbit: ResMut<camera::OrbitCamera>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut text = text_q.get_mut(children[0]).unwrap();

        match *interaction {
            Interaction::Pressed => {
                orbit.theta = 0.6;
                orbit.phi = 1.0;
                orbit.target = Vec3::ZERO;
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

fn handle_delete_node_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (With<Interaction>, With<DeleteNodeButton>),
    >,
    mut text_q: Query<&mut Text>,
    mut state: ResMut<AstState>,
    mut orbit: ResMut<camera::OrbitCamera>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut pick: ResMut<PickState>,
    mut commands: Commands,
    scene_entities: Query<Entity, With<AstSceneEntity>>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut text = text_q.get_mut(children[0]).unwrap();

        match *interaction {
            Interaction::Pressed => {
                if let Some(selected_node_id) = &pick.selected {
                    state.layout_ast = state.layout_ast.minus_node(&selected_node_id.clone());
                    rebuild.0 = true;
                }
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

fn handle_add_node_button(
    mut interaction_q: Query<
        (
            &Interaction,
            &mut BackgroundColor,
            &Children,
            &EAstActionButton,
        ),
        With<Interaction>,
    >,
    mut text_q: Query<&mut Text>,
    mut state: ResMut<AstState>,
    mut current_input_string: ResMut<CurrentInputString>,
    mut orbit: ResMut<camera::OrbitCamera>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut pick: ResMut<PickState>,
    mut commands: Commands,
    scene_entities: Query<Entity, With<AstSceneEntity>>,
) {
    for (interaction, mut bg, children, action) in interaction_q.iter_mut() {
        let mut text = text_q.get_mut(children[0]).unwrap();

        match *interaction {
            Interaction::Pressed => {
                if let Some(selected_node_id) = &pick.selected {
                    let selected_pos = state
                        .layout_ast
                        .layout_nodes
                        .get(selected_node_id)
                        .unwrap()
                        .pos;
                    let new_pos = Vec3::new(selected_pos.x + 1.0, selected_pos.y, selected_pos.z);
                    state.layout_ast = match action {
                        EAstActionButton::AddIntIntroductionButton => {
                            state.layout_ast.plus_type_introduction(
                                ast::node::EType::Int {
                                    value: Some(current_input_string.0.clone()),
                                },
                                new_pos,
                            )
                        }
                        EAstActionButton::AddBoolIntroductionButton => {
                            state.layout_ast.plus_type_introduction(
                                ast::node::EType::Bool {
                                    value: Some(current_input_string.0.clone()),
                                },
                                new_pos,
                            )
                        }
                        EAstActionButton::AddFunctionCallButton => {
                            state.layout_ast.plus_function_call(
                                state
                                    .function_declarations
                                    .iter()
                                    .find(|(_, d)| current_input_string.0 == d.name)
                                    .map(|(id, decl)| (id.clone(), decl))
                                    .unwrap(),
                                new_pos,
                            )
                        }
                        EAstActionButton::AddIntEliminationButton => {
                            state.layout_ast.plus_type_elimination(
                                ast::node::EType::Int {
                                    value: if current_input_string.0.clone() == "".to_string() {
                                        None
                                    } else {
                                        Some(current_input_string.0.clone())
                                    },
                                },
                                new_pos,
                            )
                        }
                        EAstActionButton::AddMatchButton => state.layout_ast.plus_match(new_pos),
                    };
                    rebuild.0 = true;
                }
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

/* curved edges
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
*/

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

fn clear_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<AstState>,
    rebuild: ResMut<NeedsRebuild>,
    query_ast_entities: Query<Entity, With<AstSceneEntity>>,
) {
    if rebuild.0 {
        for entity in query_ast_entities.iter() {
            commands.entity(entity).despawn_recursive();
        }

        let mesh_ids: Vec<_> = meshes.ids().collect();
        for id in mesh_ids {
            meshes.remove(id);
        }

        let mat_ids: Vec<_> = materials.ids().collect();
        for id in mat_ids {
            materials.remove(id);
        }
    }
}
fn rebuild_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<AstState>,
    mut rebuild: ResMut<NeedsRebuild>,
    query_ast_entities: Query<Entity, With<AstSceneEntity>>,
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
    render_label: render::RenderLabel,
    marker: impl Bundle,
) -> Entity {
    commands
        .spawn((
            TextBundle {
                text: Text::from_section(
                    render_label.text,
                    TextStyle {
                        font_size: render_label.font_size,
                        color: render_label.color,
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
            WorldLabel {
                world_pos: render_label.world_pos,
                offset: render_label.offset,
            },
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
    /*
    mut input_q: Query<(&mut TextInput, &Children), With<TextInputBox>>,
    mut current_input_string: ResMut<CurrentInputString>,
    */
    state: Res<AstState>,
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
    let mut closest: Option<(ast::node::Id, f32)> = None;

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
        /*
        println!("selected!");
        if let Some(selected_id) = &pick.selected {
        println!("selected!2");

                match state.layout_ast.ast.nodes.get(&selected_id) {
                    Some(ast::EAstNode::BoolLiteral { value, .. })
                    | Some(ast::EAstNode::NumLiteral { value, .. }) => {
                        current_input_string.0 = value.to_string();
                    }
                    _ => (),
                };
            for (mut input, _) in input_q.iter_mut() {
        println!("selected!3");
                /**
                if !input.focused {
                    continue;
                }
                **/

                match state.layout_ast.ast.nodes.get(&selected_id) {
                    Some(ast::EAstNode::BoolLiteral { value, .. })
                    | Some(ast::EAstNode::NumLiteral { value, .. }) => {
                        input.value = value.to_string();
                    }
                    _ => (),
                };
            }
        }
        */
    }
}

fn highlight_hovered(
    pick: Res<PickState>,
    node_q: Query<(&AstNodeEntity, &Handle<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (node_ent, mat_handle) in node_q.iter() {
        let Some(mat) = materials.get_mut(mat_handle) else {
            continue;
        };

        let base = render::emissive_color(mat.base_color);
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
                "{} : {}",
                node.label(&state.function_declarations),
                match eval::eval_type(
                    &node,
                    &state.layout_ast.ast,
                    &state.function_declarations,
                    vec![]
                ) {
                    Ok(r#type) => r#type.to_string(),
                    Err(message) => format!("error: {}", message),
                }
            );
            text.sections[0].style.color = Color::WHITE
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
    mut current_input_string: ResMut<CurrentInputString>,
    //mut state: ResMut<AstState>,
    mut orbit: ResMut<camera::OrbitCamera>,
    //    mut rebuild: ResMut<NeedsRebuild>,
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

        current_input_string.0 = input.value.to_string();

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
            //orbit.auto_rotate = true;
            //rebuild.0 = true;
            //orbit.theta = 0.6;
            //orbit.phi = 1.0;
        }
    }
}

fn handle_arrow_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<AstState>,
    mut pick: ResMut<PickState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    if keys.just_pressed(KeyCode::ArrowUp) {
        if shift {
            if let Some(selected_node_id) = &pick.selected {
                state.layout_ast = state
                    .layout_ast
                    .move_node_delta(selected_node_id.clone(), Vec3::new(0.0, 1.0, 0.0));
                rebuild.0 = true;
            }
        } else {
            if let Some(selected_node_id) = &pick.selected {
                state.layout_ast = state
                    .layout_ast
                    .move_node_delta(selected_node_id.clone(), Vec3::new(-1.0, 0.0, 0.0));
                rebuild.0 = true;
            }
        }
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        if shift {
            if let Some(selected_node_id) = &pick.selected {
                state.layout_ast = state
                    .layout_ast
                    .move_node_delta(selected_node_id.clone(), Vec3::new(0.0, -1.0, 0.0));
                rebuild.0 = true;
            }
        } else {
            if let Some(selected_node_id) = &pick.selected {
                state.layout_ast = state
                    .layout_ast
                    .move_node_delta(selected_node_id.clone(), Vec3::new(1.0, 0.0, 0.0));
                rebuild.0 = true;
            }
        }
    }
    if keys.just_pressed(KeyCode::ArrowLeft) {
        if shift {
            // Shift + Left
        } else {
            if let Some(selected_node_id) = &pick.selected {
                state.layout_ast = state
                    .layout_ast
                    .move_node_delta(selected_node_id.clone(), Vec3::new(0.0, 0.0, 1.0));
                rebuild.0 = true;
            }
        }
    }
    if keys.just_pressed(KeyCode::ArrowRight) {
        if shift {
            // Shift + Right
        } else {
            if let Some(selected_node_id) = &pick.selected {
                state.layout_ast = state
                    .layout_ast
                    .move_node_delta(selected_node_id.clone(), Vec3::new(0.0, 0.0, -1.0));
                rebuild.0 = true;
            }
        }
    }
}

fn anchor_hover_system(
    mut commands: Commands,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    anchors: Query<(Entity, &GlobalTransform), With<EAnchor>>,
    existing_hovers: Query<Entity, With<AnchorHovered>>,
) {
    // Alle vorherigen Hovers entfernen
    for e in &existing_hovers {
        commands.entity(e).remove::<AnchorHovered>();
    }

    let Ok(window) = windows.get_single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_tf)) = camera_q.get_single() else {
        return;
    };

    let mut closest: Option<(Entity, f32)> = None;

    for (entity, global_tf) in &anchors {
        let Some(screen_pos) = camera.world_to_viewport(cam_tf, global_tf.translation()) else {
            continue;
        };

        let dist = cursor.distance(screen_pos);
        if dist < 25.0 {
            if closest.map_or(true, |(_, d)| dist < d) {
                closest = Some((entity, dist));
            }
        }
    }

    if let Some((entity, _)) = closest {
        commands.entity(entity).insert(AnchorHovered);
    }
}

fn anchor_hover_visual_system(
    mut anchors: Query<(
        &EAnchor,
        &mut Transform,
        &mut Handle<StandardMaterial>,
        Option<&AnchorHovered>,
        &AnchorAssets,
    )>,
) {
    for (anchor, mut tf, mut mat, hovered, assets) in &mut anchors {
        let (is_hovered, target_mat) = if hovered.is_some() {
            (true, &assets.mat_hovered)
        } else {
            (false, &assets.mat_normal)
        };

        // Smooth scale
        //let s = tf.scale.x + (target_scale - tf.scale.x) * 0.18;
        //tf.scale = Vec3::splat(s);
        *tf = match anchor {
            EAnchor::Input { render_objects, .. } | EAnchor::Output { render_objects, .. } => {
                if hovered.is_some() {
                    render_objects.hovered.transform
                } else {
                    render_objects.normal.transform
                }
            }
        };

        // Material swap (Handle-Vergleich ist billig)
        if *mat != *target_mat {
            *mat = target_mat.clone();
        }
    }
}

fn draw_edges_gizmos(edges: Query<&Edge>, transforms: Query<&GlobalTransform>, mut gizmos: Gizmos) {
    for edge in &edges {
        let (Ok(from), Ok(to)) = (
            transforms.get(edge.from_anchor),
            transforms.get(edge.to_anchor),
        ) else {
            continue;
        };
        gizmos.line(from.translation(), to.translation(), Color::WHITE);
    }
}

fn draw_drag_preview(drag: Res<DragState>, mut gizmos: Gizmos) {
    let Some(ref info) = drag.active else { return };
    let color = if info.target_anchor.is_some() {
        Color::rgb(0.3, 1.0, 0.4) // grün = eingeschnappt
    } else {
        Color::rgb(1.0, 0.9, 0.3) // gelb = dragging
    };
    gizmos.line(info.source_pos, info.current_end, color);
}

fn drag_start_system(
    mouse: Res<ButtonInput<MouseButton>>,
    hovered: Query<(Entity, &GlobalTransform, &EAnchor), (With<AnchorHovered>)>,
    mut drag: ResMut<DragState>,
) {
    if mouse.just_pressed(MouseButton::Left) {
        if let Ok((entity, tf, anchor)) = hovered.get_single() {
            let pos = tf.translation();
            drag.active = Some(DragInfo {
                source_anchor: entity,
                source_anchor_id: anchor.id(),
                source_pos: pos,
                current_end: pos,
                target_anchor: None,
                target_anchor_id: None,
            });
        }
    }
}

fn drag_update_system(
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    hovered: Query<(Entity, &GlobalTransform, &EAnchor), (With<AnchorHovered>)>,
    mut drag: ResMut<DragState>,
) {
    let Some(ref mut info) = drag.active else {
        return;
    };

    let Ok(window) = windows.get_single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_tf)) = camera_q.get_single() else {
        return;
    };

    // Ray durch Cursor
    let Some(ray) = camera.viewport_to_world(cam_tf, cursor) else {
        return;
    };

    // Schnitt mit Ebene durch source_pos, senkrecht zur Kamera
    let normal: Vec3 = -*cam_tf.forward();
    let denom = ray.direction.dot(normal);
    if denom.abs() > 1e-6 {
        let t = (info.source_pos - ray.origin).dot(normal) / denom;
        if t > 0.0 {
            info.current_end = ray.origin + *ray.direction * t;
        }
    }

    // Snap zu hovering target
    info.target_anchor = None;
    if let Ok((entity, tf, anchor)) = hovered.get_single() {
        if entity != info.source_anchor {
            info.target_anchor = Some(entity);
            info.target_anchor_id = Some(anchor.id());
            info.current_end = tf.translation();
        }
    }
}

fn drag_end_system(
    mouse: Res<ButtonInput<MouseButton>>,
    mut drag: ResMut<DragState>,
    mut commands: Commands,
    mut rebuild: ResMut<NeedsRebuild>,
    mut state: ResMut<AstState>,
) {
    if mouse.just_released(MouseButton::Left) {
        if let Some(info) = drag.active.take() {
            if let Some(target_id) = info.target_anchor_id {
                state.layout_ast = state.layout_ast.plus_edge(info.source_anchor_id, target_id);
                rebuild.0 = true;
            }
            // Kein target → Drag wird einfach verworfen
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
        .add_plugins(grid::GridPlugin)
        .init_resource::<AstState>()
        .init_resource::<CurrentInputString>()
        .init_resource::<NeedsRebuild>()
        .init_resource::<PickState>()
        .init_resource::<DragState>()
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
        .add_systems(
            Update,
            (
                (
                    (
                        anchor_hover_visual_system,
                        draw_edges_gizmos,
                        draw_drag_preview,
                    )
                        .chain(),
                    //draw_edges,
                    animate_nodes,
                    (
                        handle_delete_node_button,
                        handle_reset_camera_button,
                        handle_add_node_button,
                        pick_nodes,
                    )
                        .chain(),
                    highlight_hovered,
                    update_selection_display,
                    update_cursor,
                    text_input_focus,
                    text_input_keyboard,
                    handle_arrow_keys,
                ),
                (
                    anchor_hover_system,
                    drag_start_system,
                    drag_update_system,
                    drag_end_system,
                    clear_scene,
                    apply_deferred,
                    rebuild_scene,
                )
                    .chain(),
            )
                .chain(),
        )
        .add_systems(Update, update_world_labels)
        .run();
}
