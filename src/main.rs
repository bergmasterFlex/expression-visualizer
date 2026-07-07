mod ast;
mod camera;
mod colors;
mod eval;
mod grid;
mod layout;
mod mesh;
mod render;

use std::{collections::hash_map, f32::consts::PI};

use ast::FunctionParameterDeclaration;
use bevy::core_pipeline::oit::OrderIndependentTransparencySettings;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::{input::keyboard::KeyboardInput, math::VectorSpace, prelude::*};

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
            layout_ast: layout::LayoutAst::empty(),
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
                (
                    ast::FunctionDeclarationId(2),
                    ast::FunctionDeclaration {
                        name: "charAt".to_string(),
                        inputs: vec![
                            FunctionParameterDeclaration {
                                name: "str".to_string(),
                                r#type: eval::EType::String(None),
                            },
                            FunctionParameterDeclaration {
                                name: "i".to_string(),
                                r#type: eval::EType::Int(None),
                            },
                        ],
                        output_type: eval::EType::SumType(vec![
                            eval::EType::Char(None),
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
    pub color: Color,
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
struct HamburgerButton;
#[derive(Component, Clone)]
enum EAstActionButton {
    AddConstDeclButton,
    AddVarDeclButton,
    AddTypeCastButton,
    AddFunctionCallButton,
    AddMatchFrontButton,
}

#[derive(Resource, Default, PartialEq, Eq, Clone, Copy)]
enum UiMode {
    #[default]
    Playground,
    Examples,
}

#[derive(Resource)]
struct StartMenu {
    showing: bool,
    has_cancel: bool,
}

impl Default for StartMenu {
    fn default() -> Self {
        Self {
            showing: true,
            has_cancel: false,
        }
    }
}

#[derive(Component)]
struct StartMenuEntity;
#[derive(Component)]
struct StartMenuNewButton;
#[derive(Component)]
struct StartMenuLoadExampleButton;
#[derive(Component)]
struct StartMenuCancelButton;
#[derive(Component)]
struct StartMenuControlsButton;

/// Marker for UI entities that should be hidden while the start menu is open.
#[derive(Component)]
struct HideDuringStartMenu;

#[derive(Component)]
struct PlaygroundOnly;
#[derive(Component)]
struct ExamplesOnly;

#[derive(Component, Clone)]
enum ExampleButton {
    Sink,
    FuncCall,
    Match,
    Pattern,
    TypeCast,
    VarDecl,
    ConstDecl,
    TypeClass,
}

/// Stores the currently selected grid position and hover state.
///
/// Selection is a grid position (always some), not a node. A node is
/// considered selected iff its layout position rounds to `selected_pos`.
#[derive(Resource)]
struct PickState {
    /// Currently selected grid position (layout coordinates, always set).
    selected_pos: IVec3,
    /// Node under the cursor (ray-sphere hit), if any.
    hovered_node: Option<ast::node::Id>,
    /// Grid crossing under the cursor (ray hit on y=0 plane), if within
    /// the visible grid extent and no node was hit.
    hovered_pos: Option<IVec3>,
    /// Cursor position at the last left-mouse press. Used to distinguish
    /// click vs drag — a release within `CLICK_MOVE_THRESHOLD` of this
    /// counts as a click and updates the selection; further movement is
    /// treated as a drag and leaves selection untouched.
    press_cursor: Option<Vec2>,
    /// True if the last left-mouse press landed on a `Button` UI element.
    /// The release-time grid selection is suppressed in that case so
    /// clicking a dropdown/text-input/checkbox doesn't move the selection
    /// to whatever grid cell the ray passes through behind the panel.
    press_over_ui: bool,
}

impl Default for PickState {
    fn default() -> Self {
        Self {
            selected_pos: IVec3::ZERO,
            hovered_node: None,
            hovered_pos: None,
            press_cursor: None,
            press_over_ui: false,
        }
    }
}

/// UI text showing the selected node's info.
#[derive(Component)]
struct SelectionDisplay;

/// Marker for the FPS counter text in the top-right corner.
#[derive(Component)]
struct FpsDisplay;

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

// ── Stepwise evaluation ─────────────────────────────────────

#[derive(Clone, Default)]
enum EvalPhase {
    #[default]
    Idle,
    ErrorModal(String),
    ControlsModal,
    VarDeclPrompt {
        /// Stable node_id order; values mirror what the user has typed so far.
        inputs: Vec<(ast::node::Id, String)>,
    },
    Running {
        steps: Vec<std::collections::HashMap<ast::node::Id, String>>,
        current: usize,
    },
}

#[derive(Resource)]
struct EvalState {
    phase: EvalPhase,
    rng: rand::rngs::SmallRng,
}

impl Default for EvalState {
    fn default() -> Self {
        use rand::SeedableRng;
        Self {
            phase: EvalPhase::Idle,
            rng: rand::rngs::SmallRng::from_os_rng(),
        }
    }
}

fn is_evaluating(eval: &EvalState) -> bool {
    !matches!(eval.phase, EvalPhase::Idle)
}

fn modal_is_open(eval: &EvalState) -> bool {
    matches!(
        eval.phase,
        EvalPhase::ErrorModal(_) | EvalPhase::ControlsModal | EvalPhase::VarDeclPrompt { .. }
    )
}

#[derive(Component)]
struct EvaluateButton;

/// Tags any entity that belongs to the currently-displayed modal so we can
/// nuke the whole subtree on phase transition.
#[derive(Component)]
struct ModalEntity;

#[derive(Component)]
struct ModalOkButton;
#[derive(Component)]
struct ModalCancelButton;
#[derive(Component)]
struct ModalEvaluateButton;
#[derive(Component)]
struct ControlsModalOkButton;

/// Marker on a TextInputBox inside the VarDecl modal so we can collect
/// typed values per VarDecl when the user confirms.
#[derive(Component)]
struct ModalVarDeclInput {
    node_id: ast::node::Id,
}

/// Tags entities that make up the Prev/Next/Exit bottom bar.
#[derive(Component)]
struct EvalStepBarEntity;

#[derive(Component)]
struct PrevStepButton;
#[derive(Component)]
struct NextStepButton;
#[derive(Component)]
struct ExitEvaluationButton;

/// World-space text node showing a node's current evaluated value.
#[derive(Component)]
struct ValueLabel {
    node_id: ast::node::Id,
}

// ── Node editor panel ───────────────────────────────────────

#[derive(Component)]
struct NodeEditorPanel;

/// Tag on every descendant of the editor panel that is rebuilt when the
/// selection or dropdown state changes.
#[derive(Component)]
struct NodeEditorEntity;

#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeEditorField {
    VarDeclName,
    Value,
}

#[derive(Component)]
struct NodeEditorTextInput {
    node_id: ast::node::Id,
    field: NodeEditorField,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DropdownKind {
    Type,
    Function,
    BoolValue,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TypeChoice {
    String,
    Char,
    Bool,
    Int,
    Float,
    Undefined,
}

const TYPE_CHOICES: [TypeChoice; 6] = [
    TypeChoice::String,
    TypeChoice::Char,
    TypeChoice::Bool,
    TypeChoice::Int,
    TypeChoice::Float,
    TypeChoice::Undefined,
];

#[derive(Clone, PartialEq, Eq)]
enum DropdownChoice {
    Type(TypeChoice),
    Function(ast::FunctionDeclarationId),
    BoolValue(bool),
}

#[derive(Component)]
struct Dropdown {
    node_id: ast::node::Id,
    kind: DropdownKind,
}

#[derive(Component)]
struct DropdownOption {
    node_id: ast::node::Id,
    kind: DropdownKind,
    choice: DropdownChoice,
}

#[derive(Component)]
struct ValueEnableCheckbox {
    node_id: ast::node::Id,
}

/// At most one dropdown is open at a time; `open` identifies which one by
/// `(node_id, kind)` — stable across panel rebuilds.
#[derive(Resource, Default)]
struct DropdownState {
    open: Option<(ast::node::Id, DropdownKind)>,
}

#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum NodeVariantKind {
    #[default]
    None,
    TypeIntroduction,
    TypeElimination,
    VarDecl,
    FunctionCall,
    Other,
}

fn variant_kind(node: Option<&ast::node::ENode>) -> NodeVariantKind {
    match node {
        None => NodeVariantKind::None,
        Some(ast::node::ENode::TypeIntroduction { .. }) => NodeVariantKind::TypeIntroduction,
        Some(ast::node::ENode::TypeElimination { .. }) => NodeVariantKind::TypeElimination,
        Some(ast::node::ENode::VarDecl { .. }) => NodeVariantKind::VarDecl,
        Some(ast::node::ENode::FunctionCall { .. }) => NodeVariantKind::FunctionCall,
        Some(_) => NodeVariantKind::Other,
    }
}

fn type_choice_of(t: &ast::node::EType) -> Option<TypeChoice> {
    match t {
        ast::node::EType::Bool { .. } => Some(TypeChoice::Bool),
        ast::node::EType::Int { .. } => Some(TypeChoice::Int),
        ast::node::EType::Float { .. } => Some(TypeChoice::Float),
        ast::node::EType::String { .. } => Some(TypeChoice::String),
        ast::node::EType::Char { .. } => Some(TypeChoice::Char),
        ast::node::EType::Undefined => Some(TypeChoice::Undefined),
        ast::node::EType::Any | ast::node::EType::Exception { .. } => None,
    }
}

fn type_choice_label(t: TypeChoice) -> &'static str {
    match t {
        TypeChoice::String => "string",
        TypeChoice::Char => "char",
        TypeChoice::Bool => "bool",
        TypeChoice::Int => "int",
        TypeChoice::Float => "float",
        TypeChoice::Undefined => "undefined",
    }
}

fn value_of_etype(t: &ast::node::EType) -> Option<String> {
    match t {
        ast::node::EType::Bool { value }
        | ast::node::EType::Int { value }
        | ast::node::EType::Float { value }
        | ast::node::EType::String { value }
        | ast::node::EType::Char { value } => value.clone(),
        _ => None,
    }
}

fn make_etype(choice: TypeChoice, value: Option<String>) -> ast::node::EType {
    match choice {
        TypeChoice::Bool => ast::node::EType::Bool { value },
        TypeChoice::Int => ast::node::EType::Int { value },
        TypeChoice::Float => ast::node::EType::Float { value },
        TypeChoice::String => ast::node::EType::String { value },
        TypeChoice::Char => ast::node::EType::Char { value },
        TypeChoice::Undefined => ast::node::EType::Undefined,
    }
}

// ── Colors ──────────────────────────────────────────────────

// ── Fonts ───────────────────────────────────────────────────

#[derive(Resource, Clone)]
struct UiFont(Handle<Font>);

fn load_ui_font(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(UiFont(asset_server.load("fonts/JetBrainsMono-Regular.ttf")));
}

fn text_font(font: &Handle<Font>, size: f32) -> TextFont {
    TextFont {
        font: font.clone(),
        font_size: size,
        ..default()
    }
}

// ── Systems ─────────────────────────────────────────────────

/// Initial scene setup: camera, lights, ambient.
fn setup_scene(mut commands: Commands) {
    // Camera with order-independent transparency for correct intersection
    // of the two walls and the grid. OIT requires MSAA off.
    // The grid shader calls `oit_draw()` under #ifdef OIT_ENABLED to
    // participate in the OIT layer buffer (see assets/shaders/grid.wgsl).
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.031, 0.031, 0.102)),
            ..default()
        },
        Transform::from_xyz(0.0, 5.0, 12.0).looking_at(Vec3::ZERO, Vec3::Y),
        OrderIndependentTransparencySettings::default(),
        Msaa::Off,
        camera::OrbitCameraTag,
        DistanceFog {
            color: Color::srgba(0.02, 0.02, 0.36, 1.0),
            falloff: FogFalloff::Exponential { density: 0.03 },
            ..default()
        },
        AmbientLight {
            color: Color::srgb(0.25, 0.25, 0.38),
            brightness: 200.0,
            ..default()
        },
    ));

    // Directional light
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(5.0, 10.0, 7.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Two colored point lights
    commands.spawn((
        PointLight {
            color: Color::srgb(0.133, 0.827, 0.933),
            intensity: 50_000.0,
            range: 30.0,
            ..default()
        },
        Transform::from_xyz(0.0, 5.0, 8.0),
    ));
    commands.spawn((
        PointLight {
            color: Color::srgb(1.0, 0.42, 0.42),
            intensity: 30_000.0,
            range: 30.0,
            ..default()
        },
        Transform::from_xyz(0.0, 5.0, -8.0),
    ));
}

/// Spawn the transparent grey front wall at z = +12,
/// spanning x ∈ [-30, 30] and y ∈ [-3, 3].
fn spawn_walls(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let wall_mesh = meshes.add(Cuboid::new(40.0, 6.0, 0.05));
    let wall_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.5, 0.5, 0.5, 0.5),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..default()
    });

    commands.spawn((
        Mesh3d(wall_mesh),
        MeshMaterial3d(wall_material),
        Transform::from_xyz(0.0, 0.0, 12.0),
    ));
}

/// Spawn the AST node meshes.
fn spawn_ast_nodes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut materials_grid: ResMut<Assets<grid::GridMaterial>>,
    state: Res<AstState>,
    ui_font: Res<UiFont>,
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
        if let ast::node::ENode::MatchGrid { width, depth } = node {
            let world_pos = layout_node.pos * Vec3::new(3.0, 3.0, 3.0);
            let size_x = *width as f32 * 3.0;
            let size_z = *depth as f32 * 3.0;
            let tf = Transform::from_translation(
                world_pos + Vec3::new(size_x / 2.0 - 1.5, 0.0, -size_z / 2.0),
            );
            let entity = commands
                .spawn((
                    Mesh3d(meshes.add(Plane3d::default().mesh().size(size_x, size_z).build())),
                    MeshMaterial3d(materials_grid.add(grid::GridMaterial {
                        plane_color: LinearRgba::new(0.07, 0.07, 0.1, 0.55),
                        line_color: LinearRgba::new(0.2, 0.2, 0.5, 0.4),
                        spacing: 3.0,
                        fade_start: 15.0,
                        fade_end: 100.0,
                        line_thickness: 1.5,
                        hover_pos: Vec2::ZERO,
                        hover_active: 0.0,
                        _pad: 0.0,
                    })),
                    tf,
                    AstNodeEntity {
                        node_id: node_id.clone(),
                    },
                    AstSceneEntity,
                ))
                .id();
            node_entites.insert(node_id.clone(), entity);
            continue;
        }
        let render_node = render::layoutnode_to_rendernode(
            &layout_node,
            &state.layout_ast.ast,
            &state.function_declarations,
        );
        let node_entity = commands
            .spawn((
                Mesh3d(meshes.add(render_node.node.mesh)),
                MeshMaterial3d(materials.add(render_node.node.material)),
                render_node.node.transform,
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
                            Mesh3d(meshes.add(render_anchor.normal.mesh.clone())),
                            MeshMaterial3d(materials.add(render_anchor.normal.material.clone())),
                            render_anchor.normal.transform,
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
                            AstSceneEntity,
                        ))
                        .id(),
                );
            });

        node_entites.insert(node_id.clone(), node_entity.clone());

        render_node.labels.into_iter().for_each(|l| {
            spawn_world_label(&mut commands, &ui_font.0, l, AstSceneEntity);
        });
    }

    for e in state.layout_ast.edges() {
        commands.spawn((
            Edge {
                from_anchor: *anchor_entities.get(&e.from_anchor.anchor_id).unwrap(),
                to_anchor: *anchor_entities.get(&e.to_anchor.anchor_id).unwrap(),
                color: e.color,
            },
            AstSceneEntity,
        ));
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

fn spawn_ui(mut commands: Commands, ui_font: Res<UiFont>) {
    // Hamburger menu button (top-left) — opens the menu modal.
    spawn_hamburger_button(&mut commands, Vec2::new(12.0, 12.0));

    // Playground-mode controls.
    let mut y_offset = 60.0;
    spawn_ui_button(
        &mut commands,
        &ui_font.0,
        "Delete Node",
        (DeleteNodeButton, PlaygroundOnly),
        Vec2::new(12.0, y_offset),
        Display::Flex,
    );
    for (label, action) in [
        ("Add ConstDecl", EAstActionButton::AddConstDeclButton),
        ("Add VarDecl", EAstActionButton::AddVarDeclButton),
        ("Add FunctionCall", EAstActionButton::AddFunctionCallButton),
        ("Add Match", EAstActionButton::AddMatchFrontButton),
        ("Add TypeCast", EAstActionButton::AddTypeCastButton),
    ] {
        y_offset += 36.0;
        spawn_ui_button(
            &mut commands,
            &ui_font.0,
            label,
            (action, PlaygroundOnly),
            Vec2::new(12.0, y_offset),
            Display::Flex,
        );
    }

    // Examples-mode buttons (initially hidden; update_mode_visibility keeps them in sync).
    let mut y_offset = 60.0;
    for (label, kind) in [
        ("Sink", ExampleButton::Sink),
        ("FuncCall", ExampleButton::FuncCall),
        ("Match", ExampleButton::Match),
        ("Pattern", ExampleButton::Pattern),
        ("TypeCast", ExampleButton::TypeCast),
        ("VarDecl", ExampleButton::VarDecl),
        ("ConstDecl", ExampleButton::ConstDecl),
        ("TypeClass", ExampleButton::TypeClass),
    ] {
        spawn_ui_button(
            &mut commands,
            &ui_font.0,
            label,
            (kind, ExamplesOnly),
            Vec2::new(12.0, y_offset),
            Display::None,
        );
        y_offset += 36.0;
    }

    // "Evaluate" button at the bottom-right corner — visible in both
    // Playground and Examples modes.
    spawn_corner_button(
        &mut commands,
        &ui_font.0,
        "Evaluate",
        EvaluateButton,
        Val::Px(12.0),
        Val::Px(12.0),
    );
}

fn spawn_corner_button<C: Bundle>(
    commands: &mut Commands,
    font: &Handle<Font>,
    label: &str,
    component: C,
    right: Val,
    bottom: Val,
) {
    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Auto,
                left: Val::Auto,
                right,
                bottom,
                padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.16, 0.16, 0.22, 0.9)),
            component,
            HideDuringStartMenu,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(label),
                text_font(font, 14.0),
                TextColor(Color::srgb(0.6, 0.6, 0.7)),
            ));
        });
}

fn spawn_ui_button<C: Bundle>(
    commands: &mut Commands,
    font: &Handle<Font>,
    label: &str,
    component: C,
    pos: Vec2,
    initial_display: Display,
) {
    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(pos.y),
                left: Val::Px(pos.x),
                padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                display: initial_display,
                ..default()
            },
            BackgroundColor(Color::srgba(0.16, 0.16, 0.22, 0.9)),
            component,
            HideDuringStartMenu,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(label),
                text_font(font, 14.0),
                TextColor(Color::srgb(0.6, 0.6, 0.7)),
            ));
        });
}

fn spawn_hamburger_button(commands: &mut Commands, pos: Vec2) {
    let bar = || Node {
        width: Val::Px(20.0),
        height: Val::Px(2.5),
        margin: UiRect::vertical(Val::Px(2.0)),
        ..default()
    };
    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(pos.y),
                left: Val::Px(pos.x),
                width: Val::Px(36.0),
                height: Val::Px(36.0),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.16, 0.16, 0.22, 0.9)),
            HamburgerButton,
            HideDuringStartMenu,
        ))
        .with_children(|parent| {
            let bar_color = Color::srgb(0.85, 0.85, 0.9);
            parent.spawn((bar(), BackgroundColor(bar_color)));
            parent.spawn((bar(), BackgroundColor(bar_color)));
            parent.spawn((bar(), BackgroundColor(bar_color)));
        });
}

fn update_mode_visibility(
    mode: Res<UiMode>,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    mut playground_q: Query<&mut Node, (With<PlaygroundOnly>, Without<ExamplesOnly>)>,
    mut examples_q: Query<&mut Node, (With<ExamplesOnly>, Without<PlaygroundOnly>)>,
    mut text_inputs: Query<&mut TextInput>,
) {
    if start_menu.showing || modal_is_open(&eval) {
        return;
    }
    if !mode.is_changed() && !start_menu.is_changed() && !eval.is_changed() {
        return;
    }
    let (playground_display, examples_display) = match *mode {
        UiMode::Playground => (Display::Flex, Display::None),
        UiMode::Examples => (Display::None, Display::Flex),
    };
    for mut node in playground_q.iter_mut() {
        node.display = playground_display;
    }
    for mut node in examples_q.iter_mut() {
        node.display = examples_display;
    }
    if *mode != UiMode::Playground {
        for mut input in text_inputs.iter_mut() {
            input.focused = false;
        }
    }
}

fn handle_delete_node_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<DeleteNodeButton>)>,
    mut state: ResMut<AstState>,
    mut rebuild: ResMut<NeedsRebuild>,
    pick: Res<PickState>,
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    for interaction in interaction_q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if let Some(selected_node_id) = state.layout_ast.node_at(pick.selected_pos) {
            let is_sink_wall = matches!(
                state.layout_ast.ast.nodes.get(&selected_node_id),
                Some(ast::node::ENode::SinkWall { .. })
            );
            if !is_sink_wall {
                state.layout_ast = state.layout_ast.minus_node(&selected_node_id);
                rebuild.0 = true;
            }
        }
    }
}

fn update_delete_button_visuals(
    pick: Res<PickState>,
    state: Res<AstState>,
    mut button_q: Query<(&Interaction, &mut BackgroundColor, &Children), With<DeleteNodeButton>>,
    mut text_color_q: Query<&mut TextColor>,
) {
    let enabled = match state.layout_ast.node_at(pick.selected_pos) {
        Some(id) => !matches!(
            state.layout_ast.ast.nodes.get(&id),
            Some(ast::node::ENode::SinkWall { .. })
        ),
        None => false,
    };
    for (interaction, mut bg, children) in button_q.iter_mut() {
        let Ok(mut text_color) = text_color_q.get_mut(children[0]) else {
            continue;
        };
        if !enabled {
            bg.0 = Color::srgba(0.10, 0.10, 0.13, 0.9);
            text_color.0 = Color::srgb(0.35, 0.35, 0.4);
            continue;
        }
        match *interaction {
            Interaction::Hovered | Interaction::Pressed => {
                bg.0 = Color::srgba(0.2, 0.2, 0.3, 0.95);
                text_color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.16, 0.16, 0.22, 0.9);
                text_color.0 = Color::srgb(0.6, 0.6, 0.7);
            }
        }
    }
}

fn handle_example_buttons(
    mut interaction_q: Query<
        (
            &Interaction,
            &mut BackgroundColor,
            &Children,
            &ExampleButton,
        ),
        (Changed<Interaction>, With<ExampleButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut state: ResMut<AstState>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut pick: ResMut<PickState>,
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    for (interaction, mut bg, children, kind) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();

        match *interaction {
            Interaction::Pressed => {
                state.layout_ast = match kind {
                    ExampleButton::Sink => layout::LayoutAst::empty().plus_sink_example(),
                    ExampleButton::VarDecl => layout::LayoutAst::empty().plus_vardecl_example(),
                    ExampleButton::ConstDecl => layout::LayoutAst::empty().plus_constdecl_example(),
                    ExampleButton::FuncCall => {
                        let decl = state
                            .function_declarations
                            .iter()
                            .find(|(_, d)| d.name == "charAt")
                            .map(|(id, decl)| (id.clone(), decl))
                            .unwrap();
                        layout::LayoutAst::empty().plus_funccall_example(decl)
                    }
                    ExampleButton::Match => layout::LayoutAst::empty().plus_match_example(),
                    _ => layout::LayoutAst::empty().plus_sink_wall(),
                };
                pick.selected_pos = IVec3::ZERO;
                pick.hovered_node = None;
                pick.hovered_pos = None;
                rebuild.0 = true;
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.2, 0.2, 0.3, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.16, 0.16, 0.22, 0.9);
                color.0 = Color::srgb(0.6, 0.6, 0.7);
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
        (Changed<Interaction>, With<EAstActionButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut state: ResMut<AstState>,
    mut orbit: ResMut<camera::OrbitCamera>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut pick: ResMut<PickState>,
    mut commands: Commands,
    scene_entities: Query<Entity, With<AstSceneEntity>>,
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    for (interaction, mut bg, children, action) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();

        match *interaction {
            Interaction::Pressed => {
                let new_pos = pick.selected_pos.as_vec3();
                state.layout_ast = match action {
                    EAstActionButton::AddConstDeclButton => state
                        .layout_ast
                        .plus_type_introduction(ast::node::EType::Int { value: None }, new_pos),
                    EAstActionButton::AddVarDeclButton => state.layout_ast.plus_var_decl(new_pos),
                    EAstActionButton::AddFunctionCallButton => state.layout_ast.plus_function_call(
                        state
                            .function_declarations
                            .iter()
                            .find(|(_, d)| d.name == "+")
                            .map(|(id, decl)| (id.clone(), decl))
                            .unwrap(),
                        new_pos,
                    ),
                    EAstActionButton::AddTypeCastButton => state
                        .layout_ast
                        .plus_type_elimination(ast::node::EType::Int { value: None }, new_pos),
                    EAstActionButton::AddMatchFrontButton => {
                        state.layout_ast.plus_match_front(new_pos)
                    }
                };
                rebuild.0 = true;
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.2, 0.2, 0.3, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.16, 0.16, 0.22, 0.9);
                color.0 = Color::srgb(0.6, 0.6, 0.7);
            }
        }
    }
}

// ── Node editor UI ──────────────────────────────────────────

fn spawn_node_editor_panel(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(60.0),
            right: Val::Px(14.0),
            width: Val::Px(280.0),
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(Val::Px(10.0)),
            border_radius: BorderRadius::all(Val::Px(6.0)),
            row_gap: Val::Px(6.0),
            display: Display::None,
            ..default()
        },
        BackgroundColor(Color::srgba(0.10, 0.10, 0.16, 0.9)),
        Button,
        NodeEditorPanel,
    ));
}

#[derive(Default, PartialEq, Eq, Clone)]
struct NodeEditorFingerprint {
    node_id: Option<ast::node::Id>,
    variant: NodeVariantKind,
    type_choice: Option<TypeChoice>,
    typeelim_has_value: bool,
    func_id: Option<ast::FunctionDeclarationId>,
    dropdown_open: Option<(ast::node::Id, DropdownKind)>,
    visible: bool,
}

fn sync_node_editor_ui(
    mut commands: Commands,
    state: Res<AstState>,
    pick: Res<PickState>,
    dropdown_state: Res<DropdownState>,
    ui_mode: Res<UiMode>,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    ui_font: Res<UiFont>,
    mut panel_q: Query<(Entity, &mut Node), With<NodeEditorPanel>>,
    editor_children_q: Query<Entity, With<NodeEditorEntity>>,
    mut cache: Local<NodeEditorFingerprint>,
) {
    let node_id = state.layout_ast.node_at(pick.selected_pos);
    let node = node_id
        .as_ref()
        .and_then(|id| state.layout_ast.ast.nodes.get(id));
    let variant = variant_kind(node);
    let type_choice = node.and_then(|n| match n {
        ast::node::ENode::TypeIntroduction { r#type, .. }
        | ast::node::ENode::TypeElimination { r#type, .. } => type_choice_of(r#type),
        _ => None,
    });
    let typeelim_has_value = match node {
        Some(ast::node::ENode::TypeElimination { r#type, .. }) => value_of_etype(r#type).is_some(),
        _ => false,
    };
    let func_id = match node {
        Some(ast::node::ENode::FunctionCall {
            function_declaration_id,
            ..
        }) => Some(function_declaration_id.clone()),
        _ => None,
    };

    let editable = matches!(
        variant,
        NodeVariantKind::TypeIntroduction
            | NodeVariantKind::TypeElimination
            | NodeVariantKind::VarDecl
            | NodeVariantKind::FunctionCall
    );
    let visible =
        editable && *ui_mode == UiMode::Playground && !start_menu.showing && !is_evaluating(&eval);

    let fp = NodeEditorFingerprint {
        node_id: node_id.clone(),
        variant,
        type_choice,
        typeelim_has_value,
        func_id: func_id.clone(),
        dropdown_open: dropdown_state.open.clone(),
        visible,
    };

    if *cache == fp {
        return;
    }
    *cache = fp;

    for e in editor_children_q.iter() {
        commands.entity(e).despawn();
    }

    let Ok((panel_entity, mut panel_node)) = panel_q.single_mut() else {
        return;
    };
    panel_node.display = if visible {
        Display::Flex
    } else {
        Display::None
    };
    if !visible {
        return;
    }

    let node = node.unwrap();
    let node_id = node_id.unwrap();

    let font = &ui_font.0;
    commands
        .entity(panel_entity)
        .with_children(|panel| match node {
            ast::node::ENode::TypeIntroduction { r#type, .. } => {
                spawn_editor_label(panel, font, "ConstDecl");
                spawn_labeled_row(panel, font, "Type", |slot| {
                    spawn_type_dropdown(slot, font, &node_id, r#type, &dropdown_state.open);
                });
                if !matches!(r#type, ast::node::EType::Undefined) {
                    spawn_labeled_row(panel, font, "Value", |slot| {
                        spawn_value_widget(
                            slot,
                            font,
                            &node_id,
                            r#type,
                            true,
                            &dropdown_state.open,
                        );
                    });
                }
            }
            ast::node::ENode::VarDecl { name, r#type, .. } => {
                spawn_labeled_row(panel, font, "Name", |slot| {
                    spawn_name_input(slot, font, &node_id, name);
                });
                spawn_labeled_row(panel, font, "Type", |slot| {
                    spawn_type_dropdown(slot, font, &node_id, r#type, &dropdown_state.open);
                });
            }
            ast::node::ENode::TypeElimination { r#type, .. } => {
                spawn_editor_label(panel, font, "TypeCast");
                spawn_labeled_row(panel, font, "Type", |slot| {
                    spawn_type_dropdown(slot, font, &node_id, r#type, &dropdown_state.open);
                });
                if !matches!(r#type, ast::node::EType::Undefined) {
                    spawn_labeled_row(panel, font, "Value", |slot| {
                        spawn_typeelim_checkbox_and_value(
                            slot,
                            font,
                            &node_id,
                            r#type,
                            &dropdown_state.open,
                        );
                    });
                }
            }
            ast::node::ENode::FunctionCall {
                function_declaration_id,
                ..
            } => {
                spawn_labeled_row(panel, font, "Function", |slot| {
                    spawn_function_dropdown(
                        slot,
                        font,
                        &node_id,
                        function_declaration_id,
                        &state.function_declarations,
                        &dropdown_state.open,
                    );
                });
            }
            _ => {}
        });
}

fn spawn_editor_label(panel: &mut ChildSpawnerCommands, font: &Handle<Font>, text: &str) {
    panel.spawn((
        Text::new(text),
        text_font(font, 15.0),
        TextColor(Color::srgb(0.75, 0.75, 0.9)),
        NodeEditorEntity,
    ));
}

fn spawn_labeled_row(
    panel: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    label: &str,
    widget: impl FnOnce(&mut ChildSpawnerCommands),
) {
    panel
        .spawn((
            Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                ..default()
            },
            NodeEditorEntity,
        ))
        .with_children(|row| {
            row.spawn((
                Text::new(label.to_string()),
                text_font(font, 13.0),
                TextColor(Color::srgb(0.6, 0.6, 0.7)),
                Node {
                    width: Val::Px(70.0),
                    flex_shrink: 0.0,
                    ..default()
                },
                NodeEditorEntity,
            ));
            row.spawn((
                Node {
                    flex_grow: 1.0,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(6.0),
                    ..default()
                },
                NodeEditorEntity,
            ))
            .with_children(widget);
        });
}

fn spawn_name_input(
    panel: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    node_id: &ast::node::Id,
    current: &str,
) {
    panel
        .spawn((
            Button,
            editor_text_input_node(),
            BackgroundColor(Color::srgba(0.06, 0.06, 0.12, 0.95)),
            BorderColor::all(Color::srgb(0.12, 0.12, 0.24)),
            TextInputBox,
            TextInput {
                value: current.to_string(),
                focused: false,
                cursor: current.len(),
            },
            NodeEditorTextInput {
                node_id: node_id.clone(),
                field: NodeEditorField::VarDeclName,
            },
            NodeEditorEntity,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(current.to_string()),
                text_font(font, 14.0),
                TextColor(Color::srgb(0.91, 0.89, 0.87)),
                TextInputDisplay,
                NodeEditorEntity,
            ));
        });
}

fn spawn_type_dropdown(
    panel: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    node_id: &ast::node::Id,
    current: &ast::node::EType,
    open: &Option<(ast::node::Id, DropdownKind)>,
) {
    let current_choice = type_choice_of(current);
    let label = current_choice.map(type_choice_label).unwrap_or("?");
    spawn_dropdown_root(
        panel,
        font,
        node_id,
        DropdownKind::Type,
        label,
        open,
        |options| {
            for tc in TYPE_CHOICES {
                let is_current = Some(tc) == current_choice;
                spawn_dropdown_option(
                    options,
                    font,
                    node_id,
                    DropdownKind::Type,
                    DropdownChoice::Type(tc),
                    type_choice_label(tc),
                    is_current,
                );
            }
        },
    );
}

fn spawn_function_dropdown(
    panel: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    node_id: &ast::node::Id,
    current: &ast::FunctionDeclarationId,
    declarations: &std::collections::HashMap<ast::FunctionDeclarationId, ast::FunctionDeclaration>,
    open: &Option<(ast::node::Id, DropdownKind)>,
) {
    let label = declarations
        .get(current)
        .map(|d| d.name.as_str())
        .unwrap_or("?");
    let mut entries: Vec<(ast::FunctionDeclarationId, String)> = declarations
        .iter()
        .map(|(id, d)| (id.clone(), d.name.clone()))
        .collect();
    entries.sort_by(|a, b| a.1.cmp(&b.1));

    spawn_dropdown_root(
        panel,
        font,
        node_id,
        DropdownKind::Function,
        label,
        open,
        |options| {
            for (id, name) in &entries {
                let is_current = id == current;
                spawn_dropdown_option(
                    options,
                    font,
                    node_id,
                    DropdownKind::Function,
                    DropdownChoice::Function(id.clone()),
                    name,
                    is_current,
                );
            }
        },
    );
}

fn spawn_value_widget(
    panel: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    node_id: &ast::node::Id,
    current: &ast::node::EType,
    enabled: bool,
    open: &Option<(ast::node::Id, DropdownKind)>,
) {
    match current {
        ast::node::EType::Undefined => {}
        ast::node::EType::Bool { value } => {
            let current_bool = value.as_deref() == Some("true");
            let label = value.as_deref().unwrap_or("bool");
            spawn_dropdown_root(
                panel,
                font,
                node_id,
                DropdownKind::BoolValue,
                label,
                open,
                |options| {
                    for v in [true, false] {
                        spawn_dropdown_option(
                            options,
                            font,
                            node_id,
                            DropdownKind::BoolValue,
                            DropdownChoice::BoolValue(v),
                            if v { "true" } else { "false" },
                            value.is_some() && v == current_bool,
                        );
                    }
                },
            );
        }
        _ => {
            let initial = value_of_etype(current).unwrap_or_default();
            let (bg, border, fg) = if enabled {
                (
                    Color::srgba(0.06, 0.06, 0.12, 0.95),
                    Color::srgb(0.12, 0.12, 0.24),
                    Color::srgb(0.91, 0.89, 0.87),
                )
            } else {
                (
                    Color::srgba(0.06, 0.06, 0.12, 0.6),
                    Color::srgb(0.12, 0.12, 0.24),
                    Color::srgb(0.45, 0.45, 0.5),
                )
            };
            let mut e = panel.spawn((
                Button,
                editor_text_input_node(),
                BackgroundColor(bg),
                BorderColor::all(border),
                TextInput {
                    value: initial.clone(),
                    focused: false,
                    cursor: initial.len(),
                },
                NodeEditorTextInput {
                    node_id: node_id.clone(),
                    field: NodeEditorField::Value,
                },
                NodeEditorEntity,
            ));
            if enabled {
                e.insert(TextInputBox);
            }
            e.with_children(|p| {
                p.spawn((
                    Text::new(initial),
                    text_font(font, 14.0),
                    TextColor(fg),
                    TextInputDisplay,
                    NodeEditorEntity,
                ));
            });
        }
    }
}

fn spawn_typeelim_checkbox_and_value(
    row: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    node_id: &ast::node::Id,
    current: &ast::node::EType,
    open: &Option<(ast::node::Id, DropdownKind)>,
) {
    let enabled = value_of_etype(current).is_some();
    row.spawn((
        Button,
        Node {
            width: Val::Px(16.0),
            height: Val::Px(16.0),
            border: UiRect::all(Val::Px(1.5)),
            border_radius: BorderRadius::all(Val::Px(3.0)),
            flex_shrink: 0.0,
            ..default()
        },
        BackgroundColor(if enabled {
            Color::srgb(0.133, 0.827, 0.933)
        } else {
            Color::srgba(0.06, 0.06, 0.12, 0.95)
        }),
        BorderColor::all(Color::srgb(0.35, 0.35, 0.5)),
        ValueEnableCheckbox {
            node_id: node_id.clone(),
        },
        NodeEditorEntity,
    ));
    spawn_value_widget(row, font, node_id, current, enabled, open);
}

fn spawn_dropdown_root(
    panel: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    node_id: &ast::node::Id,
    kind: DropdownKind,
    label: &str,
    open: &Option<(ast::node::Id, DropdownKind)>,
    spawn_options: impl FnOnce(&mut ChildSpawnerCommands),
) {
    let is_open = matches!(open, Some((oid, ok)) if oid == node_id && *ok == kind);
    let mut root = panel.spawn((
        Node {
            flex_direction: FlexDirection::Column,
            flex_grow: 1.0,
            ..default()
        },
        NodeEditorEntity,
    ));
    root.with_children(|dd| {
        dd.spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                border: UiRect::all(Val::Px(1.5)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                flex_direction: FlexDirection::Row,
                ..default()
            },
            BackgroundColor(Color::srgba(0.06, 0.06, 0.12, 0.95)),
            BorderColor::all(Color::srgb(0.12, 0.12, 0.24)),
            Dropdown {
                node_id: node_id.clone(),
                kind,
            },
            NodeEditorEntity,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(label.to_string()),
                text_font(font, 14.0),
                TextColor(Color::srgb(0.91, 0.89, 0.87)),
                NodeEditorEntity,
            ));
            b.spawn((
                Text::new(if is_open { "▴" } else { "▾" }),
                text_font(font, 12.0),
                TextColor(Color::srgb(0.6, 0.6, 0.7)),
                NodeEditorEntity,
            ));
        });
        if is_open {
            dd.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    margin: UiRect::top(Val::Px(2.0)),
                    padding: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.08, 0.08, 0.14, 0.98)),
                NodeEditorEntity,
            ))
            .with_children(|options| {
                spawn_options(options);
            });
        }
    });
}

fn spawn_dropdown_option(
    options: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    node_id: &ast::node::Id,
    kind: DropdownKind,
    choice: DropdownChoice,
    label: &str,
    is_current: bool,
) {
    options
        .spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                justify_content: JustifyContent::SpaceBetween,
                flex_direction: FlexDirection::Row,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.0)),
            DropdownOption {
                node_id: node_id.clone(),
                kind,
                choice,
            },
            NodeEditorEntity,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(label.to_string()),
                text_font(font, 14.0),
                TextColor(Color::srgb(0.85, 0.85, 0.9)),
                NodeEditorEntity,
            ));
            b.spawn((
                Text::new(if is_current { "✓" } else { " " }),
                text_font(font, 13.0),
                TextColor(Color::srgb(0.133, 0.827, 0.933)),
                NodeEditorEntity,
            ));
        });
}

fn editor_text_input_node() -> Node {
    Node {
        padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
        border: UiRect::all(Val::Px(1.5)),
        border_radius: BorderRadius::all(Val::Px(4.0)),
        flex_grow: 1.0,
        min_width: Val::Px(0.0),
        ..default()
    }
}

fn handle_dropdown_click(
    interaction_q: Query<(&Interaction, &Dropdown), Changed<Interaction>>,
    mut dropdown_state: ResMut<DropdownState>,
) {
    for (interaction, dd) in interaction_q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let key = (dd.node_id.clone(), dd.kind);
        dropdown_state.open = if dropdown_state.open.as_ref() == Some(&key) {
            None
        } else {
            Some(key)
        };
    }
}

fn handle_dropdown_option_click(
    interaction_q: Query<(&Interaction, &DropdownOption), Changed<Interaction>>,
    mut state: ResMut<AstState>,
    mut dropdown_state: ResMut<DropdownState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    for (interaction, option) in interaction_q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match &option.choice {
            DropdownChoice::Type(new_choice) => {
                let node = state.layout_ast.ast.nodes.get_mut(&option.node_id);
                match node {
                    Some(ast::node::ENode::TypeIntroduction { r#type, .. })
                    | Some(ast::node::ENode::TypeElimination { r#type, .. })
                    | Some(ast::node::ENode::VarDecl { r#type, .. }) => {
                        let value = value_of_etype(r#type);
                        *r#type = make_etype(*new_choice, value);
                        rebuild.0 = true;
                    }
                    _ => {}
                }
            }
            DropdownChoice::BoolValue(v) => {
                let node = state.layout_ast.ast.nodes.get_mut(&option.node_id);
                match node {
                    Some(ast::node::ENode::TypeIntroduction { r#type, .. })
                    | Some(ast::node::ENode::TypeElimination { r#type, .. })
                    | Some(ast::node::ENode::VarDecl { r#type, .. }) => {
                        if let ast::node::EType::Bool { value } = r#type {
                            *value = Some(if *v { "true" } else { "false" }.to_string());
                            rebuild.0 = true;
                        }
                    }
                    _ => {}
                }
            }
            DropdownChoice::Function(new_fn_id) => {
                let decl = state.function_declarations.get(new_fn_id).cloned();
                if let Some(new_decl) = decl {
                    let new_layout = state.layout_ast.with_function_call_replaced(
                        &option.node_id,
                        (new_fn_id.clone(), &new_decl),
                    );
                    state.layout_ast = new_layout;
                    rebuild.0 = true;
                }
            }
        }
        dropdown_state.open = None;
    }
}

fn handle_value_enable_checkbox(
    interaction_q: Query<(&Interaction, &ValueEnableCheckbox), Changed<Interaction>>,
    mut state: ResMut<AstState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    for (interaction, cb) in interaction_q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if let Some(ast::node::ENode::TypeElimination { r#type, .. }) =
            state.layout_ast.ast.nodes.get_mut(&cb.node_id)
        {
            let current = value_of_etype(r#type);
            let toggled: Option<String> = if current.is_some() {
                None
            } else {
                Some(String::new())
            };
            if let Some(choice) = type_choice_of(r#type) {
                *r#type = make_etype(choice, toggled);
                rebuild.0 = true;
            }
        }
    }
}

fn handle_node_editor_text_input(
    input_q: Query<(&NodeEditorTextInput, &TextInput), Changed<TextInput>>,
    mut state: ResMut<AstState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    for (editor_input, input) in input_q.iter() {
        let Some(node) = state.layout_ast.ast.nodes.get_mut(&editor_input.node_id) else {
            continue;
        };
        match editor_input.field {
            NodeEditorField::VarDeclName => {
                if let ast::node::ENode::VarDecl { name, .. } = node {
                    if *name != input.value {
                        *name = input.value.clone();
                        rebuild.0 = true;
                    }
                }
            }
            NodeEditorField::Value => {
                let r#type = match node {
                    ast::node::ENode::TypeIntroduction { r#type, .. }
                    | ast::node::ENode::TypeElimination { r#type, .. }
                    | ast::node::ENode::VarDecl { r#type, .. } => r#type,
                    _ => continue,
                };
                let Some(choice) = type_choice_of(r#type) else {
                    continue;
                };
                // Ignore the initial spawn's Added-tick: an empty input against
                // a None value is not a user edit.
                if input.value.is_empty() && value_of_etype(r#type).is_none() {
                    continue;
                }
                let new_value = Some(input.value.clone());
                if value_of_etype(r#type) != new_value {
                    *r#type = make_etype(choice, new_value);
                    rebuild.0 = true;
                }
            }
        }
    }
}

fn handle_evaluate_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<EvaluateButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut eval: ResMut<EvalState>,
    state: Res<AstState>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                if is_evaluating(&eval) {
                    // Already showing a modal or running — ignore.
                    continue;
                }
                let ast = &state.layout_ast.ast;
                if !eval::sink_has_input(ast) {
                    eval.phase = EvalPhase::ErrorModal(
                        "Cannot evaluate, because no node is connected to the sink".to_string(),
                    );
                    continue;
                }
                let var_decls = eval::collect_var_decls(ast);
                if !var_decls.is_empty() {
                    eval.phase = EvalPhase::VarDeclPrompt {
                        inputs: var_decls
                            .into_iter()
                            .map(|(id, _name)| (id, String::new()))
                            .collect(),
                    };
                } else {
                    let initial =
                        eval::initial_values(ast, &std::collections::HashMap::new(), &mut eval.rng);
                    eval.phase = EvalPhase::Running {
                        steps: vec![initial],
                        current: 0,
                    };
                }
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.2, 0.2, 0.3, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.16, 0.16, 0.22, 0.9);
                color.0 = Color::srgb(0.6, 0.6, 0.7);
            }
        }
    }
}

/// Identifier of the currently displayed modal kind, used by sync_modal_ui
/// to detect phase transitions without comparing full enum payloads.
#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum ModalKind {
    #[default]
    None,
    Error,
    Controls,
    VarDeclPrompt,
}

fn modal_kind(phase: &EvalPhase) -> ModalKind {
    match phase {
        EvalPhase::ErrorModal(_) => ModalKind::Error,
        EvalPhase::ControlsModal => ModalKind::Controls,
        EvalPhase::VarDeclPrompt { .. } => ModalKind::VarDeclPrompt,
        _ => ModalKind::None,
    }
}

fn sync_modal_ui(
    mut commands: Commands,
    eval: Res<EvalState>,
    state: Res<AstState>,
    ui_font: Res<UiFont>,
    modal_q: Query<Entity, With<ModalEntity>>,
    mut last_kind: Local<ModalKind>,
) {
    let kind_now = modal_kind(&eval.phase);
    if kind_now == *last_kind {
        return;
    }
    *last_kind = kind_now;
    // Tear down the previous modal.
    for e in modal_q.iter() {
        commands.entity(e).despawn();
    }

    match &eval.phase {
        EvalPhase::ErrorModal(msg) => {
            spawn_error_modal(&mut commands, &ui_font.0, msg.clone());
        }
        EvalPhase::ControlsModal => {
            spawn_controls_modal(&mut commands, &ui_font.0);
        }
        EvalPhase::VarDeclPrompt { inputs } => {
            let ast = &state.layout_ast.ast;
            let rows: Vec<(ast::node::Id, String)> = inputs
                .iter()
                .map(|(id, _)| {
                    let name = match ast.nodes.get(id) {
                        Some(ast::node::ENode::VarDecl { name, .. }) => name.clone(),
                        _ => "?".to_string(),
                    };
                    (id.clone(), name)
                })
                .collect();
            spawn_vardecl_modal(&mut commands, &ui_font.0, rows);
        }
        _ => {}
    }
}

fn spawn_error_modal(commands: &mut Commands, font: &Handle<Font>, msg: String) {
    commands
        .spawn((
            backdrop_node(),
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            GlobalZIndex(50),
            ModalEntity,
        ))
        .with_children(|root| {
            root.spawn((
                panel_node(),
                BackgroundColor(Color::srgba(0.10, 0.10, 0.16, 0.98)),
                BorderColor::all(Color::srgb(0.25, 0.25, 0.4)),
                ModalEntity,
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new(msg),
                    text_font(font, 16.0),
                    TextColor(Color::srgb(0.95, 0.95, 1.0)),
                    Node {
                        margin: UiRect::all(Val::Px(12.0)),
                        ..default()
                    },
                    ModalEntity,
                ));
                panel
                    .spawn((
                        Button,
                        modal_button_node(),
                        BackgroundColor(Color::srgba(0.18, 0.18, 0.28, 0.95)),
                        ModalOkButton,
                        ModalEntity,
                    ))
                    .with_children(|b| {
                        b.spawn((
                            Text::new("OK"),
                            text_font(font, 14.0),
                            TextColor(Color::srgb(0.85, 0.85, 0.9)),
                        ));
                    });
            });
        });
}

fn spawn_controls_modal(commands: &mut Commands, font: &Handle<Font>) {
    let mouse_bindings: &[(&str, &str)] = &[
        ("Left click", "Select node / grid position"),
        ("Left drag on anchor", "Connect nodes"),
        ("Ctrl + Left drag", "Orbit camera"),
        ("Ctrl + Right drag", "Pan camera"),
        ("Ctrl + Scroll", "Zoom"),
    ];
    let key_bindings: &[(&str, &str)] = &[
        ("Arrow keys", "Move selection along the grid"),
        ("Ctrl + Arrow", "Move the selected node"),
        (
            "Ctrl + Shift + Up/Down",
            "Move the selected node vertically",
        ),
        ("Escape", "Unfocus text input"),
    ];

    commands
        .spawn((
            backdrop_node(),
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            GlobalZIndex(50),
            ModalEntity,
        ))
        .with_children(|root| {
            root.spawn((
                panel_node(),
                BackgroundColor(Color::srgba(0.10, 0.10, 0.16, 0.98)),
                BorderColor::all(Color::srgb(0.25, 0.25, 0.4)),
                ModalEntity,
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("Controls"),
                    text_font(font, 20.0),
                    TextColor(Color::srgb(0.95, 0.95, 1.0)),
                    Node {
                        margin: UiRect::all(Val::Px(12.0)),
                        align_self: AlignSelf::Center,
                        ..default()
                    },
                    ModalEntity,
                ));

                spawn_controls_section(panel, font, "Mouse", mouse_bindings);
                spawn_controls_section(panel, font, "Keyboard", key_bindings);

                panel
                    .spawn((
                        Node {
                            flex_direction: FlexDirection::Row,
                            justify_content: JustifyContent::Center,
                            margin: UiRect::all(Val::Px(8.0)),
                            ..default()
                        },
                        ModalEntity,
                    ))
                    .with_children(|btns| {
                        btns.spawn((
                            Button,
                            modal_button_node(),
                            BackgroundColor(Color::srgba(0.18, 0.18, 0.28, 0.95)),
                            ControlsModalOkButton,
                            ModalEntity,
                        ))
                        .with_children(|b| {
                            b.spawn((
                                Text::new("OK"),
                                text_font(font, 14.0),
                                TextColor(Color::srgb(0.85, 0.85, 0.9)),
                                ModalEntity,
                            ));
                        });
                    });
            });
        });
}

fn spawn_controls_section(
    panel: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    title: &str,
    rows: &[(&str, &str)],
) {
    panel.spawn((
        Text::new(title),
        text_font(font, 15.0),
        TextColor(Color::srgb(0.75, 0.75, 0.9)),
        Node {
            margin: UiRect {
                left: Val::Px(12.0),
                right: Val::Px(12.0),
                top: Val::Px(8.0),
                bottom: Val::Px(4.0),
            },
            ..default()
        },
        ModalEntity,
    ));
    for (binding, desc) in rows {
        panel
            .spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    margin: UiRect::axes(Val::Px(12.0), Val::Px(2.0)),
                    ..default()
                },
                ModalEntity,
            ))
            .with_children(|row| {
                row.spawn((
                    Text::new(*binding),
                    text_font(font, 14.0),
                    TextColor(Color::srgb(0.9, 0.9, 0.7)),
                    Node {
                        width: Val::Px(200.0),
                        ..default()
                    },
                    ModalEntity,
                ));
                row.spawn((
                    Text::new(*desc),
                    text_font(font, 14.0),
                    TextColor(Color::srgb(0.85, 0.85, 0.9)),
                    ModalEntity,
                ));
            });
    }
}

fn spawn_vardecl_modal(
    commands: &mut Commands,
    font: &Handle<Font>,
    rows: Vec<(ast::node::Id, String)>,
) {
    commands
        .spawn((
            backdrop_node(),
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            GlobalZIndex(50),
            ModalEntity,
        ))
        .with_children(|root| {
            root.spawn((
                panel_node(),
                BackgroundColor(Color::srgba(0.10, 0.10, 0.16, 0.98)),
                BorderColor::all(Color::srgb(0.25, 0.25, 0.4)),
                ModalEntity,
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("Enter values for the variable declarations:"),
                    text_font(font, 16.0),
                    TextColor(Color::srgb(0.95, 0.95, 1.0)),
                    Node {
                        margin: UiRect::all(Val::Px(8.0)),
                        ..default()
                    },
                    ModalEntity,
                ));
                for (node_id, name) in rows {
                    panel
                        .spawn((
                            Node {
                                flex_direction: FlexDirection::Row,
                                align_items: AlignItems::Center,
                                margin: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                                ..default()
                            },
                            ModalEntity,
                        ))
                        .with_children(|row| {
                            row.spawn((
                                Text::new(format!("{}:", name)),
                                text_font(font, 14.0),
                                TextColor(Color::srgb(0.85, 0.85, 0.9)),
                                Node {
                                    width: Val::Px(120.0),
                                    ..default()
                                },
                                ModalEntity,
                            ));
                            row.spawn((
                                Button,
                                Node {
                                    padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                                    min_width: Val::Px(180.0),
                                    border: UiRect::all(Val::Px(1.5)),
                                    border_radius: BorderRadius::all(Val::Px(4.0)),
                                    ..default()
                                },
                                BackgroundColor(Color::srgba(0.06, 0.06, 0.12, 0.95)),
                                BorderColor::all(Color::srgb(0.12, 0.12, 0.24)),
                                TextInputBox,
                                TextInput {
                                    value: String::new(),
                                    focused: false,
                                    cursor: 0,
                                },
                                ModalVarDeclInput { node_id },
                                ModalEntity,
                            ))
                            .with_children(|input| {
                                input.spawn((
                                    Text::new(""),
                                    text_font(font, 14.0),
                                    TextColor(Color::srgb(0.91, 0.89, 0.87)),
                                    TextInputDisplay,
                                    ModalEntity,
                                ));
                            });
                        });
                }
                panel
                    .spawn((
                        Node {
                            flex_direction: FlexDirection::Row,
                            justify_content: JustifyContent::FlexEnd,
                            margin: UiRect::all(Val::Px(8.0)),
                            ..default()
                        },
                        ModalEntity,
                    ))
                    .with_children(|btns| {
                        btns.spawn((
                            Button,
                            modal_button_node(),
                            BackgroundColor(Color::srgba(0.18, 0.18, 0.28, 0.95)),
                            ModalCancelButton,
                            ModalEntity,
                        ))
                        .with_children(|b| {
                            b.spawn((
                                Text::new("Cancel"),
                                text_font(font, 14.0),
                                TextColor(Color::srgb(0.85, 0.85, 0.9)),
                            ));
                        });
                        btns.spawn((
                            Button,
                            modal_button_node(),
                            BackgroundColor(Color::srgba(0.18, 0.18, 0.28, 0.95)),
                            ModalEvaluateButton,
                            ModalEntity,
                        ))
                        .with_children(|b| {
                            b.spawn((
                                Text::new("Evaluate"),
                                text_font(font, 14.0),
                                TextColor(Color::srgb(0.85, 0.85, 0.9)),
                            ));
                        });
                    });
            });
        });
}

fn backdrop_node() -> Node {
    Node {
        position_type: PositionType::Absolute,
        top: Val::Px(0.0),
        left: Val::Px(0.0),
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        ..default()
    }
}

fn panel_node() -> Node {
    Node {
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Stretch,
        padding: UiRect::all(Val::Px(16.0)),
        min_width: Val::Px(360.0),
        border_radius: BorderRadius::all(Val::Px(8.0)),
        border: UiRect::all(Val::Px(1.0)),
        ..default()
    }
}

fn modal_button_node() -> Node {
    Node {
        padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
        margin: UiRect::axes(Val::Px(6.0), Val::Px(0.0)),
        border_radius: BorderRadius::all(Val::Px(6.0)),
        ..default()
    }
}

fn spawn_start_menu(mut commands: Commands, start_menu: Res<StartMenu>, ui_font: Res<UiFont>) {
    spawn_start_menu_ui(&mut commands, &ui_font.0, start_menu.has_cancel);
}

fn spawn_start_menu_ui(commands: &mut Commands, font: &Handle<Font>, has_cancel: bool) {
    commands
        .spawn((
            backdrop_node(),
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            GlobalZIndex(50),
            StartMenuEntity,
        ))
        .with_children(|root| {
            root.spawn((
                panel_node(),
                BackgroundColor(Color::srgba(0.10, 0.10, 0.16, 0.98)),
                BorderColor::all(Color::srgb(0.25, 0.25, 0.4)),
                StartMenuEntity,
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("Expression Visualizer"),
                    text_font(font, 24.0),
                    TextColor(Color::srgb(0.95, 0.95, 1.0)),
                    Node {
                        margin: UiRect::all(Val::Px(12.0)),
                        align_self: AlignSelf::Center,
                        ..default()
                    },
                    StartMenuEntity,
                ));
                panel
                    .spawn((
                        Node {
                            flex_direction: FlexDirection::Row,
                            justify_content: JustifyContent::Center,
                            margin: UiRect::all(Val::Px(8.0)),
                            ..default()
                        },
                        StartMenuEntity,
                    ))
                    .with_children(|btns| {
                        btns.spawn((
                            Button,
                            modal_button_node(),
                            BackgroundColor(Color::srgba(0.18, 0.18, 0.28, 0.95)),
                            StartMenuNewButton,
                            StartMenuEntity,
                        ))
                        .with_children(|b| {
                            b.spawn((
                                Text::new("New"),
                                text_font(font, 14.0),
                                TextColor(Color::srgb(0.85, 0.85, 0.9)),
                                StartMenuEntity,
                            ));
                        });
                        btns.spawn((
                            Button,
                            modal_button_node(),
                            BackgroundColor(Color::srgba(0.18, 0.18, 0.28, 0.95)),
                            StartMenuLoadExampleButton,
                            StartMenuEntity,
                        ))
                        .with_children(|b| {
                            b.spawn((
                                Text::new("Load Example"),
                                text_font(font, 14.0),
                                TextColor(Color::srgb(0.85, 0.85, 0.9)),
                                StartMenuEntity,
                            ));
                        });
                        btns.spawn((
                            Button,
                            modal_button_node(),
                            BackgroundColor(Color::srgba(0.18, 0.18, 0.28, 0.95)),
                            StartMenuControlsButton,
                            StartMenuEntity,
                        ))
                        .with_children(|b| {
                            b.spawn((
                                Text::new("Controls"),
                                text_font(font, 14.0),
                                TextColor(Color::srgb(0.85, 0.85, 0.9)),
                                StartMenuEntity,
                            ));
                        });
                        if has_cancel {
                            btns.spawn((
                                Button,
                                modal_button_node(),
                                BackgroundColor(Color::srgba(0.18, 0.18, 0.28, 0.95)),
                                StartMenuCancelButton,
                                StartMenuEntity,
                            ))
                            .with_children(|b| {
                                b.spawn((
                                    Text::new("Cancel"),
                                    text_font(font, 14.0),
                                    TextColor(Color::srgb(0.85, 0.85, 0.9)),
                                    StartMenuEntity,
                                ));
                            });
                        }
                    });
            });
        });
}

fn handle_start_menu_new_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<StartMenuNewButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut state: ResMut<AstState>,
    mut mode: ResMut<UiMode>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut start_menu: ResMut<StartMenu>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                state.layout_ast = layout::LayoutAst::empty().plus_sink_wall();
                *mode = UiMode::Playground;
                rebuild.0 = true;
                start_menu.showing = false;
                start_menu.has_cancel = false;
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.25, 0.25, 0.35, 0.95);
                color.0 = Color::srgb(1.0, 1.0, 1.0);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.18, 0.18, 0.28, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
        }
    }
}

fn handle_start_menu_load_example_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<StartMenuLoadExampleButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut state: ResMut<AstState>,
    mut mode: ResMut<UiMode>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut start_menu: ResMut<StartMenu>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                state.layout_ast = layout::LayoutAst::empty().plus_sink_example();
                *mode = UiMode::Examples;
                rebuild.0 = true;
                start_menu.showing = false;
                start_menu.has_cancel = false;
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.25, 0.25, 0.35, 0.95);
                color.0 = Color::srgb(1.0, 1.0, 1.0);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.18, 0.18, 0.28, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
        }
    }
}

fn handle_start_menu_controls_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<StartMenuControlsButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut start_menu: ResMut<StartMenu>,
    mut eval: ResMut<EvalState>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                start_menu.showing = false;
                eval.phase = EvalPhase::ControlsModal;
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.25, 0.25, 0.35, 0.95);
                color.0 = Color::srgb(1.0, 1.0, 1.0);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.18, 0.18, 0.28, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
        }
    }
}

fn handle_start_menu_cancel_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<StartMenuCancelButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut start_menu: ResMut<StartMenu>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                start_menu.showing = false;
                start_menu.has_cancel = false;
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.25, 0.25, 0.35, 0.95);
                color.0 = Color::srgb(1.0, 1.0, 1.0);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.18, 0.18, 0.28, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
        }
    }
}

fn handle_hamburger_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<HamburgerButton>),
    >,
    mut start_menu: ResMut<StartMenu>,
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    for (interaction, mut bg) in interaction_q.iter_mut() {
        match *interaction {
            Interaction::Pressed => {
                start_menu.showing = true;
                start_menu.has_cancel = true;
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.2, 0.2, 0.3, 0.95);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.16, 0.16, 0.22, 0.9);
            }
        }
    }
}

fn sync_start_menu_ui(
    mut commands: Commands,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    ui_font: Res<UiFont>,
    menu_entities: Query<Entity, With<StartMenuEntity>>,
    mut hideable: Query<&mut Node, With<HideDuringStartMenu>>,
    mut last_showing: Local<Option<bool>>,
    mut last_hidden: Local<Option<bool>>,
) {
    if *last_showing != Some(start_menu.showing) {
        let was_showing = *last_showing;
        *last_showing = Some(start_menu.showing);

        if !start_menu.showing {
            for e in menu_entities.iter() {
                commands.entity(e).despawn();
            }
        } else if was_showing == Some(false) {
            // Re-open after a Cancel/close — respawn the menu.
            for e in menu_entities.iter() {
                commands.entity(e).despawn();
            }
            spawn_start_menu_ui(&mut commands, &ui_font.0, start_menu.has_cancel);
        }
    }

    let hidden = start_menu.showing || modal_is_open(&eval);
    if *last_hidden == Some(hidden) {
        return;
    }
    *last_hidden = Some(hidden);
    let d = if hidden { Display::None } else { Display::Flex };
    for mut n in hideable.iter_mut() {
        n.display = d;
    }
}

fn handle_modal_ok_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<ModalOkButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut eval: ResMut<EvalState>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                eval.phase = EvalPhase::Idle;
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.25, 0.25, 0.35, 0.95);
                color.0 = Color::srgb(1.0, 1.0, 1.0);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.18, 0.18, 0.28, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
        }
    }
}

fn handle_controls_modal_ok_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<ControlsModalOkButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut eval: ResMut<EvalState>,
    mut start_menu: ResMut<StartMenu>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                eval.phase = EvalPhase::Idle;
                start_menu.showing = true;
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.25, 0.25, 0.35, 0.95);
                color.0 = Color::srgb(1.0, 1.0, 1.0);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.18, 0.18, 0.28, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
        }
    }
}

fn handle_modal_cancel_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<ModalCancelButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut eval: ResMut<EvalState>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                eval.phase = EvalPhase::Idle;
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.25, 0.25, 0.35, 0.95);
                color.0 = Color::srgb(1.0, 1.0, 1.0);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.18, 0.18, 0.28, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
        }
    }
}

fn handle_modal_evaluate_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<ModalEvaluateButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut eval: ResMut<EvalState>,
    state: Res<AstState>,
    input_q: Query<(&ModalVarDeclInput, &TextInput)>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                let mut user_values: std::collections::HashMap<ast::node::Id, String> =
                    std::collections::HashMap::new();
                for (m, input) in input_q.iter() {
                    user_values.insert(m.node_id.clone(), input.value.clone());
                }
                let initial =
                    eval::initial_values(&state.layout_ast.ast, &user_values, &mut eval.rng);
                eval.phase = EvalPhase::Running {
                    steps: vec![initial],
                    current: 0,
                };
            }
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.25, 0.25, 0.35, 0.95);
                color.0 = Color::srgb(1.0, 1.0, 1.0);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.18, 0.18, 0.28, 0.95);
                color.0 = Color::srgb(0.85, 0.85, 0.9);
            }
        }
    }
}

fn sync_eval_step_bar(
    mut commands: Commands,
    eval: Res<EvalState>,
    ui_font: Res<UiFont>,
    bar_q: Query<Entity, With<EvalStepBarEntity>>,
    mut was_running: Local<bool>,
) {
    let running_now = matches!(eval.phase, EvalPhase::Running { .. });
    if running_now == *was_running {
        return;
    }
    for e in bar_q.iter() {
        commands.entity(e).despawn();
    }
    *was_running = running_now;
    if !running_now {
        return;
    }
    // Spawn Prev / Next / Exit, right-aligned bottom.
    spawn_corner_button(
        &mut commands,
        &ui_font.0,
        "Exit Evaluation",
        (ExitEvaluationButton, EvalStepBarEntity),
        Val::Px(12.0),
        Val::Px(12.0),
    );
    spawn_corner_button(
        &mut commands,
        &ui_font.0,
        "Next",
        (NextStepButton, EvalStepBarEntity),
        Val::Px(170.0),
        Val::Px(12.0),
    );
    spawn_corner_button(
        &mut commands,
        &ui_font.0,
        "Prev",
        (PrevStepButton, EvalStepBarEntity),
        Val::Px(240.0),
        Val::Px(12.0),
    );
}

fn handle_eval_step_buttons(
    prev_q: Query<&Interaction, (With<PrevStepButton>, Changed<Interaction>)>,
    next_q: Query<&Interaction, (With<NextStepButton>, Changed<Interaction>)>,
    exit_q: Query<&Interaction, (With<ExitEvaluationButton>, Changed<Interaction>)>,
    mut eval: ResMut<EvalState>,
    state: Res<AstState>,
) {
    if exit_q.iter().any(|i| *i == Interaction::Pressed) {
        eval.phase = EvalPhase::Idle;
        return;
    }
    if prev_q.iter().any(|i| *i == Interaction::Pressed) {
        if let EvalPhase::Running { current, .. } = &mut eval.phase {
            if *current > 0 {
                *current -= 1;
            }
        }
    }
    if next_q.iter().any(|i| *i == Interaction::Pressed) {
        // Split borrow: take a mutable reference to the whole struct, then
        // disjoint-borrow `phase` and `rng` simultaneously.
        let EvalState { phase, rng } = &mut *eval;
        if let EvalPhase::Running { steps, current } = phase {
            if let Some(snapshot) = eval::step_next(&state.layout_ast.ast, &steps[*current], rng) {
                steps.truncate(*current + 1);
                steps.push(snapshot);
                *current += 1;
            }
        }
    }
}

fn update_step_button_visuals(
    eval: Res<EvalState>,
    state: Res<AstState>,
    mut prev_q: Query<
        (&mut BackgroundColor, &Children),
        (With<PrevStepButton>, Without<NextStepButton>),
    >,
    mut next_q: Query<
        (&mut BackgroundColor, &Children),
        (With<NextStepButton>, Without<PrevStepButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
) {
    let (prev_enabled, next_enabled) = match &eval.phase {
        EvalPhase::Running { steps, current } => {
            let next_possible = eval::can_step_next(&state.layout_ast.ast, &steps[*current]);
            (*current > 0, next_possible)
        }
        _ => (false, false),
    };
    let apply = |enabled: bool, bg: &mut BackgroundColor, text_color: &mut TextColor| {
        if enabled {
            bg.0 = Color::srgba(0.16, 0.16, 0.22, 0.9);
            text_color.0 = Color::srgb(0.85, 0.85, 0.9);
        } else {
            bg.0 = Color::srgba(0.10, 0.10, 0.13, 0.9);
            text_color.0 = Color::srgb(0.35, 0.35, 0.4);
        }
    };
    for (mut bg, children) in prev_q.iter_mut() {
        if let Ok(mut c) = text_color_q.get_mut(children[0]) {
            apply(prev_enabled, &mut *bg, &mut *c);
        }
    }
    for (mut bg, children) in next_q.iter_mut() {
        if let Ok(mut c) = text_color_q.get_mut(children[0]) {
            apply(next_enabled, &mut *bg, &mut *c);
        }
    }
}

fn sync_evaluate_button_visibility(
    eval: Res<EvalState>,
    start_menu: Res<StartMenu>,
    mut q: Query<&mut Node, With<EvaluateButton>>,
) {
    let running = matches!(eval.phase, EvalPhase::Running { .. });
    let desired = if start_menu.showing || running || modal_is_open(&eval) {
        Display::None
    } else {
        Display::Flex
    };
    for mut node in q.iter_mut() {
        if node.display != desired {
            node.display = desired;
        }
    }
}

fn sync_value_labels(
    mut commands: Commands,
    eval: Res<EvalState>,
    state: Res<AstState>,
    ui_font: Res<UiFont>,
    mut existing_q: Query<(Entity, &ValueLabel, &mut Text)>,
) {
    let snapshot: Option<&std::collections::HashMap<ast::node::Id, String>> = match &eval.phase {
        EvalPhase::Running { steps, current } => Some(&steps[*current]),
        _ => None,
    };
    let Some(snapshot) = snapshot else {
        for (entity, _, _) in existing_q.iter() {
            commands.entity(entity).despawn();
        }
        return;
    };

    let mut kept: std::collections::HashSet<ast::node::Id> = std::collections::HashSet::new();
    for (entity, label, mut text) in existing_q.iter_mut() {
        if let Some(value) = snapshot.get(&label.node_id) {
            if text.0 != *value {
                text.0 = value.clone();
            }
            kept.insert(label.node_id.clone());
        } else {
            commands.entity(entity).despawn();
        }
    }

    for (id, value) in snapshot.iter() {
        if kept.contains(id) {
            continue;
        }
        let Some(layout_node) = state.layout_ast.layout_nodes.get(id) else {
            continue;
        };
        let world_pos = layout_node.pos * Vec3::new(3.0, 1.5, 3.0);
        commands.spawn((
            Text::new(value.clone()),
            text_font(&ui_font.0, 28.0),
            TextColor(Color::srgb(1.0, 0.95, 0.3)),
            Node {
                position_type: PositionType::Absolute,
                ..default()
            },
            Visibility::Hidden,
            WorldLabel {
                world_pos,
                offset: Vec2::new(60.0, 0.0),
            },
            ValueLabel {
                node_id: id.clone(),
            },
        ));
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
            commands.entity(entity).despawn();
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
    mut materials_grid: ResMut<Assets<grid::GridMaterial>>,
    state: Res<AstState>,
    ui_font: Res<UiFont>,
    mut rebuild: ResMut<NeedsRebuild>,
    query_ast_entities: Query<Entity, With<AstSceneEntity>>,
) {
    if rebuild.0 {
        spawn_ast_nodes(commands, meshes, materials, materials_grid, state, ui_font);
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
    font: &Handle<Font>,
    render_label: render::RenderLabel,
    marker: impl Bundle,
) -> Entity {
    commands
        .spawn((
            Text::new(render_label.text),
            text_font(font, render_label.font_size),
            TextColor(render_label.color),
            Node {
                position_type: PositionType::Absolute,
                ..default()
            },
            Visibility::Hidden,
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
    mut label_q: Query<(&WorldLabel, &mut Node, &mut Visibility, &ComputedNode)>,
) {
    let Ok((camera, cam_gt)) = camera_q.single() else {
        return;
    };

    for (label, mut node, mut vis, computed) in label_q.iter_mut() {
        if let Ok(screen_pos) = camera.world_to_viewport(cam_gt, label.world_pos) {
            let size = computed.size();
            node.left = Val::Px(screen_pos.x - size.x / 2.0 + label.offset.x);
            node.top = Val::Px(screen_pos.y - size.y / 2.0 + label.offset.y);
            *vis = Visibility::Visible;
        } else {
            // Behind camera
            *vis = Visibility::Hidden;
        }
    }
}

fn spawn_selection_display(mut commands: Commands, ui_font: Res<UiFont>) {
    commands.spawn((
        Text::new(""),
        text_font(&ui_font.0, 16.0),
        TextColor(Color::srgb(0.85, 0.85, 0.9)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(34.0),
            right: Val::Px(14.0),
            ..default()
        },
        SelectionDisplay,
        HideDuringStartMenu,
    ));
}

fn spawn_fps_display(mut commands: Commands, ui_font: Res<UiFont>) {
    commands.spawn((
        Text::new("--"),
        text_font(&ui_font.0, 12.0),
        TextColor(Color::srgba(0.5, 0.5, 0.6, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            right: Val::Px(12.0),
            ..default()
        },
        FpsDisplay,
    ));
}

fn update_fps_display(
    diagnostics: Res<DiagnosticsStore>,
    mut text_q: Query<&mut Text, With<FpsDisplay>>,
) {
    let Ok(mut text) = text_q.single_mut() else {
        return;
    };
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed());
    // Current (non-smoothed) frame time in ms — reveals stutter that
    // the smoothed FPS number hides.
    let ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.value());
    text.0 = match (fps, ms) {
        (Some(f), Some(t)) => format!("{f:.0} fps · {t:.1}ms"),
        (Some(f), None) => format!("{f:.0} fps"),
        _ => "-- fps".into(),
    };
}

fn pick_nodes(
    camera_q: Query<(&Camera, &GlobalTransform), With<camera::OrbitCameraTag>>,
    windows: Query<&Window>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut pick: ResMut<PickState>,
    node_q: Query<(&AstNodeEntity, &Transform)>,
    state: Res<AstState>,
    grid_config: Res<grid::GridConfig>,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    ui_interactions: Query<&Interaction, With<Button>>,
) {
    if start_menu.showing {
        pick.hovered_node = None;
        pick.hovered_pos = None;
        pick.press_cursor = None;
        pick.press_over_ui = false;
        return;
    }
    if modal_is_open(&eval) {
        return;
    }
    let Ok((camera, cam_gt)) = camera_q.single() else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        pick.hovered_node = None;
        pick.hovered_pos = None;
        return;
    };

    let Ok(ray) = camera.viewport_to_world(cam_gt, cursor) else {
        pick.hovered_node = None;
        pick.hovered_pos = None;
        return;
    };

    // Cursor sitting over any UI element (button, dropdown, input, checkbox,
    // or the editor panel background) suppresses grid hover and click-to-select.
    let over_ui = ui_interactions
        .iter()
        .any(|i| matches!(*i, Interaction::Hovered | Interaction::Pressed));

    // Ray-sphere test against nodes (radius 0.35).
    let radius = 0.35_f32;
    let mut closest: Option<(ast::node::Id, f32)> = None;
    if !over_ui {
        for (node_ent, transform) in node_q.iter() {
            let center = transform.translation;
            let oc = ray.origin - center;
            let b = oc.dot(*ray.direction);
            let c = oc.dot(oc) - radius * radius;
            let disc = b * b - c;
            if disc >= 0.0 {
                let t = -b - disc.sqrt();
                if t > 0.0 && closest.as_ref().map_or(true, |(_, tc)| t < *tc) {
                    closest = Some((node_ent.node_id.clone(), t));
                }
            }
        }
    }
    pick.hovered_node = closest.as_ref().map(|(id, _)| id.clone());

    // Ray-plane test against y=0 for grid hover. Skip if a node is hovered
    // or the cursor is over a UI element.
    let mut grid_hit: Option<IVec3> = None;
    if !over_ui && closest.is_none() && ray.direction.y.abs() > 1e-4 {
        let t = -ray.origin.y / ray.direction.y;
        if t > 0.0 {
            let hit = ray.origin + *ray.direction * t;
            if Vec2::new(hit.x, hit.z).length() <= grid_config.fade_start {
                let spacing = grid_config.spacing;
                let gx = (hit.x / spacing).round() as i32;
                let gz = (hit.z / spacing).round() as i32;
                grid_hit = Some(IVec3::new(gx, 0, gz));
            }
        }
    }
    pick.hovered_pos = grid_hit;

    const CLICK_MOVE_THRESHOLD: f32 = 5.0;
    if mouse.just_pressed(MouseButton::Left) {
        pick.press_cursor = Some(cursor);
        pick.press_over_ui = over_ui;
    }
    if mouse.just_released(MouseButton::Left) {
        let is_click = pick
            .press_cursor
            .map(|p| (cursor - p).length() < CLICK_MOVE_THRESHOLD)
            .unwrap_or(false);
        let press_over_ui = pick.press_over_ui;
        pick.press_cursor = None;
        pick.press_over_ui = false;
        if is_click && !press_over_ui && !over_ui {
            if let Some((node_id, _)) = closest {
                if let Some(ln) = state.layout_ast.layout_nodes.get(&node_id) {
                    pick.selected_pos = ln.pos.round().as_ivec3();
                }
            } else if let Some(pos) = grid_hit {
                pick.selected_pos = pos;
            }
            // Click into truly empty space (beyond visible grid) leaves
            // the selection unchanged.
        }
    }
}

fn highlight_hovered(
    pick: Res<PickState>,
    state: Res<AstState>,
    node_q: Query<(&AstNodeEntity, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let selected_node = state.layout_ast.node_at(pick.selected_pos);
    for (node_ent, mat_handle) in node_q.iter() {
        let Some(mat) = materials.get_mut(&mat_handle.0) else {
            continue;
        };

        let base = render::emissive_color(mat.base_color);
        let is_hovered = pick.hovered_node.as_ref() == Some(&node_ent.node_id);
        let is_selected = selected_node.as_ref() == Some(&node_ent.node_id);

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
    mut display_q: Query<(&mut Text, &mut TextColor), With<SelectionDisplay>>,
) {
    let Ok((mut text, mut color)) = display_q.single_mut() else {
        return;
    };
    if let Some(id) = state.layout_ast.node_at(pick.selected_pos) {
        if let Some(node) = state.layout_ast.ast.nodes.get(&id) {
            text.0 = format!(
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
            color.0 = Color::WHITE;
        }
    } else {
        text.0 = format!(
            "({}, {}, {})",
            pick.selected_pos.x, pick.selected_pos.y, pick.selected_pos.z
        );
        color.0 = Color::srgb(0.55, 0.55, 0.6);
    }
}

fn update_grid_hover_material(
    pick: Res<PickState>,
    grid_config: Res<grid::GridConfig>,
    mut materials: ResMut<Assets<grid::GridMaterial>>,
) {
    let (pos, active) = match pick.hovered_pos {
        Some(p) => (
            Vec2::new(
                p.x as f32 * grid_config.spacing,
                p.z as f32 * grid_config.spacing,
            ),
            1.0,
        ),
        None => (Vec2::ZERO, 0.0),
    };
    for (_, mat) in materials.iter_mut() {
        mat.hover_pos = pos;
        mat.hover_active = active;
    }
}

fn update_cursor(
    pick: Res<PickState>,
    mut commands: Commands,
    windows: Query<Entity, With<Window>>,
) {
    use bevy::window::CursorIcon;
    use bevy::window::SystemCursorIcon;
    let Ok(entity) = windows.single() else {
        return;
    };
    commands
        .entity(entity)
        .insert(if pick.hovered_node.is_some() {
            CursorIcon::System(SystemCursorIcon::Pointer)
        } else {
            CursorIcon::System(SystemCursorIcon::Default)
        });
}

fn text_input_focus(
    mut input_q: Query<
        (
            &Interaction,
            &mut TextInput,
            &mut BorderColor,
            Option<&ModalEntity>,
        ),
        With<TextInputBox>,
    >,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    eval: Res<EvalState>,
) {
    let clicked_outside = mouse.just_pressed(MouseButton::Left);
    let evaluating = is_evaluating(&eval);

    for (interaction, mut input, mut border, modal_tag) in input_q.iter_mut() {
        // During evaluation only modal inputs may take focus.
        if evaluating && modal_tag.is_none() {
            input.focused = false;
            *border = BorderColor::all(Color::srgb(0.12, 0.12, 0.24));
            continue;
        }
        if *interaction == Interaction::Pressed {
            input.focused = true;
        } else if clicked_outside && *interaction == Interaction::None {
            input.focused = false;
        }

        if keys.just_pressed(KeyCode::Escape) {
            input.focused = false;
        }

        // Visual feedback
        *border = BorderColor::all(if input.focused {
            Color::srgb(0.133, 0.827, 0.933) // cyan when focused
        } else {
            Color::srgb(0.12, 0.12, 0.24)
        });
    }
}

fn text_input_keyboard(
    mut input_q: Query<(&mut TextInput, &Children), With<TextInputBox>>,
    mut text_q: Query<&mut Text, With<TextInputDisplay>>,
    mut key_events: MessageReader<KeyboardInput>,
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

        // Update display text with blinking cursor
        if let Ok(mut text) = text_q.get_mut(children[0]) {
            let (before, after) = input.value.split_at(input.cursor);
            text.0 = if input.focused {
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
                commands.entity(entity).despawn();
            }
            */
            //orbit.auto_rotate = true;
            //rebuild.0 = true;
            //orbit.theta = 0.6;
            //orbit.phi = 1.0;
        }
    }
}

/// Detect a change in `PickState::selected_pos` and start a camera
/// auto-focus tween toward the new position. The first observation
/// (fresh `Local`) does not trigger, so app startup doesn't jump.
fn trigger_camera_focus_on_selection_change(
    pick: Res<PickState>,
    orbit: Res<camera::OrbitCamera>,
    mut tween: ResMut<camera::CameraTween>,
    mut last_pos: Local<Option<IVec3>>,
    start_menu: Res<StartMenu>,
) {
    if start_menu.showing {
        return;
    }
    let current = pick.selected_pos;
    if *last_pos != Some(current) {
        if last_pos.is_some() {
            let world = render::layout_to_world(current.as_vec3());
            tween.focus_on(&orbit, world);
        }
        *last_pos = Some(current);
    }
}

fn handle_arrow_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<AstState>,
    mut pick: ResMut<PickState>,
    mut rebuild: ResMut<NeedsRebuild>,
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    // Direction of each arrow in layout coordinates.
    let delta = if keys.just_pressed(KeyCode::ArrowUp) {
        if ctrl && shift {
            Some(IVec3::new(0, 1, 0))
        } else if !shift {
            Some(IVec3::new(-1, 0, 0))
        } else {
            None
        }
    } else if keys.just_pressed(KeyCode::ArrowDown) {
        if ctrl && shift {
            Some(IVec3::new(0, -1, 0))
        } else if !shift {
            Some(IVec3::new(1, 0, 0))
        } else {
            None
        }
    } else if keys.just_pressed(KeyCode::ArrowLeft) {
        if !shift {
            Some(IVec3::new(0, 0, 1))
        } else {
            None
        }
    } else if keys.just_pressed(KeyCode::ArrowRight) {
        if !shift {
            Some(IVec3::new(0, 0, -1))
        } else {
            None
        }
    } else {
        None
    };

    let Some(delta) = delta else {
        return;
    };

    if ctrl {
        // Move the node under the current selection (if any) and keep
        // the selection anchored to it.
        if let Some(node_id) = state.layout_ast.node_at(pick.selected_pos) {
            state.layout_ast = state.layout_ast.move_node_delta(node_id, delta.as_vec3());
            pick.selected_pos += delta;
            rebuild.0 = true;
        }
    } else {
        // Plain arrow: navigate the selection between grid crossings.
        pick.selected_pos += delta;
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

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_tf)) = camera_q.single() else {
        return;
    };

    let mut closest: Option<(Entity, f32)> = None;

    for (entity, global_tf) in &anchors {
        let Ok(screen_pos) = camera.world_to_viewport(cam_tf, global_tf.translation()) else {
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
        &mut MeshMaterial3d<StandardMaterial>,
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
        if mat.0 != *target_mat {
            mat.0 = target_mat.clone();
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
        gizmos.line(from.translation(), to.translation(), edge.color);
    }
}

fn draw_drag_preview(drag: Res<DragState>, mut gizmos: Gizmos) {
    let Some(ref info) = drag.active else { return };
    let color = if info.target_anchor.is_some() {
        Color::srgb(0.3, 1.0, 0.4) // grün = eingeschnappt
    } else {
        Color::srgb(1.0, 0.9, 0.3) // gelb = dragging
    };
    gizmos.line(info.source_pos, info.current_end, color);
}

fn drag_start_system(
    mouse: Res<ButtonInput<MouseButton>>,
    hovered: Query<(Entity, &GlobalTransform, &EAnchor), (With<AnchorHovered>)>,
    mut drag: ResMut<DragState>,
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    if mouse.just_pressed(MouseButton::Left) {
        if let Ok((entity, tf, anchor)) = hovered.single() {
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
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    let Some(ref mut info) = drag.active else {
        return;
    };

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_tf)) = camera_q.single() else {
        return;
    };

    // Ray durch Cursor
    let Ok(ray) = camera.viewport_to_world(cam_tf, cursor) else {
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
    if let Ok((entity, tf, anchor)) = hovered.single() {
        if entity != info.source_anchor {
            info.target_anchor = Some(entity);
            info.target_anchor_id = Some(anchor.id());
            info.current_end = tf.translation();
        }
    }
}

#[derive(Component, Clone, Copy)]
enum CrosshairPart {
    LineUp,
    LineDown,
    LineLeft,
    LineRight,
    TickUp,
    TickDown,
    TickLeft,
    TickRight,
}

fn spawn_crosshair(mut commands: Commands) {
    let color = Color::srgba(0.55, 0.54, 0.52, 0.22);

    for part in [
        CrosshairPart::LineUp,
        CrosshairPart::LineDown,
        CrosshairPart::LineLeft,
        CrosshairPart::LineRight,
        CrosshairPart::TickUp,
        CrosshairPart::TickDown,
        CrosshairPart::TickLeft,
        CrosshairPart::TickRight,
    ] {
        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                ..default()
            },
            BackgroundColor(color),
            Visibility::Hidden,
            part,
        ));
    }
}

fn update_crosshair(
    pick: Res<PickState>,
    state: Res<AstState>,
    camera_q: Query<(&Camera, &GlobalTransform), With<camera::OrbitCameraTag>>,
    node_q: Query<(&AstNodeEntity, &GlobalTransform)>,
    windows: Query<&Window>,
    mut crosshair_q: Query<(&CrosshairPart, &mut Node, &mut Visibility)>,
) {
    let hide_all = |q: &mut Query<(&CrosshairPart, &mut Node, &mut Visibility)>| {
        for (_, _, mut vis) in q.iter_mut() {
            *vis = Visibility::Hidden;
        }
    };

    let Ok((camera, cam_tf)) = camera_q.single() else {
        hide_all(&mut crosshair_q);
        return;
    };
    let Ok(window) = windows.single() else {
        hide_all(&mut crosshair_q);
        return;
    };

    // Anchor world position: use node transform if a node lives at the
    // selected position (accounts for any drift), otherwise derive from
    // the layout-scaled selected_pos.
    let anchor_world = state
        .layout_ast
        .node_at(pick.selected_pos)
        .and_then(|id| {
            node_q
                .iter()
                .find(|(e, _)| e.node_id == id)
                .map(|(_, tf)| tf.translation())
        })
        .unwrap_or_else(|| render::layout_to_world(pick.selected_pos.as_vec3()));

    let Ok(screen) = camera.world_to_viewport(cam_tf, anchor_world) else {
        hide_all(&mut crosshair_q);
        return;
    };

    // Project a world-space sphere with constant radius onto the screen
    // to get a perspective-correct, distance-aware box size.
    const CROSSHAIR_RADIUS: f32 = 0.5;
    let edge_world = anchor_world + *cam_tf.right() * CROSSHAIR_RADIUS;
    let Ok(edge_screen) = camera.world_to_viewport(cam_tf, edge_world) else {
        hide_all(&mut crosshair_q);
        return;
    };
    let half: f32 = (edge_screen - screen).length();
    let thickness: f32 = 2.0;
    let w = window.width();
    let h = window.height();

    let tick_len = half;
    let tick_half = tick_len * 0.5;

    for (part, mut style, mut vis) in crosshair_q.iter_mut() {
        *vis = Visibility::Visible;
        match *part {
            CrosshairPart::LineUp => {
                style.left = Val::Px(screen.x - thickness * 0.5);
                style.top = Val::Px(0.0);
                style.width = Val::Px(thickness);
                style.height = Val::Px((screen.y - half).max(0.0));
            }
            CrosshairPart::LineDown => {
                style.left = Val::Px(screen.x - thickness * 0.5);
                style.top = Val::Px(screen.y + half);
                style.width = Val::Px(thickness);
                style.height = Val::Px((h - screen.y - half).max(0.0));
            }
            CrosshairPart::LineLeft => {
                style.left = Val::Px(0.0);
                style.top = Val::Px(screen.y - thickness * 0.5);
                style.width = Val::Px((screen.x - half).max(0.0));
                style.height = Val::Px(thickness);
            }
            CrosshairPart::LineRight => {
                style.left = Val::Px(screen.x + half);
                style.top = Val::Px(screen.y - thickness * 0.5);
                style.width = Val::Px((w - screen.x - half).max(0.0));
                style.height = Val::Px(thickness);
            }
            CrosshairPart::TickUp => {
                style.left = Val::Px(screen.x - tick_half);
                style.top = Val::Px(screen.y - half - thickness * 0.5);
                style.width = Val::Px(tick_len);
                style.height = Val::Px(thickness);
            }
            CrosshairPart::TickDown => {
                style.left = Val::Px(screen.x - tick_half);
                style.top = Val::Px(screen.y + half - thickness * 0.5);
                style.width = Val::Px(tick_len);
                style.height = Val::Px(thickness);
            }
            CrosshairPart::TickLeft => {
                style.left = Val::Px(screen.x - half - thickness * 0.5);
                style.top = Val::Px(screen.y - tick_half);
                style.width = Val::Px(thickness);
                style.height = Val::Px(tick_len);
            }
            CrosshairPart::TickRight => {
                style.left = Val::Px(screen.x + half - thickness * 0.5);
                style.top = Val::Px(screen.y - tick_half);
                style.width = Val::Px(thickness);
                style.height = Val::Px(tick_len);
            }
        }
    }
}

fn drag_end_system(
    mouse: Res<ButtonInput<MouseButton>>,
    mut drag: ResMut<DragState>,
    mut commands: Commands,
    mut rebuild: ResMut<NeedsRebuild>,
    mut state: ResMut<AstState>,
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        drag.active = None;
        return;
    }
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
                    title: "AST Visualizer 3D — Bevy + WebGPU".into(),
                    canvas: Some("#bevy-canvas".into()),
                    fit_canvas_to_parent: true,
                    prevent_default_event_handling: true,
                    present_mode: bevy::window::PresentMode::AutoVsync,
                    ..default()
                }),
                ..default()
            }),
            camera::OrbitCameraPlugin,
            FrameTimeDiagnosticsPlugin::default(),
        ))
        .add_plugins(grid::GridPlugin)
        .init_resource::<AstState>()
        .init_resource::<NeedsRebuild>()
        .init_resource::<PickState>()
        .init_resource::<DragState>()
        .init_resource::<UiMode>()
        .init_resource::<EvalState>()
        .init_resource::<StartMenu>()
        .init_resource::<DropdownState>()
        .add_systems(
            Startup,
            (
                load_ui_font,
                setup_scene,
                spawn_walls,
                spawn_ast_nodes,
                spawn_ui,
                spawn_selection_display,
                spawn_node_editor_panel,
                spawn_fps_display,
                spawn_crosshair,
                spawn_start_menu,
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
                        handle_add_node_button,
                        handle_example_buttons,
                        handle_hamburger_button,
                        handle_start_menu_new_button,
                        handle_start_menu_load_example_button,
                        handle_start_menu_controls_button,
                        handle_start_menu_cancel_button,
                        sync_start_menu_ui,
                        update_mode_visibility,
                        pick_nodes,
                    )
                        .chain(),
                    highlight_hovered,
                    update_selection_display,
                    update_grid_hover_material,
                    update_cursor,
                    text_input_focus,
                    text_input_keyboard,
                    handle_arrow_keys,
                    trigger_camera_focus_on_selection_change,
                ),
                (
                    anchor_hover_system,
                    drag_start_system,
                    drag_update_system,
                    drag_end_system,
                    clear_scene,
                    ApplyDeferred,
                    rebuild_scene,
                )
                    .chain(),
            )
                .chain(),
        )
        .add_systems(
            Update,
            (update_world_labels, update_crosshair, update_fps_display),
        )
        .add_systems(
            Update,
            (
                handle_evaluate_button,
                handle_modal_ok_button,
                handle_controls_modal_ok_button,
                handle_modal_cancel_button,
                handle_modal_evaluate_button,
                handle_eval_step_buttons,
                sync_modal_ui,
                sync_eval_step_bar,
                sync_evaluate_button_visibility,
                update_step_button_visuals,
                update_delete_button_visuals,
                sync_value_labels,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                handle_dropdown_click,
                handle_dropdown_option_click,
                handle_value_enable_checkbox,
                handle_node_editor_text_input,
                sync_node_editor_ui,
            )
                .chain(),
        )
        .run();
}
