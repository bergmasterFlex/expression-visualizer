mod ast;
mod camera;
mod colors;
mod edge;
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
    /// Root LayoutAst. Contains exactly one Program node; the user-visible
    /// scene lives in `layout_ast.sub_layouts[program_id]`. Reads/writes go
    /// through `program_ast()` / `program_ast_mut()`.
    layout_ast: layout::LayoutAst,
    program_id: ast::node::Id,
    function_declarations:
        std::collections::HashMap<ast::FunctionDeclarationId, ast::FunctionDeclaration>,
}

impl AstState {
    fn program_ast(&self) -> &layout::LayoutAst {
        self.layout_ast.sub_layouts.get(&self.program_id).unwrap()
    }

    fn program_ast_mut(&mut self) -> &mut layout::LayoutAst {
        self.layout_ast
            .sub_layouts
            .get_mut(&self.program_id)
            .unwrap()
    }
}

impl Default for AstState {
    fn default() -> Self {
        let (layout_ast, program_id) = layout::LayoutAst::empty_with_program();
        Self {
            layout_ast,
            program_id,
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
                (
                    ast::FunctionDeclarationId(3),
                    ast::FunctionDeclaration {
                        name: "*(-1)".to_string(),
                        inputs: vec![FunctionParameterDeclaration {
                            name: "number".to_string(),
                            r#type: eval::EType::Int(None),
                        }],
                        output_type: eval::EType::SumType(vec![eval::EType::Int(None)]),
                    },
                ),
                (
                    ast::FunctionDeclarationId(4),
                    ast::FunctionDeclaration {
                        name: "substr".to_string(),
                        inputs: vec![
                            FunctionParameterDeclaration {
                                name: "str".to_string(),
                                r#type: eval::EType::String(None),
                            },
                            FunctionParameterDeclaration {
                                name: "begin".to_string(),
                                r#type: eval::EType::Int(None),
                            },
                            FunctionParameterDeclaration {
                                name: "length".to_string(),
                                r#type: eval::EType::Int(None),
                            },
                        ],
                        output_type: eval::EType::SumType(vec![eval::EType::String(None)]),
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
    pub source_anchor_id: ast::AnchorId,
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

/// Marker for a per-AST grid mesh (one per Program-Ast / Pattern sub-AST).
/// Carries just enough info to map a raycast hit back to a local grid cell
/// in the owning LayoutAst.
#[derive(Component, Clone)]
struct AstGridEntity {
    /// Owner path from the root LayoutAst down to this AST's LayoutAst.
    /// Empty = root (the outer container that owns the Program node).
    /// Length 1 with the Program's id = the Program-Ast itself.
    context: Vec<ast::node::Id>,
    /// Accumulated grid-space offset of this AST's origin from the root.
    origin_offset: Vec3,
    /// Local grid-space bounds this grid currently spans (inclusive).
    min: IVec3,
    max: IVec3,
}

/// Marker for the Program-Ast's front wall at Z=0. VarDecls attach to this
/// wall; the wall itself is a pickable vertical plane. Bounds are in the
/// Program-Ast's local grid X coordinates (inclusive), with half-a-cell of
/// padding on each side baked into the mesh but *not* into the pick region.
#[derive(Component, Clone)]
struct ProgramWallEntity {
    min_x: i32,
    max_x: i32,
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
#[derive(Component, Clone, PartialEq, Eq)]
enum EAstActionButton {
    AddConstDeclButton,
    AddVarDeclButton,
    AddTypeCastButton,
    AddFunctionCallButton,
    AddMatchButton,
    AddPatternButton,
    SelectAstButton,
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
struct StartMenuCancelButton;
#[derive(Component)]
struct StartMenuControlsButton;

/// Marker for UI entities that should be hidden while the start menu is open.
#[derive(Component)]
struct HideDuringStartMenu;

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
    /// AST grid cell under the cursor (ray hit on an `AstGridEntity`), if
    /// any. Includes both the local cell coords and the owning AST's
    /// context path so click routing knows where to write.
    hovered_grid: Option<HoveredGrid>,
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
    /// Path of container ids from the Program-Ast down to the currently
    /// active editing context. Empty = Program-Ast (top-level user code);
    /// a non-empty path names Pattern ids in outer-to-inner order. Add/
    /// delete/move handlers route through `current_ast(_mut)` so the
    /// change lands in the right sub-ast. Stays empty until Step 5 wires
    /// the "Select Ast" button.
    context_path: Vec<ast::node::Id>,
    /// Owner path from the Program-Ast to the LayoutAst that holds the
    /// currently selected node. Empty = selection lives in the Program-
    /// Ast. Independent of `context_path`: selection may land in a sub-
    /// AST while add/delete/move still target the Program (until Step 5
    /// adds "Select Ast"). `selected_pos` is expressed in the coordinate
    /// system of the LayoutAst identified by `selected_context`.
    selected_context: Vec<ast::node::Id>,
}

/// Populated when the cursor is over a per-AST grid mesh.
#[derive(Clone)]
struct HoveredGrid {
    /// Local grid coords inside the hovered AST.
    local_pos: IVec3,
    /// Owner path (from Program-Ast) to the hovered AST.
    context: Vec<ast::node::Id>,
    /// Entity of the `AstGridEntity` mesh that was hit — used so the hover
    /// shader wash flips only on the AST grid actually under the cursor.
    entity: Entity,
    /// World XZ of the hovered cell's center. Fed to the grid shader's
    /// `hover_pos` uniform.
    world_center: Vec2,
}

impl Default for PickState {
    fn default() -> Self {
        Self {
            selected_pos: IVec3::ZERO,
            hovered_node: None,
            hovered_grid: None,
            press_cursor: None,
            press_over_ui: false,
            context_path: Vec::new(),
            selected_context: Vec::new(),
        }
    }
}

impl PickState {
    fn current_ast<'a>(&self, state: &'a AstState) -> &'a layout::LayoutAst {
        let mut ast = state.program_ast();
        for id in &self.context_path {
            ast = ast.sub_layouts.get(id).unwrap();
        }
        ast
    }

    fn current_ast_mut<'a>(&self, state: &'a mut AstState) -> &'a mut layout::LayoutAst {
        let mut ast = state.program_ast_mut();
        for id in &self.context_path {
            ast = ast.sub_layouts.get_mut(id).unwrap();
        }
        ast
    }

    /// LayoutAst that owns the currently selected node. Follows
    /// `selected_context` — distinct from `current_ast`, which follows
    /// `context_path`.
    fn selected_ast<'a>(&self, state: &'a AstState) -> &'a layout::LayoutAst {
        state.program_ast().resolve_context(&self.selected_context)
    }

    fn selected_ast_mut<'a>(&self, state: &'a mut AstState) -> &'a mut layout::LayoutAst {
        let mut ast = state.program_ast_mut();
        for id in &self.selected_context {
            ast = ast.sub_layouts.get_mut(id).unwrap();
        }
        ast
    }
}

/// UI text showing the selected node's info.
#[derive(Component)]
struct SelectionDisplay;

/// Marker for the FPS counter text in the top-right corner.
#[derive(Component)]
struct FpsDisplay;

/// UI text showing the current editing context (context_path) as breadcrumb.
#[derive(Component)]
struct BreadcrumbDisplay;

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
    Pattern,
    Other,
}

fn variant_kind(node: Option<&ast::node::ENode>) -> NodeVariantKind {
    match node {
        None => NodeVariantKind::None,
        Some(ast::node::ENode::TypeIntroduction { .. }) => NodeVariantKind::TypeIntroduction,
        Some(ast::node::ENode::TypeElimination { .. }) => NodeVariantKind::TypeElimination,
        Some(ast::node::ENode::VarDecl { .. }) => NodeVariantKind::VarDecl,
        Some(ast::node::ENode::FunctionCall { .. }) => NodeVariantKind::FunctionCall,
        Some(ast::node::ENode::Pattern { .. }) => NodeVariantKind::Pattern,
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

use layout::value_of_etype;

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

/// Spawn the AST node meshes.
fn spawn_ast_nodes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut materials_grid: ResMut<Assets<grid::GridMaterial>>,
    mut materials_edge: ResMut<Assets<edge::EdgeMaterial>>,
    mut images: ResMut<Assets<Image>>,
    edge_labels: Option<Res<edge::EdgeLabelTextures>>,
    state: Res<AstState>,
    ui_font: Res<UiFont>,
    pick: Res<PickState>,
) {
    let mut node_entites = std::collections::HashMap::<ast::node::Id, Entity>::new();
    let mut anchor_entities = std::collections::HashMap::<ast::AnchorId, Entity>::new();
    let mut anchor_world_positions = std::collections::HashMap::<ast::AnchorId, Vec3>::new();
    for walked in state.layout_ast.walk_all() {
        let layout_node = walked.layout_node;
        let node_id = &layout_node.node_id;
        let node = walked.layout_ast.ast.nodes.get(node_id).unwrap();
        if let ast::node::ENode::MatchGrid { width, depth } = node {
            let world_pos = (layout_node.pos + walked.extra_offset) * Vec3::new(3.0, 3.0, 3.0);
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
            layout_node,
            walked.layout_ast,
            &state.function_declarations,
            walked.extra_offset,
            walked.sink_scale,
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
                let render::RenderAnchor {
                    normal,
                    hovered,
                    type_markers,
                } = render_anchor;

                for marker in type_markers {
                    let render::RenderTypeMarker {
                        rect,
                        label,
                        value_line,
                        value_label,
                    } = marker;
                    commands.spawn((
                        Mesh3d(meshes.add(rect.mesh)),
                        MeshMaterial3d(materials.add(rect.material)),
                        rect.transform,
                        AstSceneEntity,
                    ));
                    spawn_world_label(&mut commands, &ui_font.0, label, AstSceneEntity);
                    if let Some(line) = value_line {
                        commands.spawn((
                            Mesh3d(meshes.add(line.mesh)),
                            MeshMaterial3d(materials.add(line.material)),
                            line.transform,
                            AstSceneEntity,
                        ));
                    }
                    if let Some(vlabel) = value_label {
                        spawn_world_label(&mut commands, &ui_font.0, vlabel, AstSceneEntity);
                    }
                }

                let render_anchor = render::RenderAnchor {
                    normal,
                    hovered,
                    type_markers: vec![],
                };

                let layout_anchor = walked.layout_ast.layout_anchor(anchor_id.clone());
                let anchor_world = render_anchor.normal.transform.translation;
                let spawned = commands
                    .spawn((
                        Mesh3d(meshes.add(render_anchor.normal.mesh.clone())),
                        MeshMaterial3d(materials.add(render_anchor.normal.material.clone())),
                        render_anchor.normal.transform,
                        match layout_anchor.anchor {
                            ast::EAnchor::Input { .. } => EAnchor::Input {
                                id: anchor_id.clone(),
                                render_objects: render_anchor,
                            },
                            ast::EAnchor::Output => EAnchor::Output {
                                id: anchor_id.clone(),
                                render_objects: render_anchor,
                            },
                        },
                        AstSceneEntity,
                    ))
                    .id();
                anchor_entities.insert(anchor_id.clone(), spawned);
                anchor_world_positions.insert(anchor_id, anchor_world);
            });

        node_entites.insert(node_id.clone(), node_entity.clone());

        render_node.labels.into_iter().for_each(|l| {
            spawn_world_label(&mut commands, &ui_font.0, l, AstSceneEntity);
        });
    }

    if let Some(edge_labels) = edge_labels.as_ref() {
        // Cache value+type marquee textures per (kind, value) within this
        // spawn pass. Rebuilds are rare (only on scene rebuild) and edge
        // counts are ~O(30), so a local map is cheaper than a resource.
        let mut value_marquee_cache: std::collections::HashMap<
            (edge::LeafKind, String),
            Handle<Image>,
        > = std::collections::HashMap::new();
        let value_font =
            ab_glyph::FontRef::try_from_slice(edge::FONT_BYTES).expect("bundled font is valid");

        for e in state.program_ast().edges() {
            let src_id = &e.from_anchor.anchor_id;
            let tgt_id = &e.to_anchor.anchor_id;

            let Some(&from_world) = anchor_world_positions.get(src_id) else {
                continue;
            };
            let Some(&to_world) = anchor_world_positions.get(tgt_id) else {
                continue;
            };

            let src_type = state
                .program_ast()
                .anchor_type(src_id, &state.function_declarations)
                .unwrap_or(eval::EType::Undefined);
            let source_leaves = render::ordered_supported_leaves(&src_type);
            let n_src = source_leaves.len();
            if n_src == 0 {
                continue;
            }

            let tgt_type = state
                .program_ast()
                .anchor_type(tgt_id, &state.function_declarations);
            let target_leaves = tgt_type
                .as_ref()
                .map(|t| render::ordered_supported_leaves(t))
                .unwrap_or_default();
            let n_tgt = target_leaves.len();

            // AST-level literal on the source anchor (VarDecl / TypeIntroduction
            // / Pattern / TypeElimination). `None` for FunctionCall outputs and
            // structural nodes. When present, the sole rendered leaf swaps to
            // the thin "value line" style.
            let src_ast_value = state.program_ast().anchor_ast_value(src_id);

            let curve = edge::EdgeCurve::from_endpoints(from_world, to_world);

            let edge_root = commands
                .spawn((
                    Edge {
                        from_anchor: *anchor_entities.get(src_id).unwrap(),
                        to_anchor: *anchor_entities.get(tgt_id).unwrap(),
                        source_anchor_id: src_id.clone(),
                    },
                    Transform::IDENTITY,
                    Visibility::Inherited,
                    AstSceneEntity,
                ))
                .id();

            for (k, leaf) in source_leaves.iter().enumerate() {
                let y_src = (k as f32 - (n_src as f32 - 1.0) / 2.0) * render::TYPE_MARKER_Y_STEP;
                let leaf_kind = edge::leaf_kind_of(leaf);
                let y_tgt = if let Some(idx) = target_leaves
                    .iter()
                    .position(|l| edge::leaf_kind_of(l) == leaf_kind)
                {
                    (idx as f32 - (n_tgt as f32 - 1.0) / 2.0) * render::TYPE_MARKER_Y_STEP
                } else {
                    0.0
                };
                let Some(kind) = edge::leaf_kind_of(leaf) else {
                    continue;
                };
                let (height, label, line_mode) = if let Some(value) = src_ast_value.as_deref() {
                    let text = format!("  {}  {}  ", value, kind.type_name());
                    let handle = value_marquee_cache
                        .entry((kind, text.clone()))
                        .or_insert_with(|| {
                            edge::rasterize_marquee_text(&value_font, &text, &mut images)
                        })
                        .clone();
                    (edge::RIBBON_LINE_HEIGHT, handle, 1.0)
                } else {
                    (
                        edge::RIBBON_HEIGHT,
                        edge_labels.by_kind.get(&kind).cloned().unwrap(),
                        0.0,
                    )
                };
                let mesh = edge::build_ribbon_mesh(
                    &curve,
                    from_world.y + y_src,
                    to_world.y + y_tgt,
                    height,
                );
                commands.spawn((
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(materials_edge.add(edge::EdgeMaterial {
                        band_color: render::type_marker_color(leaf).to_linear(),
                        letter_color: LinearRgba::WHITE,
                        scroll_speed: 1.5,
                        tile_length: 3.0,
                        time: 0.0,
                        line_mode,
                        line_half_thickness: edge::RIBBON_LINE_HALF_THICKNESS_UV,
                        _pad0: 0.0,
                        _pad1: 0.0,
                        _pad2: 0.0,
                        label,
                    })),
                    ChildOf(edge_root),
                    AstSceneEntity,
                ));
            }
        }
    }

    for walked_ast in state.program_ast().walk_all_asts() {
        let active_selection = if walked_ast.context == pick.selected_context {
            Some(pick.selected_pos)
        } else {
            None
        };
        let Some(bounds) = walked_ast.layout_ast.ast_grid_bounds(active_selection) else {
            continue;
        };
        let width_cells = (bounds.max.x - bounds.min.x + 1) as f32;
        let depth_cells = (bounds.max.z - bounds.min.z + 1) as f32;
        let size_x = width_cells * 3.0;
        let size_z = depth_cells * 3.0;
        let center_local = Vec3::new(
            (bounds.min.x + bounds.max.x) as f32 * 0.5,
            0.0,
            (bounds.min.z + bounds.max.z) as f32 * 0.5,
        );
        let world_center = (center_local + walked_ast.extra_offset) * Vec3::new(3.0, 3.0, 3.0);
        commands.spawn((
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
            Transform::from_translation(world_center),
            AstGridEntity {
                context: walked_ast.context.clone(),
                origin_offset: walked_ast.extra_offset,
                min: bounds.min,
                max: bounds.max,
            },
            AstSceneEntity,
        ));

        // Program-Ast: front wall at Z=0. Grid-wide plus half a cell of
        // padding on each side, so VarDecls at the outermost grid X still
        // have wall to attach to.
        if walked_ast.context.is_empty() {
            let wall_width_cells = width_cells + 1.0;
            let wall_size_x = wall_width_cells * 3.0;
            let wall_center_x =
                (bounds.min.x + bounds.max.x) as f32 * 0.5 * 3.0 + walked_ast.extra_offset.x * 3.0;
            commands.spawn((
                Mesh3d(meshes.add(Cuboid::new(wall_size_x, 6.0, 0.05))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgba(0.5, 0.5, 0.5, 0.5),
                    alpha_mode: AlphaMode::Blend,
                    cull_mode: None,
                    ..default()
                })),
                Transform::from_xyz(wall_center_x, 0.0, 0.0),
                ProgramWallEntity {
                    min_x: bounds.min.x,
                    max_x: bounds.max.x,
                },
                AstSceneEntity,
            ));
        }
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

    let mut y_offset = 60.0;
    spawn_ui_button(
        &mut commands,
        &ui_font.0,
        "Delete Node",
        DeleteNodeButton,
        Vec2::new(12.0, y_offset),
        Display::Flex,
    );
    for (label, action) in [
        ("Add ConstDecl", EAstActionButton::AddConstDeclButton),
        ("Add VarDecl", EAstActionButton::AddVarDeclButton),
        ("Add FunctionCall", EAstActionButton::AddFunctionCallButton),
        ("Add Match", EAstActionButton::AddMatchButton),
        ("Add Pattern", EAstActionButton::AddPatternButton),
        ("Add TypeCast", EAstActionButton::AddTypeCastButton),
        ("Select Ast", EAstActionButton::SelectAstButton),
    ] {
        y_offset += 36.0;
        spawn_ui_button(
            &mut commands,
            &ui_font.0,
            label,
            action,
            Vec2::new(12.0, y_offset),
            Display::Flex,
        );
    }

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
        if let Some(selected_node_id) = pick.selected_ast(&state).node_at(pick.selected_pos) {
            let is_sink_wall = matches!(
                pick.selected_ast(&state).ast.nodes.get(&selected_node_id),
                Some(ast::node::ENode::SinkWall { .. })
            );
            if !is_sink_wall {
                let updated = pick.selected_ast(&state).minus_node(&selected_node_id);
                *pick.selected_ast_mut(&mut state) = updated;
                rebuild.0 = true;
            }
        }
    }
}

fn update_add_pattern_button_visuals(
    pick: Res<PickState>,
    state: Res<AstState>,
    mut button_q: Query<(
        &Interaction,
        &mut BackgroundColor,
        &Children,
        &EAstActionButton,
    )>,
    mut text_color_q: Query<&mut TextColor>,
) {
    let context_mismatch = pick.selected_context != pick.context_path;
    let enabled = !context_mismatch
        && matches!(
            pick.current_ast(&state)
                .node_at(pick.selected_pos)
                .and_then(|id| pick.current_ast(&state).ast.nodes.get(&id).cloned()),
            Some(ast::node::ENode::Pattern { .. })
        );
    for (interaction, mut bg, children, action) in button_q.iter_mut() {
        if *action != EAstActionButton::AddPatternButton {
            continue;
        }
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

fn update_add_generic_button_visuals(
    pick: Res<PickState>,
    state: Res<AstState>,
    mut button_q: Query<(
        &Interaction,
        &mut BackgroundColor,
        &Children,
        &EAstActionButton,
    )>,
    mut text_color_q: Query<&mut TextColor>,
) {
    let pos_free = pick
        .current_ast(&state)
        .node_at(pick.selected_pos)
        .is_none();
    for (interaction, mut bg, children, action) in button_q.iter_mut() {
        if *action == EAstActionButton::AddPatternButton
            || *action == EAstActionButton::SelectAstButton
        {
            continue;
        }
        let Ok(mut text_color) = text_color_q.get_mut(children[0]) else {
            continue;
        };
        let wall_slot =
            pick.context_path.is_empty() && pick.selected_pos.y == 0 && pick.selected_pos.z == 0;
        let vardecl_locked = *action == EAstActionButton::AddVarDeclButton && !wall_slot;
        let wall_slot_locked = wall_slot
            && matches!(
                *action,
                EAstActionButton::AddConstDeclButton
                    | EAstActionButton::AddFunctionCallButton
                    | EAstActionButton::AddTypeCastButton
                    | EAstActionButton::AddMatchButton
            );
        let context_mismatch = pick.selected_context != pick.context_path;
        let enabled = pos_free && !vardecl_locked && !wall_slot_locked && !context_mismatch;
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

fn update_select_ast_button_visuals(
    pick: Res<PickState>,
    state: Res<AstState>,
    mut button_q: Query<(
        &Interaction,
        &mut BackgroundColor,
        &Children,
        &EAstActionButton,
    )>,
    mut text_color_q: Query<&mut TextColor>,
) {
    let enabled = pick
        .selected_ast(&state)
        .node_at(pick.selected_pos)
        .is_some();
    for (interaction, mut bg, children, action) in button_q.iter_mut() {
        if *action != EAstActionButton::SelectAstButton {
            continue;
        }
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

fn update_delete_button_visuals(
    pick: Res<PickState>,
    state: Res<AstState>,
    mut button_q: Query<(&Interaction, &mut BackgroundColor, &Children), With<DeleteNodeButton>>,
    mut text_color_q: Query<&mut TextColor>,
) {
    let enabled = match pick.selected_ast(&state).node_at(pick.selected_pos) {
        Some(id) => !matches!(
            pick.selected_ast(&state).ast.nodes.get(&id),
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
                if pick.selected_context != pick.context_path {
                    continue;
                }
                let new_pos = pick.selected_pos.as_vec3();
                let target_occupied = pick
                    .current_ast(&state)
                    .node_at(pick.selected_pos)
                    .is_some();
                if target_occupied && *action != EAstActionButton::AddPatternButton {
                    continue;
                }
                // Wall row (Program-Ast, Y=0, Z=0) is VarDecl-only. Refuse
                // every other creation action defensively.
                let is_wall_slot = pick.context_path.is_empty()
                    && pick.selected_pos.y == 0
                    && pick.selected_pos.z == 0;
                if is_wall_slot
                    && !matches!(
                        *action,
                        EAstActionButton::AddVarDeclButton
                            | EAstActionButton::SelectAstButton
                            | EAstActionButton::AddPatternButton
                    )
                {
                    continue;
                }
                let new_layout = match action {
                    EAstActionButton::SelectAstButton => continue,
                    EAstActionButton::AddConstDeclButton => pick
                        .current_ast(&state)
                        .plus_type_introduction(ast::node::EType::Int { value: None }, new_pos),
                    EAstActionButton::AddVarDeclButton => {
                        // VarDecls live on the Program-Ast's front wall at
                        // (x, 0, 0) — refuse anything else defensively; the
                        // enable-check normally greys the button out first.
                        if !pick.context_path.is_empty()
                            || pick.selected_pos.y != 0
                            || pick.selected_pos.z != 0
                        {
                            continue;
                        }
                        pick.current_ast(&state).plus_var_decl(new_pos)
                    }
                    EAstActionButton::AddFunctionCallButton => {
                        pick.current_ast(&state).plus_function_call(
                            state
                                .function_declarations
                                .iter()
                                .find(|(_, d)| d.name == "+")
                                .map(|(id, decl)| (id.clone(), decl))
                                .unwrap(),
                            new_pos,
                        )
                    }
                    EAstActionButton::AddTypeCastButton => pick
                        .current_ast(&state)
                        .plus_type_elimination(ast::node::EType::Int { value: None }, new_pos),
                    EAstActionButton::AddMatchButton => {
                        pick.current_ast(&state).plus_match_new(new_pos)
                    }
                    EAstActionButton::AddPatternButton => {
                        match pick.current_ast(&state).node_at(pick.selected_pos) {
                            Some(id)
                                if matches!(
                                    pick.current_ast(&state).ast.nodes.get(&id),
                                    Some(ast::node::ENode::Pattern { .. })
                                ) =>
                            {
                                let updated = pick.current_ast(&state).plus_pattern_above(&id);
                                pick.selected_pos.y += 1;
                                updated
                            }
                            _ => continue,
                        }
                    }
                };
                *pick.current_ast_mut(&mut state) = new_layout;
                state.layout_ast = state.layout_ast.settle_matches();
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

fn handle_select_ast_button(
    interaction_q: Query<
        (&Interaction, &EAstActionButton),
        (Changed<Interaction>, With<EAstActionButton>),
    >,
    state: Res<AstState>,
    mut pick: ResMut<PickState>,
    mut rebuild: ResMut<NeedsRebuild>,
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    for (interaction, action) in interaction_q.iter() {
        if *action != EAstActionButton::SelectAstButton {
            continue;
        }
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(selected_node_id) = pick.selected_ast(&state).node_at(pick.selected_pos) else {
            continue;
        };
        let is_pattern = matches!(
            pick.selected_ast(&state).ast.nodes.get(&selected_node_id),
            Some(ast::node::ENode::Pattern { .. })
        );
        let mut new_context = pick.selected_context.clone();
        if is_pattern {
            new_context.push(selected_node_id);
        }
        pick.context_path = new_context;
        rebuild.0 = true;
    }
}

// ── Node editor UI ──────────────────────────────────────────

fn spawn_node_editor_panel(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(82.0),
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
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    ui_font: Res<UiFont>,
    mut panel_q: Query<(Entity, &mut Node), With<NodeEditorPanel>>,
    editor_children_q: Query<Entity, With<NodeEditorEntity>>,
    mut cache: Local<NodeEditorFingerprint>,
) {
    let selected_ast = pick.selected_ast(&state);
    let node_id = selected_ast.node_at(pick.selected_pos);
    let node = node_id
        .as_ref()
        .and_then(|id| selected_ast.ast.nodes.get(id));
    let variant = variant_kind(node);
    let type_choice = node.and_then(|n| match n {
        ast::node::ENode::TypeIntroduction { r#type, .. }
        | ast::node::ENode::TypeElimination { r#type, .. }
        | ast::node::ENode::Pattern { r#type, .. } => type_choice_of(r#type),
        _ => None,
    });
    let typeelim_has_value = match node {
        Some(ast::node::ENode::TypeElimination { r#type, .. })
        | Some(ast::node::ENode::Pattern { r#type, .. }) => value_of_etype(r#type).is_some(),
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
            | NodeVariantKind::Pattern
    );
    let visible = editable && !start_menu.showing && !is_evaluating(&eval);

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
            ast::node::ENode::Pattern { r#type, .. } => {
                spawn_editor_label(panel, font, "Pattern");
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
                let node = state
                    .layout_ast
                    .find_node_ast_mut(&option.node_id)
                    .and_then(|a| a.ast.nodes.get_mut(&option.node_id));
                match node {
                    Some(ast::node::ENode::TypeIntroduction { r#type, .. })
                    | Some(ast::node::ENode::TypeElimination { r#type, .. })
                    | Some(ast::node::ENode::VarDecl { r#type, .. })
                    | Some(ast::node::ENode::Pattern { r#type, .. }) => {
                        let value = value_of_etype(r#type);
                        *r#type = make_etype(*new_choice, value);
                        rebuild.0 = true;
                    }
                    _ => {}
                }
            }
            DropdownChoice::BoolValue(v) => {
                let node = state
                    .layout_ast
                    .find_node_ast_mut(&option.node_id)
                    .and_then(|a| a.ast.nodes.get_mut(&option.node_id));
                match node {
                    Some(ast::node::ENode::TypeIntroduction { r#type, .. })
                    | Some(ast::node::ENode::TypeElimination { r#type, .. })
                    | Some(ast::node::ENode::VarDecl { r#type, .. })
                    | Some(ast::node::ENode::Pattern { r#type, .. }) => {
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
                    if let Some(owning_ast) = state.layout_ast.find_node_ast_mut(&option.node_id) {
                        let new_layout = owning_ast.with_function_call_replaced(
                            &option.node_id,
                            (new_fn_id.clone(), &new_decl),
                        );
                        *owning_ast = new_layout;
                        rebuild.0 = true;
                    }
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
        let r#type = match state
            .layout_ast
            .find_node_ast_mut(&cb.node_id)
            .and_then(|a| a.ast.nodes.get_mut(&cb.node_id))
        {
            Some(ast::node::ENode::TypeElimination { r#type, .. })
            | Some(ast::node::ENode::Pattern { r#type, .. }) => r#type,
            _ => continue,
        };
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

fn handle_node_editor_text_input(
    input_q: Query<(&NodeEditorTextInput, &TextInput), Changed<TextInput>>,
    mut state: ResMut<AstState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    for (editor_input, input) in input_q.iter() {
        let Some(node) = state
            .layout_ast
            .find_node_ast_mut(&editor_input.node_id)
            .and_then(|a| a.ast.nodes.get_mut(&editor_input.node_id))
        else {
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
                    | ast::node::ENode::VarDecl { r#type, .. }
                    | ast::node::ENode::Pattern { r#type, .. } => r#type,
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
                let ast = &state.program_ast().ast;
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
            let ast = &state.program_ast().ast;
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
        ("Shift + Up/Down", "Move selection vertically"),
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
    mut rebuild: ResMut<NeedsRebuild>,
    mut start_menu: ResMut<StartMenu>,
    mut pick: ResMut<PickState>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                *state.program_ast_mut() = layout::LayoutAst::empty().plus_sink_wall();
                pick.context_path.clear();
                pick.selected_context.clear();
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
                    eval::initial_values(&state.program_ast().ast, &user_values, &mut eval.rng);
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
            if let Some(snapshot) = eval::step_next(&state.program_ast().ast, &steps[*current], rng)
            {
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
            let next_possible = eval::can_step_next(&state.program_ast().ast, &steps[*current]);
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
        let Some(layout_node) = state.program_ast().layout_nodes.get(id) else {
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
    commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<StandardMaterial>>,
    materials_grid: ResMut<Assets<grid::GridMaterial>>,
    materials_edge: ResMut<Assets<edge::EdgeMaterial>>,
    images: ResMut<Assets<Image>>,
    edge_labels: Option<Res<edge::EdgeLabelTextures>>,
    state: Res<AstState>,
    ui_font: Res<UiFont>,
    pick: Res<PickState>,
    mut rebuild: ResMut<NeedsRebuild>,
    _query_ast_entities: Query<Entity, With<AstSceneEntity>>,
) {
    if rebuild.0 {
        spawn_ast_nodes(
            commands,
            meshes,
            materials,
            materials_grid,
            materials_edge,
            images,
            edge_labels,
            state,
            ui_font,
            pick,
        );
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

fn spawn_breadcrumb_display(mut commands: Commands, ui_font: Res<UiFont>) {
    commands.spawn((
        Text::new("Program"),
        text_font(&ui_font.0, 12.0),
        TextColor(Color::srgba(0.55, 0.55, 0.65, 0.9)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(58.0),
            right: Val::Px(14.0),
            max_width: Val::Px(260.0),
            overflow: Overflow::clip(),
            ..default()
        },
        BreadcrumbDisplay,
        HideDuringStartMenu,
    ));
}

fn update_breadcrumb_display(
    pick: Res<PickState>,
    mut text_q: Query<&mut Text, With<BreadcrumbDisplay>>,
) {
    let Ok(mut text) = text_q.single_mut() else {
        return;
    };
    let mut parts = vec!["Program".to_string()];
    for id in &pick.context_path {
        parts.push(format!("Pattern({})", id.0));
    }
    text.0 = parts.join(" > ");
}

fn pick_nodes(
    camera_q: Query<(&Camera, &GlobalTransform), With<camera::OrbitCameraTag>>,
    windows: Query<&Window>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut pick: ResMut<PickState>,
    node_q: Query<(&AstNodeEntity, &Transform)>,
    grid_q: Query<(Entity, &AstGridEntity)>,
    wall_q: Query<(Entity, &ProgramWallEntity)>,
    state: Res<AstState>,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    ui_interactions: Query<&Interaction, With<Button>>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    if start_menu.showing {
        pick.hovered_node = None;
        pick.hovered_grid = None;
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
        pick.hovered_grid = None;
        return;
    };

    let Ok(ray) = camera.viewport_to_world(cam_gt, cursor) else {
        pick.hovered_node = None;
        pick.hovered_grid = None;
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

    // Ray-plane test against each spawned AST grid. Each AST grid sits at
    // its own Y (Program-Ast Y=0; Pattern sub-AST at Pattern's world Y) and
    // spans a rectangle in local grid coords. Pick the closest rect hit.
    const LAYOUT_SCALE_F: f32 = 3.0;
    let mut grid_hit: Option<HoveredGrid> = None;
    let mut best_t = f32::INFINITY;
    if !over_ui && closest.is_none() {
        for (entity, ag) in grid_q.iter() {
            let plane_y = ag.origin_offset.y * LAYOUT_SCALE_F;
            let denom = ray.direction.y;
            if denom.abs() < 1e-4 {
                continue;
            }
            let t = (plane_y - ray.origin.y) / denom;
            if t <= 0.0 || t >= best_t {
                continue;
            }
            let hit = ray.origin + *ray.direction * t;
            let world_offset_x = ag.origin_offset.x * LAYOUT_SCALE_F;
            let world_offset_z = ag.origin_offset.z * LAYOUT_SCALE_F;
            let local_x = ((hit.x - world_offset_x) / LAYOUT_SCALE_F).round() as i32;
            let local_z = ((hit.z - world_offset_z) / LAYOUT_SCALE_F).round() as i32;
            if local_x < ag.min.x || local_x > ag.max.x {
                continue;
            }
            if local_z < ag.min.z || local_z > ag.max.z {
                continue;
            }
            let world_cx = (local_x as f32 + ag.origin_offset.x) * LAYOUT_SCALE_F;
            let world_cz = (local_z as f32 + ag.origin_offset.z) * LAYOUT_SCALE_F;
            best_t = t;
            grid_hit = Some(HoveredGrid {
                local_pos: IVec3::new(local_x, 0, local_z),
                context: ag.context.clone(),
                entity,
                world_center: Vec2::new(world_cx, world_cz),
            });
        }
        // Program-Ast front wall: vertical plane at Z=0. Half-a-cell of
        // pick padding on each side matches the wall mesh's own padding.
        for (entity, wall) in wall_q.iter() {
            let denom = ray.direction.z;
            if denom.abs() < 1e-4 {
                continue;
            }
            let t = -ray.origin.z / denom;
            if t <= 0.0 || t >= best_t {
                continue;
            }
            let hit = ray.origin + *ray.direction * t;
            let min_wx = (wall.min_x as f32 - 0.5) * LAYOUT_SCALE_F;
            let max_wx = (wall.max_x as f32 + 0.5) * LAYOUT_SCALE_F;
            if hit.x < min_wx || hit.x > max_wx {
                continue;
            }
            if hit.y < -3.0 || hit.y > 3.0 {
                continue;
            }
            let local_x = (hit.x / LAYOUT_SCALE_F).round() as i32;
            let world_cx = local_x as f32 * LAYOUT_SCALE_F;
            best_t = t;
            grid_hit = Some(HoveredGrid {
                local_pos: IVec3::new(local_x, 0, 0),
                context: Vec::new(),
                entity,
                world_center: Vec2::new(world_cx, 0.0),
            });
        }
    }
    pick.hovered_grid = grid_hit.clone();

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
                if let Some(ctx) = state.program_ast().context_of_node(&node_id) {
                    let owning_ast = state.program_ast().resolve_context(&ctx);
                    if let Some(ln) = owning_ast.layout_nodes.get(&node_id) {
                        let new_pos = ln.pos.round().as_ivec3();
                        if pick.selected_pos != new_pos || pick.selected_context != ctx {
                            pick.selected_pos = new_pos;
                            pick.selected_context = ctx;
                            rebuild.0 = true;
                        }
                    }
                }
            } else if let Some(hit) = grid_hit {
                if pick.selected_pos != hit.local_pos || pick.selected_context != hit.context {
                    pick.selected_pos = hit.local_pos;
                    pick.selected_context = hit.context;
                    rebuild.0 = true;
                }
            }
            // Click into truly empty space (no AST grid hit) leaves the
            // selection unchanged.
        }
    }
}

fn highlight_hovered(
    pick: Res<PickState>,
    state: Res<AstState>,
    node_q: Query<(&AstNodeEntity, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let selected_node = pick.selected_ast(&state).node_at(pick.selected_pos);
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
    let selected_ast = pick.selected_ast(&state);
    if let Some(id) = selected_ast.node_at(pick.selected_pos) {
        if let Some(node) = selected_ast.ast.nodes.get(&id) {
            text.0 = format!(
                "{} : {}",
                node.label(&state.function_declarations),
                match eval::eval_type(
                    &node,
                    &selected_ast.ast,
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
    grid_q: Query<(Entity, &MeshMaterial3d<grid::GridMaterial>), With<AstGridEntity>>,
    mut materials: ResMut<Assets<grid::GridMaterial>>,
) {
    let hit_entity = pick.hovered_grid.as_ref().map(|h| h.entity);
    let hit_center = pick
        .hovered_grid
        .as_ref()
        .map(|h| h.world_center)
        .unwrap_or(Vec2::ZERO);
    for (entity, mat_handle) in grid_q.iter() {
        let Some(mat) = materials.get_mut(&mat_handle.0) else {
            continue;
        };
        if Some(entity) == hit_entity {
            mat.hover_pos = hit_center;
            mat.hover_active = 1.0;
        } else {
            mat.hover_active = 0.0;
        }
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
    state: Res<AstState>,
    orbit: Res<camera::OrbitCamera>,
    mut tween: ResMut<camera::CameraTween>,
    mut last_selection: Local<Option<(IVec3, Vec<ast::node::Id>)>>,
    start_menu: Res<StartMenu>,
) {
    if start_menu.showing {
        return;
    }
    let current = (pick.selected_pos, pick.selected_context.clone());
    if last_selection.as_ref() != Some(&current) {
        if last_selection.is_some() {
            let offset = state.program_ast().context_offset(&pick.selected_context);
            let world = render::layout_to_world(pick.selected_pos.as_vec3() + offset);
            tween.focus_on(&orbit, world);
        }
        *last_selection = Some(current);
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
        if shift {
            Some(IVec3::new(0, 1, 0))
        } else {
            Some(IVec3::new(-1, 0, 0))
        }
    } else if keys.just_pressed(KeyCode::ArrowDown) {
        if shift {
            Some(IVec3::new(0, -1, 0))
        } else {
            Some(IVec3::new(1, 0, 0))
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
        // the selection anchored to it — the effective position may differ
        // from `selected + delta` when the move jumped over a match.
        // Route through selected_ast (not current_ast) so a Sub-Ast node
        // can be moved without first switching context_path via SelectAst.
        if let Some(node_id) = pick.selected_ast(&state).node_at(pick.selected_pos) {
            let (new_layout, effective_pos) = pick
                .selected_ast(&state)
                .move_node_delta(node_id, delta.as_vec3());
            *pick.selected_ast_mut(&mut state) = new_layout;
            state.layout_ast = state.layout_ast.settle_matches();
            pick.selected_pos = effective_pos;
            rebuild.0 = true;
        }
    } else {
        // Plain arrow: navigate the selection between grid crossings.
        pick.selected_pos += delta;
        rebuild.0 = true;
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
    grid_config: Res<grid::GridConfig>,
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
    // the layout-scaled selected_pos. When selection lives in a sub-AST,
    // add the accumulated owner offset — selected_pos is local.
    let anchor_world = pick
        .selected_ast(&state)
        .node_at(pick.selected_pos)
        .and_then(|id| {
            node_q
                .iter()
                .find(|(e, _)| e.node_id == id)
                .map(|(_, tf)| tf.translation())
        })
        .unwrap_or_else(|| {
            let offset = state.program_ast().context_offset(&pick.selected_context);
            render::layout_to_world(pick.selected_pos.as_vec3() + offset)
        });

    let Ok(screen) = camera.world_to_viewport(cam_tf, anchor_world) else {
        hide_all(&mut crosshair_q);
        return;
    };

    // Project a world-space offset of half a cell onto the screen so the
    // crosshair frames the selected cell (perspective-correct, distance-aware).
    let crosshair_radius = grid_config.spacing * 0.5;
    let edge_world = anchor_world + *cam_tf.right() * crosshair_radius;
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
                let updated = state
                    .program_ast()
                    .plus_edge(info.source_anchor_id, target_id);
                *state.program_ast_mut() = updated;
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
        .add_plugins(edge::EdgePlugin)
        .init_resource::<AstState>()
        .init_resource::<NeedsRebuild>()
        .init_resource::<PickState>()
        .init_resource::<DragState>()
        .init_resource::<EvalState>()
        .init_resource::<StartMenu>()
        .init_resource::<DropdownState>()
        .add_systems(
            Startup,
            (
                load_ui_font,
                setup_scene,
                spawn_ast_nodes,
                spawn_ui,
                spawn_selection_display,
                spawn_node_editor_panel,
                spawn_fps_display,
                spawn_breadcrumb_display,
                spawn_crosshair,
                spawn_start_menu,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                (
                    (anchor_hover_visual_system, draw_drag_preview).chain(),
                    animate_nodes,
                    (
                        handle_delete_node_button,
                        handle_add_node_button,
                        handle_select_ast_button,
                        handle_hamburger_button,
                        handle_start_menu_new_button,
                        handle_start_menu_controls_button,
                        handle_start_menu_cancel_button,
                        sync_start_menu_ui,
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
            (
                update_world_labels,
                update_crosshair,
                update_fps_display,
                update_breadcrumb_display,
            ),
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
                update_add_pattern_button_visuals,
                update_add_generic_button_visuals,
                update_select_ast_button_visuals,
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
