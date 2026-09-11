mod camera;
mod colors;
mod common;
mod edge;
mod eval;
mod grid;
mod infer;
mod layout;
mod mesh;
mod model;
mod render;

use bevy::core_pipeline::oit::OrderIndependentTransparencySettings;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::{input::keyboard::KeyboardInput, math::VectorSpace, prelude::*};

// ── Resources ───────────────────────────────────────────────

#[derive(Resource)]
struct GraphState {
    /// Root LayoutGraph. Contains exactly one Root node; the user-visible
    /// scene lives in `layout_graph.sub_layouts[root_id]`. Reads/writes go
    /// through `root_graph()` / `root_graph_mut()`.
    layout_graph: layout::LayoutGraph,
    root_id: model::node::Id,
    /// Shared id domains for the whole LayoutGraph tree. Threaded through every
    /// `plus_*` builder so node and anchor ids stay globally unique across the
    /// root, the program sub-layout, and all pattern sub-layouts.
    node_id_domain: common::IdDomain<model::node::Id>,
    anchor_id_domain: common::IdDomain<model::anchor::Id>,
    function_declarations: std::collections::HashMap<
        model::function_declaration::FunctionDeclarationId,
        model::function_declaration::FunctionDeclaration,
    >,
}

impl GraphState {
    fn root_graph(&self) -> &layout::LayoutGraph {
        self.layout_graph.sub_layouts.get(&self.root_id).unwrap()
    }

    fn root_graph_mut(&mut self) -> &mut layout::LayoutGraph {
        self.layout_graph
            .sub_layouts
            .get_mut(&self.root_id)
            .unwrap()
    }

    /// Recompute every node's cell shape from the current types and wiring,
    /// then re-settle the layout.
    ///
    /// Shapes decide footprints and footprints decide displacement, so the two
    /// always run in this order. Anchor heights depend on edges as well as on
    /// declared types — a constraint-less input takes the height of whatever is
    /// wired into it — which is why connecting, disconnecting and deleting all
    /// have to come through here, not just moving and adding.
    fn resettle(&mut self) {
        let flat = self.root_graph().flattened_graph();
        self.layout_graph = self
            .layout_graph
            .with_shapes(&flat, &self.function_declarations)
            .settle_footprints();
    }
}

impl Default for GraphState {
    fn default() -> Self {
        let (layout_graph, root_id, node_id_domain, anchor_id_domain) =
            layout::LayoutGraph::empty_with_root();
        let mut state = Self {
            layout_graph,
            root_id,
            node_id_domain,
            anchor_id_domain,
            function_declarations: model::function_declaration::catalogue(),
        };
        // The initial scene already has a Sink; give it a real shape rather
        // than the placeholder, so the first grid and caret resolve correctly.
        state.resettle();
        state
    }
}

#[derive(Component)]
pub struct AnchorHovered;

#[derive(Component)]
pub enum EAnchor {
    Input { id: model::anchor::Id },
    Output { id: model::anchor::Id },
}

impl EAnchor {
    pub fn id(&self) -> model::anchor::Id {
        match self {
            EAnchor::Input { id } | EAnchor::Output { id } => id.clone(),
        }
    }
}

#[derive(Component)]
pub struct Edge {
    pub from_anchor: Entity,
    pub to_anchor: Entity,
    pub source_anchor_id: model::anchor::Id,
}

/// In-flight drag-to-connect state.
///
/// Deliberately holds no `Entity`: `clear_scene` despawns and respawns every
/// `SceneEntity` on each rebuild, so an entity captured at drag start is
/// stale the moment anything sets `NeedsRebuild` mid-drag. Anchor identity is
/// tracked by `AnchorId`, which survives rebuilds.
pub struct DragInfo {
    pub source_anchor_id: model::anchor::Id,
    /// `true` if the drag started on an `EAnchor::Output`. Lets the target
    /// check reject same-kind pairs and lets drag-end store the edge in the
    /// canonical output → input direction without an graph lookup.
    pub source_is_output: bool,
    pub source_pos: Vec3,
    pub current_end: Vec3,
    pub target_anchor_id: Option<model::anchor::Id>,
}

#[derive(Resource, Default)]
pub struct DragState {
    pub active: Option<DragInfo>,
}

/// Marker for graph node mesh entities (so we can despawn them on rebuild).
#[derive(Component)]
struct NodeEntity {
    node_id: model::node::Id,
}

/// Marker for a per-graph grid mesh (one per root graph / Pattern sub-graph).
/// Carries just enough info to map a raycast hit back to a local grid cell
/// in the owning LayoutGraph.
#[derive(Component, Clone)]
struct ScopeGridEntity {
    /// Owner path from the root graph down to this graph's LayoutGraph.
    /// Empty = the root graph itself; each further element names a Pattern.
    context: Vec<model::node::Id>,
    /// Accumulated grid-space offset of this graph's origin from the root.
    origin_offset: Vec3,
    /// Local grid-space bounds this grid currently spans (inclusive).
    min: IVec3,
    max: IVec3,
}

/// Flag resource that signals the scene needs rebuilding.
#[derive(Resource, Default)]
struct NeedsRebuild(bool);

/// Editor mode, vim style. NORMAL navigates the caret through the volume the
/// graph already occupies; INSERT freezes the caret and turns `Space`,
/// `Return` and `Shift+Return` into the inserts that make room inside it,
/// while every other key writes a node kind's name into `InsertPrompt`.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
enum EditorMode {
    #[default]
    Normal,
    Insert,
}

/// What has been typed in INSERT so far, and which suggestion it stands on.
///
/// Deliberately **not** a `TextInput`: a focused text field makes
/// `keyboard_captured` true, and that is what gives `Space`, `Return` and
/// `Escape` their INSERT meanings. A field here would turn `Space` into a
/// blank and `Escape` into a mere unfocus — the three behaviours that have to
/// survive. `selected` indexes the *candidate* list — not the window of it
/// that fits on screen, which is derived from `selected` rather than the other
/// way round — and is re-clamped on every rebuild rather than trusted.
#[derive(Resource, Default)]
struct InsertPrompt {
    text: String,
    selected: usize,
}

impl InsertPrompt {
    fn clear(&mut self) {
        self.text.clear();
        self.selected = 0;
    }
}

/// The node kinds a user action can create. `ENode` has more variants, but
/// `Root`, `Sink` and `BranchSource` only ever come into being as part of
/// something else, so they are not offered.
///
/// Two of them carry what makes them themselves, because the prompt offers one
/// row per possible answer rather than one row per kind: a `FunctionCall`
/// names the function it will call, a `Constant` the value it will hold.
///
/// The constant carries a `TypeChoice` and the raw literal rather than a
/// finished `model::r#type::EType` for two reasons. That type has no
/// `PartialEq`, which `Suggestion`'s fingerprint needs — and deriving one
/// would be worse than the missing derive: an `EType` is a type *plus* an
/// optional literal, so `==` on it answers neither "same type"
/// (`Int{Some("1")} != Int{Some("01")}`) nor "same value" (`Int{None}` twice).
/// The pair is also what the node editor already passes around, as
/// `make_etype(type_choice_of(…), value)`.
///
/// All of that costs `Copy` — neither `FunctionDeclarationId` nor `String`
/// derives it — so the kind travels by reference from here on.
#[derive(Clone, PartialEq, Eq)]
enum AddKind {
    Constant(TypeChoice, Option<String>),
    Source,
    FunctionCall(model::function_declaration::FunctionDeclarationId),
    Match,
    Pattern,
    TypeCast,
}

/// The kinds that name themselves, in list order, each under its `ENode`
/// variant's name — the vocabulary the thesis uses, not a UI phrase like
/// "Add Match".
///
/// Two are missing because they have no name of their own. A function call is
/// listed once per declared function, under that function's name; a constant
/// is whatever literal was typed. A constant with no value would in any case
/// be a node that cannot be evaluated — `eval_value_for_type` refuses it — and
/// that the editor cannot empty again, so the prompt no longer offers one.
///
/// The kinds stand ahead of the functions, and that is what keeps one
/// keystroke enough for them: the letters are not unique — `m` also reaches
/// `mod`, `max` and `min` — but the highlight starts on the first committable
/// row, so `m` still settles on `Match`.
const NODE_KINDS: [(AddKind, &str); 4] = [
    (AddKind::Source, "Source"),
    (AddKind::Match, "Match"),
    (AddKind::Pattern, "Pattern"),
    (AddKind::TypeCast, "TypeCast"),
];

/// The three values that are written as a word instead of as a shape. They are
/// ordinary prefix-filtered rows, not literals read off the text: `true` is
/// typed letter by letter the way `TypeCast` is, and nothing about `t`, `r`,
/// `u`, `e` announces a value the way a digit or a quote does.
///
/// `none` carries no value because it *has* none — `EType::None` has no
/// payload, and `make_etype` would drop one anyway.
const LITERAL_KEYWORDS: [(&str, TypeChoice, Option<&str>); 3] = [
    ("true", TypeChoice::Bool, Some("true")),
    ("false", TypeChoice::Bool, Some("false")),
    ("none", TypeChoice::None, None),
];

/// One row of the INSERT prompt: what it would build, what it is called, and
/// the muted right column that tells `charAt` from `concat` before the node
/// exists.
///
/// It owns its strings instead of borrowing the catalogue on purpose: the list
/// is built from a `&GraphState` and the picked kind then goes to
/// `insert_node_kind(&mut state, …)`, which a borrowed declaration in here
/// would keep from compiling.
#[derive(Clone, PartialEq, Eq)]
struct Suggestion {
    kind: AddKind,
    /// Typed to reach the row, and shown in its left column.
    label: String,
    /// The parameter types, empty for a kind that takes none.
    detail: String,
    allowed: bool,
}

/// Marker for any spawned scene entity (cleaned on rebuild).
#[derive(Component)]
struct SceneEntity;

//Buttons
#[derive(Component)]
struct DeleteNodeButton;
#[derive(Component)]
struct HamburgerButton;
/// The one button left of the six former "Add …" ones: it only enters INSERT,
/// where the kind is typed rather than picked.
#[derive(Component)]
struct AddNodeButton;

/// Root of the INSERT-mode prompt, which stands in the "Add" button's slot.
#[derive(Component)]
struct InsertPromptPanel;

/// Marker on the two rows the prompt respawns on a change — the text row and
/// the suggestion list. Only those two carry it: `despawn` takes their
/// children with them, and an entity despawned twice warns.
#[derive(Component)]
struct InsertPromptEntity;

/// A clickable suggestion row, so the mouse path the "Add" button opens does
/// not dead-end at a keyboard-only list.
#[derive(Component)]
struct InsertPromptOption(AddKind);

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
/// Selection is a **global** grid address (always some), not a node. A node is
/// considered selected iff its layout position, lifted into global
/// coordinates, rounds to `selected_pos`.
///
/// There is deliberately no editing-context field. The caret address alone
/// decides what editing acts on: `GraphState::scope_of_caret` resolves it to the
/// innermost scope whose volume contains it. A caret inside a Match branch
/// volume therefore always refers to that branch, never to the enclosing
/// parent.
#[derive(Resource)]
struct PickState {
    /// Currently selected grid cell, as a global address. Never negative,
    /// and never outside the graph volume — see `LayoutGraph::clamp_to_volume`.
    selected_pos: IVec3,
    /// Node under the cursor (ray-sphere hit), if any.
    hovered_node: Option<model::node::Id>,
    /// graph grid cell under the cursor (ray hit on an `ScopeGridEntity`), if any.
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
}

/// Populated when the cursor is over a per-graph grid mesh.
#[derive(Clone)]
struct HoveredGrid {
    /// Global grid address of the hovered cell.
    global_pos: IVec3,
    /// Entity of the `ScopeGridEntity` mesh that was hit — used so the hover
    /// shader wash flips only on the graph grid actually under the cursor.
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
        }
    }
}

/// Scope the caret currently addresses: the owner path of the innermost scope
/// containing it, plus the caret expressed in that scope's local coordinates.
struct CaretScope {
    path: Vec<model::node::Id>,
    local: IVec3,
}

impl GraphState {
    /// Resolve the caret to its owning scope. `None` when the caret sits
    /// outside every scope volume — editing is then simply unavailable.
    fn scope_of_caret(&self, pick: &PickState) -> Option<CaretScope> {
        self.root_graph()
            .scope_at(pick.selected_pos)
            .map(|(path, local)| CaretScope { path, local })
    }

    /// The LayoutGraph the caret addresses.
    fn caret_graph(&self, pick: &PickState) -> Option<(&layout::LayoutGraph, IVec3)> {
        let scope = self.scope_of_caret(pick)?;
        Some((self.root_graph().resolve_context(&scope.path), scope.local))
    }

    /// Mutable counterpart to `caret_graph`. The path is resolved first so the
    /// immutable and mutable borrows never overlap.
    fn caret_graph_mut(&mut self, pick: &PickState) -> Option<(&mut layout::LayoutGraph, IVec3)> {
        let scope = self.scope_of_caret(pick)?;
        let graph = self.root_graph_mut().resolve_context_mut(&scope.path)?;
        Some((graph, scope.local))
    }
}

/// UI text showing the selected node's info.
#[derive(Component)]
struct SelectionDisplay;

/// Marker for the FPS counter text in the top-right corner.
#[derive(Component)]
struct FpsDisplay;

/// UI text showing the scope the caret currently addresses, as a breadcrumb.
#[derive(Component)]
struct BreadcrumbDisplay;

/// UI text showing the current `EditorMode`, in the bottom-right corner.
#[derive(Component)]
struct ModeDisplay;

/// Marker for the INSERT-mode caret faces, which blink instead of standing
/// still like the NORMAL-mode outline.
#[derive(Component)]
struct CaretBlink;

/// Half-period of the caret blink. 530 ms is what text carets have used
/// forever: calm enough not to nag, quick enough to read as "here".
const CARET_BLINK_SECONDS: f32 = 0.53;

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
    SourcePrompt {
        /// Stable node_id order; values mirror what the user has typed so far.
        inputs: Vec<(model::node::Id, String)>,
    },
    Running {
        /// Full step history; `current` indexes the snapshot on screen. Each
        /// `Next` computes one more `eval_next_step`, `Prev` rewinds.
        states: Vec<eval::State>,
        current: usize,
        /// Source values from the prompt modal, needed to keep stepping.
        user_source_values: std::collections::HashMap<model::node::Id, eval::EValue>,
    },
}

#[derive(Resource)]
struct EvalState {
    phase: EvalPhase,
}

impl Default for EvalState {
    fn default() -> Self {
        Self {
            phase: EvalPhase::Idle,
        }
    }
}

fn is_evaluating(eval: &EvalState) -> bool {
    !matches!(eval.phase, EvalPhase::Idle)
}

fn modal_is_open(eval: &EvalState) -> bool {
    matches!(
        eval.phase,
        EvalPhase::ErrorModal(_) | EvalPhase::ControlsModal | EvalPhase::SourcePrompt { .. }
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

/// Marker on a TextInputBox inside the Source modal so we can collect
/// typed values per Source when the user confirms.
#[derive(Component)]
struct ModalSourceInput {
    node_id: model::node::Id,
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
    node_id: model::node::Id,
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
    SourceName,
    Value,
}

#[derive(Component)]
struct NodeEditorTextInput {
    node_id: model::node::Id,
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
    None,
}

const TYPE_CHOICES: [TypeChoice; 5] = [
    TypeChoice::String,
    TypeChoice::Char,
    TypeChoice::Bool,
    TypeChoice::Int,
    TypeChoice::None,
];

#[derive(Clone, PartialEq, Eq)]
enum DropdownChoice {
    Type(TypeChoice),
    Function(model::function_declaration::FunctionDeclarationId),
    BoolValue(bool),
}

#[derive(Component)]
struct Dropdown {
    node_id: model::node::Id,
    kind: DropdownKind,
}

#[derive(Component)]
struct DropdownOption {
    node_id: model::node::Id,
    kind: DropdownKind,
    choice: DropdownChoice,
}

#[derive(Component)]
struct ValueEnableCheckbox {
    node_id: model::node::Id,
}

/// At most one dropdown is open at a time; `open` identifies which one by
/// `(node_id, kind)` — stable across panel rebuilds.
#[derive(Resource, Default)]
struct DropdownState {
    open: Option<(model::node::Id, DropdownKind)>,
}

#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum NodeVariantKind {
    #[default]
    None,
    Constant,
    TypeCast,
    Source,
    FunctionCall,
    Pattern,
    Other,
}

fn variant_kind(node: Option<&model::node::ENode>) -> NodeVariantKind {
    match node {
        None => NodeVariantKind::None,
        Some(model::node::ENode::Constant { .. }) => NodeVariantKind::Constant,
        Some(model::node::ENode::TypeCast { .. }) => NodeVariantKind::TypeCast,
        Some(model::node::ENode::Source { .. }) => NodeVariantKind::Source,
        Some(model::node::ENode::FunctionCall { .. }) => NodeVariantKind::FunctionCall,
        Some(model::node::ENode::Pattern { .. }) => NodeVariantKind::Pattern,
        Some(_) => NodeVariantKind::Other,
    }
}

fn type_choice_of(t: &model::r#type::EType) -> TypeChoice {
    match t {
        model::r#type::EType::Bool { .. } => TypeChoice::Bool,
        model::r#type::EType::Int { .. } => TypeChoice::Int,
        model::r#type::EType::String { .. } => TypeChoice::String,
        model::r#type::EType::Char { .. } => TypeChoice::Char,
        model::r#type::EType::None { .. } => TypeChoice::None,
    }
}

fn type_choice_label(t: TypeChoice) -> &'static str {
    match t {
        TypeChoice::String => "String",
        TypeChoice::Char => "Char",
        TypeChoice::Bool => "Bool",
        TypeChoice::Int => "Integer",
        TypeChoice::None => "None",
    }
}

use layout::value_of_etype;

fn make_etype(choice: TypeChoice, value: Option<String>) -> model::r#type::EType {
    match choice {
        TypeChoice::Bool => model::r#type::EType::Bool { value },
        TypeChoice::Int => model::r#type::EType::Int { value },
        TypeChoice::String => model::r#type::EType::String { value },
        TypeChoice::Char => model::r#type::EType::Char { value },
        TypeChoice::None => model::r#type::EType::None {},
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
        // Render through an HDR texture so colours brighter than white survive
        // the main pass instead of clamping at 1.0. The INSERT caret is painted
        // past white on purpose, to land on #FFFFFF once the tonemapper has had
        // its say.
        bevy::render::view::Hdr,
        // The bound mode is the default, so the camera starts orthographic. The
        // mode switch replaces this component; nothing else touches it.
        camera::bound_projection(camera::DEFAULT_CELL_PIXELS, camera::RESET_RADIUS, 0.0),
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

/// Spawn the graph node meshes.
fn spawn_graph_nodes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut materials_grid: ResMut<Assets<grid::GridMaterial>>,
    mut materials_edge: ResMut<Assets<edge::EdgeMaterial>>,
    mut images: ResMut<Assets<Image>>,
    edge_labels: Option<Res<edge::EdgeLabelTextures>>,
    state: Res<GraphState>,
    ui_font: Res<UiFont>,
    pick: Res<PickState>,
    editor_mode: Res<EditorMode>,
) {
    let mut node_entites = std::collections::HashMap::<model::node::Id, Entity>::new();
    let mut anchor_entities = std::collections::HashMap::<model::anchor::Id, Entity>::new();
    let mut anchor_world_positions = std::collections::HashMap::<model::anchor::Id, Vec3>::new();
    // Type inference resolves edges, and every edge (pattern branches included)
    // lives in the program-level edge table — so flatten once here instead of
    // per anchor, and hand the same view to the renderer and the edge pass.
    let flat_graph = state.root_graph().flattened_graph();
    // Rasterising text needs a font synchronously — see `edge::FONT_BYTES` for
    // why that one bypasses the asset server — and both passes below want the
    // same one, so it is built once here.
    let glyph_font =
        ab_glyph::FontRef::try_from_slice(edge::FONT_BYTES).expect("bundled font is valid");
    // Cache body-face textures per (text, cells, colour) within this spawn
    // pass. Local rather than a resource on purpose: a resource would hold
    // strong handles past the rebuild, so renaming a source would leave one
    // dead texture behind per keystroke. Here the only handle lives in the
    // material, which `clear_scene` drops, and the image goes with it.
    let mut face_tex_cache: std::collections::HashMap<(String, u32, [u8; 4]), Handle<Image>> =
        std::collections::HashMap::new();
    for walked in state.layout_graph.walk_all() {
        let layout_node = walked.layout_node;
        let node_id = &layout_node.node_id;
        let node = walked.layout_graph.graph.nodes.get(node_id).unwrap();
        let render_node = render::layoutnode_to_rendernode(
            layout_node,
            walked.layout_graph,
            &flat_graph,
            &state.function_declarations,
            walked.extra_offset,
        );
        // A node whose body is a band has no mesh of its own; the entity is
        // still spawned so picking and selection keep working.
        let node_entity = match render_node.node {
            Some(obj) => commands
                .spawn((
                    Mesh3d(meshes.add(obj.mesh)),
                    MeshMaterial3d(materials.add(obj.material)),
                    obj.transform,
                    NodeEntity {
                        node_id: node_id.clone(),
                    },
                    SceneEntity,
                ))
                .id(),
            None => commands
                .spawn((
                    Transform::from_translation(render::cell_center_world(
                        layout_node.pos + walked.extra_offset,
                    )),
                    NodeEntity {
                        node_id: node_id.clone(),
                    },
                    SceneEntity,
                ))
                .id(),
        };

        // Body bands (source nodes): the same marquee ribbon the edges use, so
        // a source reads as the start of its own wire.
        if let Some(edge_labels) = edge_labels.as_ref() {
            for band in render_node.bands {
                commands.spawn((
                    Mesh3d(meshes.add(band.mesh)),
                    MeshMaterial3d(materials_edge.add(edge::EdgeMaterial {
                        band_color: band.color.to_linear(),
                        letter_color: LinearRgba::WHITE,
                        scroll_speed: render::CELL * 0.5,
                        tile_length: render::CELL,
                        time: 0.0,
                        line_mode: 0.0,
                        line_half_thickness: edge::RIBBON_LINE_HALF_THICKNESS_UV,
                        _pad0: 0.0,
                        _pad1: 0.0,
                        _pad2: 0.0,
                        label: edge_labels.by_kind[&band.kind].clone(),
                    })),
                    Transform::IDENTITY,
                    Visibility::Inherited,
                    SceneEntity,
                ));
            }
        }

        // Text printed onto a body face: the name a Source carries on its top
        // face rather than beside itself.
        for face in render_node.text_faces {
            let key = (
                face.text.clone(),
                face.cells,
                face.background.to_srgba().to_u8_array(),
            );
            let texture = face_tex_cache
                .entry(key)
                .or_insert_with(|| {
                    edge::rasterize_face_text(
                        &glyph_font,
                        &face.text,
                        face.cells,
                        face.background,
                        &mut images,
                    )
                })
                .clone();
            commands.spawn((
                Mesh3d(meshes.add(face.mesh)),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color_texture: Some(texture),
                    ..face.material
                })),
                face.transform,
                SceneEntity,
            ));
        }

        // Markers the node owns directly rather than through an anchor: a
        // Pattern is drawn as the band of the type its arm matches.
        spawn_type_markers(
            &mut commands,
            &mut meshes,
            &mut materials,
            &ui_font.0,
            render_node.markers,
        );

        render_node
            .anchors
            .into_iter()
            .for_each(|(anchor_id, render_anchor)| {
                let render::RenderAnchor {
                    pick_center,
                    type_markers,
                    plain_body,
                } = render_anchor;

                spawn_type_markers(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &ui_font.0,
                    type_markers,
                );

                // Neutral cuboid for anchors without type markers.
                if let Some(body) = plain_body {
                    commands.spawn((
                        Mesh3d(meshes.add(body.mesh)),
                        MeshMaterial3d(materials.add(body.material)),
                        body.transform,
                        SceneEntity,
                    ));
                }

                // The anchor itself is now mesh-less: screen-space hover picking
                // only needs its GlobalTransform, positioned at the cuboid centre.
                let layout_anchor = walked.layout_graph.layout_anchor(anchor_id.clone());
                let spawned = commands
                    .spawn((
                        Transform::from_translation(pick_center),
                        match layout_anchor.anchor {
                            model::anchor::EAnchor::Input { .. } => EAnchor::Input {
                                id: anchor_id.clone(),
                            },
                            model::anchor::EAnchor::Output => EAnchor::Output {
                                id: anchor_id.clone(),
                            },
                        },
                        SceneEntity,
                    ))
                    .id();
                anchor_entities.insert(anchor_id.clone(), spawned);
                anchor_world_positions.insert(anchor_id, pick_center);
            });

        node_entites.insert(node_id.clone(), node_entity.clone());

        render_node.labels.into_iter().for_each(|l| {
            spawn_world_label(&mut commands, &ui_font.0, l, SceneEntity);
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

        for e in state.root_graph().edges() {
            let src_id = &e.from_anchor.anchor_id;
            let tgt_id = &e.to_anchor.anchor_id;

            let Some(&from_world) = anchor_world_positions.get(src_id) else {
                continue;
            };
            let Some(&to_world) = anchor_world_positions.get(tgt_id) else {
                continue;
            };

            let src_type = infer::anchor_type(&flat_graph, src_id, &state.function_declarations)
                .unwrap_or(infer::EType::Pending);
            let source_leaves = render::ordered_supported_leaves(&src_type);
            if source_leaves.is_empty() {
                continue;
            }

            // A target that constrains nothing renders as tall as what arrives,
            // so the ribbons must use that same type — otherwise every leaf
            // would collapse onto the anchor's first row.
            let tgt_type = infer::anchor_type(&flat_graph, tgt_id, &state.function_declarations)
                .or_else(|| {
                    infer::incoming_anchor_type(&flat_graph, tgt_id, &state.function_declarations)
                });
            let target_leaves = tgt_type
                .as_ref()
                .map(|t| render::ordered_supported_leaves(t))
                .unwrap_or_default();

            // graph-level literal on the source anchor. When present, the sole
            // rendered leaf swaps to the thin "value line" style — same rule
            // the anchor markers follow, via the same lookup.
            let src_graph_value = infer::anchor_literal(&flat_graph, src_id);

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
                    SceneEntity,
                ))
                .id();

            for (k, leaf) in source_leaves.iter().enumerate() {
                // Same row offsets the marker stack uses, so each ribbon meets
                // its marker exactly.
                let y_src = render::leaf_row_offset(k);
                let leaf_kind = edge::leaf_kind_of(leaf);
                let y_tgt = if let Some(idx) = target_leaves
                    .iter()
                    .position(|l| edge::leaf_kind_of(l) == leaf_kind)
                {
                    render::leaf_row_offset(idx)
                } else {
                    // No matching leaf at the target: aim at its first row.
                    0.0
                };
                let Some(kind) = edge::leaf_kind_of(leaf) else {
                    continue;
                };
                let (height, label, line_mode) = if let Some(value) = src_graph_value.as_deref() {
                    let text = format!("  {}  {}  ", value, kind.type_name());
                    let handle = value_marquee_cache
                        .entry((kind, text.clone()))
                        .or_insert_with(|| {
                            edge::rasterize_marquee_text(&glyph_font, &text, &mut images)
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
                        scroll_speed: render::CELL * 0.5,
                        tile_length: render::CELL,
                        time: 0.0,
                        line_mode,
                        line_half_thickness: edge::RIBBON_LINE_HALF_THICKNESS_UV,
                        _pad0: 0.0,
                        _pad1: 0.0,
                        _pad2: 0.0,
                        label,
                    })),
                    ChildOf(edge_root),
                    SceneEntity,
                ));
            }
        }
    }

    for walked_graph in state.root_graph().walk_all_graphs() {
        let Some(bounds) = walked_graph.layout_graph.grid_bounds() else {
            continue;
        };
        let width_cells = (bounds.max.x - bounds.min.x + 1) as f32;
        let depth_cells = (bounds.max.z - bounds.min.z + 1) as f32;
        let size_x = width_cells * render::LAYOUT_SCALE.x.abs();
        let size_z = depth_cells * render::LAYOUT_SCALE.z.abs();
        // Cells are corner-anchored, so the inclusive range spans [min, max+1]
        // and its centre is (min + max + 1) / 2. Y stays on the address plane:
        // the grid is the upper bounding plane of the row it belongs to.
        let center_local = Vec3::new(
            (bounds.min.x + bounds.max.x + 1) as f32 * 0.5,
            0.0,
            (bounds.min.z + bounds.max.z + 1) as f32 * 0.5,
        );
        let world_center = render::layout_to_world(center_local + walked_graph.extra_offset);
        let offset = walked_graph.extra_offset;
        // `layout_range_to_world` re-normalises min/max: LAYOUT_SCALE negates
        // Z, so scaling the corners individually would yield an inverted rect
        // and the shader would draw no border at all.
        let (border_lo, border_hi) = render::layout_range_to_world(
            bounds.min.as_vec3() + offset,
            bounds.max.as_vec3() + offset,
            0.0,
        );
        let border_min = Vec2::new(border_lo.x, border_lo.z);
        let border_max = Vec2::new(border_hi.x, border_hi.z);
        // Collect multi-cell node footprints in this LayoutGraph and convert
        // to world-space XZ rects. Fed to the grid shader to suppress
        // interior grid lines inside merged fields.
        let mut footprints = [Vec4::ZERO; grid::MAX_FOOTPRINTS];
        let mut footprint_count: u32 = 0;
        for id in walked_graph.layout_graph.layout_nodes.keys() {
            let Some(fp) = walked_graph.layout_graph.node_footprint(id) else {
                continue;
            };
            if (fp.max.x - fp.min.x) == 0 && (fp.max.z - fp.min.z) == 0 {
                continue;
            }
            if (footprint_count as usize) >= grid::MAX_FOOTPRINTS {
                warn!(
                    "grid: more than {} footprints in one graph; truncating",
                    grid::MAX_FOOTPRINTS
                );
                break;
            }
            let (fp_lo, fp_hi) = render::layout_range_to_world(
                fp.min.as_vec3() + offset,
                fp.max.as_vec3() + offset,
                0.0,
            );
            footprints[footprint_count as usize] = Vec4::new(fp_lo.x, fp_lo.z, fp_hi.x, fp_hi.z);
            footprint_count += 1;
        }
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::default().mesh().size(size_x, size_z).build())),
            MeshMaterial3d(materials_grid.add(grid::GridMaterial {
                border_min,
                border_max,
                footprint_count,
                footprints,
                ..grid::GridMaterial::scope_surface(Vec3::X, Vec3::Z)
            })),
            Transform::from_translation(world_center),
            ScopeGridEntity {
                context: walked_graph.context.clone(),
                origin_offset: walked_graph.extra_offset,
                min: bounds.min,
                max: bounds.max,
            },
            SceneEntity,
        ));

        // root scope: the graph volume's two Z faces, drawn in the same
        // style as the Y plane the scope sits on. The front face is the Z=0
        // plane — the face of the source row that looks toward the origin,
        // where the removed wall used to stand — and the back face is the far
        // side of the Sink's cell, the volume's last Z edge. Both span the
        // graph's bounding box in X and Y, so the three surfaces together frame
        // the volume the graph grows in.
        if walked_graph.context.is_empty() {
            let height_cells = (bounds.max.y - bounds.min.y + 1) as f32;
            let size_y = height_cells * render::LAYOUT_SCALE.y.abs();
            // Cells are corner-anchored on Y too, so the face is centred on
            // the inclusive row range [min, max+1] like the X range above.
            let face_center = render::layout_to_world(
                Vec3::new(
                    (bounds.min.x + bounds.max.x + 1) as f32 * 0.5,
                    (bounds.min.y + bounds.max.y + 1) as f32 * 0.5,
                    0.0,
                ) + walked_graph.extra_offset,
            );
            // Front at the near corner of the first row, back at the far
            // corner of the Sink's row — hence `max.z + 1`.
            for z_cell in [bounds.min.z, bounds.max.z + 1] {
                let face_z = render::layout_to_world(
                    Vec3::new(0.0, 0.0, z_cell as f32) + walked_graph.extra_offset,
                )
                .z;
                commands.spawn((
                    Mesh3d(
                        meshes.add(
                            Plane3d::new(Vec3::Z, Vec2::new(size_x * 0.5, size_y * 0.5))
                                .mesh()
                                .build(),
                        ),
                    ),
                    MeshMaterial3d(
                        materials_grid.add(grid::GridMaterial::scope_surface(Vec3::X, Vec3::Y)),
                    ),
                    Transform::from_xyz(face_center.x, face_center.y, face_z),
                    SceneEntity,
                ));
            }
        }
    }

    // Selection caret, enclosing the addressed cell volume from the caret
    // address to address + (1,1,1). Rebuild-driven, like every other scene
    // entity — caret moves and mode switches already flag a rebuild.
    //
    // It shows which mode it is in: NORMAL outlines the cell, INSERT fills the
    // two faces the next insert would open along and blinks like a text caret.
    match *editor_mode {
        EditorMode::Normal => {
            for edge in render::cell_caret_edges(pick.selected_pos.as_vec3()) {
                commands.spawn((
                    Mesh3d(meshes.add(edge.mesh)),
                    MeshMaterial3d(materials.add(edge.material)),
                    edge.transform,
                    SceneEntity,
                ));
            }
        }
        EditorMode::Insert => {
            for face in render::cell_caret_faces(pick.selected_pos.as_vec3()) {
                commands.spawn((
                    Mesh3d(meshes.add(face.mesh)),
                    MeshMaterial3d(materials.add(face.material)),
                    face.transform,
                    CaretBlink,
                    SceneEntity,
                ));
            }
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
            SceneEntity,
        ));
    }
    */
}

fn spawn_ui(mut commands: Commands, ui_font: Res<UiFont>) {
    // Hamburger menu button (top-left) — opens the menu modal.
    spawn_hamburger_button(&mut commands, Vec2::new(12.0, 12.0));

    spawn_ui_button(
        &mut commands,
        &ui_font.0,
        "Delete Node",
        DeleteNodeButton,
        Vec2::new(12.0, 60.0),
        Display::Flex,
    );
    // The six "Add …" buttons are gone: what is created is typed at the
    // prompt that takes this slot in INSERT. The button is the mouse's way in
    // and does nothing but enter that mode. It hides itself, so unlike the
    // other buttons it must not also answer to `HideDuringStartMenu` — two
    // writers on one `display` flicker on the transition frame.
    spawn_insert_mode_button(&mut commands, &ui_font.0, Vec2::new(12.0, 96.0));

    // Bottom-left, opposite the mode indicator in the bottom-right corner.
    spawn_corner_button(
        &mut commands,
        &ui_font.0,
        "Evaluate",
        EvaluateButton,
        Val::Px(12.0),
        Val::Px(12.0),
    );
    // Directly above it: leaving the bound camera has to be an explicit act,
    // so it gets a control of its own rather than happening by dragging.
    spawn_corner_button(
        &mut commands,
        &ui_font.0,
        camera_mode_label(camera::CameraMode::Bound),
        CameraModeButton,
        Val::Px(12.0),
        Val::Px(52.0),
    );
    // Beside it, clear of the widest camera label.
    spawn_corner_checkbox(
        &mut commands,
        &ui_font.0,
        "semi ortho",
        SemiOrthoCheckbox,
        Val::Px(150.0),
        Val::Px(52.0),
    );
}

/// Marker for the checkbox that turns the bound mode's convergence on.
#[derive(Component)]
struct SemiOrthoCheckbox;

/// Marker on that checkbox's 16×16 swatch, which is what carries the state.
#[derive(Component)]
struct SemiOrthoCheckboxBox;

/// A labelled checkbox pinned to a screen corner.
///
/// The node editor's checkbox (`spawn_typecast_checkbox_and_value`) is a bare
/// swatch that gets respawned on every panel rebuild; a corner widget stands
/// still instead, so this one carries a label, makes the whole row clickable,
/// and leaves the swatch to a sync system. Colours and size are the panel's, so
/// the two read as the same control.
fn spawn_corner_checkbox<C: Bundle>(
    commands: &mut Commands,
    font: &Handle<Font>,
    label: &str,
    component: C,
    left: Val,
    bottom: Val,
) {
    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Auto,
                right: Val::Auto,
                left,
                bottom,
                padding: UiRect::axes(Val::Px(10.0), Val::Px(8.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.16, 0.16, 0.22, 0.9)),
            component,
            HideDuringStartMenu,
        ))
        .with_children(|parent| {
            parent.spawn((
                Node {
                    width: Val::Px(16.0),
                    height: Val::Px(16.0),
                    border: UiRect::all(Val::Px(1.5)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    flex_shrink: 0.0,
                    ..default()
                },
                BackgroundColor(UNCHECKED_COLOR),
                BorderColor::all(Color::srgb(0.35, 0.35, 0.5)),
                SemiOrthoCheckboxBox,
            ));
            parent.spawn((
                Text::new(label),
                text_font(font, 14.0),
                TextColor(Color::srgb(0.6, 0.6, 0.7)),
            ));
        });
}

const CHECKED_COLOR: Color = Color::srgb(0.133, 0.827, 0.933);
const UNCHECKED_COLOR: Color = Color::srgba(0.06, 0.06, 0.12, 0.95);

/// Toggle the bound mode's convergence. The projection eases across on its own
/// clock, so this only flips the intent.
fn handle_semi_ortho_checkbox(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<SemiOrthoCheckbox>)>,
    mut orbit: ResMut<camera::OrbitCamera>,
) {
    for interaction in interaction_q.iter() {
        if *interaction == Interaction::Pressed {
            orbit.semi_ortho = !orbit.semi_ortho;
        }
    }
}

fn sync_semi_ortho_checkbox(
    orbit: Res<camera::OrbitCamera>,
    mut box_q: Query<&mut BackgroundColor, With<SemiOrthoCheckboxBox>>,
) {
    let wanted = if orbit.semi_ortho {
        CHECKED_COLOR
    } else {
        UNCHECKED_COLOR
    };
    for mut color in box_q.iter_mut() {
        if color.0 != wanted {
            color.0 = wanted;
        }
    }
}

/// Marker for the button that switches the camera between its two modes.
#[derive(Component)]
struct CameraModeButton;

fn camera_mode_label(mode: camera::CameraMode) -> &'static str {
    match mode {
        camera::CameraMode::Bound => "Camera: bound",
        camera::CameraMode::Free => "Camera: free",
    }
}

fn spawn_corner_button<C: Bundle>(
    commands: &mut Commands,
    font: &Handle<Font>,
    label: &str,
    component: C,
    left: Val,
    bottom: Val,
) {
    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Auto,
                right: Val::Auto,
                left,
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

/// The "Add" button. Same look as `spawn_ui_button` produces, minus
/// `HideDuringStartMenu`: `sync_add_button` is its only writer, so the start
/// menu's bulk toggle must not reach it.
fn spawn_insert_mode_button(commands: &mut Commands, font: &Handle<Font>, pos: Vec2) {
    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(pos.y),
                left: Val::Px(pos.x),
                padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.16, 0.16, 0.22, 0.9)),
            AddNodeButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Add"),
                text_font(font, 14.0),
                TextColor(Color::srgb(0.6, 0.6, 0.7)),
            ));
        });
}

/// The INSERT prompt's panel, standing in the "Add" button's slot. Empty at
/// startup — `sync_insert_prompt_ui` fills it. `Button` on the root so
/// `pick_nodes`' `over_ui` test covers it and a click on the panel doesn't
/// move the caret to whatever cell lies behind it, the same reason
/// `spawn_node_editor_panel` carries one.
fn spawn_insert_prompt_panel(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(96.0),
            left: Val::Px(12.0),
            width: Val::Px(280.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(4.0),
            display: Display::None,
            ..default()
        },
        Button,
        InsertPromptPanel,
    ));
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
    mut state: ResMut<GraphState>,
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
        let Some((caret_graph, local)) = state.caret_graph(&pick) else {
            continue;
        };
        let Some(selected_node_id) = caret_graph.node_at(local) else {
            continue;
        };
        // Sink and BranchSource are constitutive parts of their scope, not
        // user-placed nodes — neither can be deleted.
        let is_fixture = matches!(
            caret_graph.graph.nodes.get(&selected_node_id),
            Some(model::node::ENode::Sink { .. } | model::node::ENode::BranchSource { .. })
        );
        if is_fixture {
            continue;
        }
        let updated = caret_graph.minus_node(&selected_node_id);
        if let Some((caret_graph_mut, _)) = state.caret_graph_mut(&pick) {
            *caret_graph_mut = updated;
            // Removing a node can shrink a constraint-less input that fed off
            // it, so shapes have to be recomputed, not just the layout.
            state.resettle();
            rebuild.0 = true;
        }
    }
}

/// May a node of this kind come into being at `local`?
///
/// The single source for both the greying of the suggestions and what `Enter`
/// accepts. It was written twice before — the handler and the button visuals —
/// and the two copies disagreed about a branch scope's `z == 0` plane and
/// about the sink row.
///
/// `graph` is the scope the caret addresses, `local` the caret in that scope's
/// coordinates.
fn kind_allowed(
    graph: &layout::LayoutGraph,
    is_root_scope: bool,
    local: IVec3,
    kind: &AddKind,
) -> bool {
    // The scope's last Z row belongs to its Sink alone.
    if graph.sink_z().is_some_and(|z| local.z >= z) {
        return false;
    }
    match kind {
        // The one kind that *needs* an occupied cell: a Pattern is added
        // below the Pattern the caret stands on.
        AddKind::Pattern => matches!(
            graph
                .node_at(local)
                .and_then(|id| graph.graph.nodes.get(&id)),
            Some(model::node::ENode::Pattern { .. })
        ),
        // Every scope reserves its Z=0 plane as the source row, and only the
        // root scope's holds Sources — a branch's already holds its
        // BranchSource.
        AddKind::Source => {
            graph.node_at(local).is_none() && is_root_scope && local.z == 0 && local.y == 0
        }
        _ => graph.node_at(local).is_none() && local.z != 0,
    }
}

/// Create a node of this kind at the caret. Returns whether the graph
/// actually changed, so the caller knows whether to flag a rebuild.
///
/// The caret does not follow: it keeps addressing the cell, which now holds
/// the new node.
fn insert_node_kind(state: &mut GraphState, pick: &PickState, kind: &AddKind) -> bool {
    // The caret's scope is the editing target — there is nothing else to
    // agree with, so no context guard is needed here.
    let Some(scope) = state.scope_of_caret(pick) else {
        return false;
    };
    let node_id_domain = state.node_id_domain.clone();
    let anchor_id_domain = state.anchor_id_domain.clone();
    let scope_graph = state.root_graph().resolve_context(&scope.path);
    if !kind_allowed(scope_graph, scope.path.is_empty(), scope.local, kind) {
        return false;
    }
    let new_pos = scope.local.as_vec3();
    let (new_layout, new_node_id_domain, new_anchor_id_domain) = match kind {
        AddKind::Constant(choice, value) => scope_graph.plus_constant(
            make_etype(*choice, value.clone()),
            new_pos,
            node_id_domain,
            anchor_id_domain,
        ),
        AddKind::Source => scope_graph.plus_source(new_pos, node_id_domain, anchor_id_domain),
        AddKind::FunctionCall(function_declaration_id) => {
            // The declaration decides the call's arity — `plus_function_call`
            // mints one input anchor per parameter — so a kind naming a
            // function the catalogue does not hold builds nothing.
            let Some(declaration) = state.function_declarations.get(function_declaration_id) else {
                return false;
            };
            scope_graph.plus_function_call(
                (function_declaration_id.clone(), declaration),
                new_pos,
                node_id_domain,
                anchor_id_domain,
            )
        }
        AddKind::TypeCast => scope_graph.plus_type_cast(
            model::r#type::EType::Int { value: None },
            new_pos,
            node_id_domain,
            anchor_id_domain,
        ),
        AddKind::Match => scope_graph.plus_match(new_pos, node_id_domain, anchor_id_domain),
        AddKind::Pattern => {
            // `kind_allowed` already established that this is a Pattern. The
            // selected Pattern keeps its row, so the caret stays where it is.
            let Some(id) = scope_graph.node_at(scope.local) else {
                return false;
            };
            scope_graph.plus_pattern_below(&id, node_id_domain, anchor_id_domain)
        }
    };
    if let Some(target) = state.root_graph_mut().resolve_context_mut(&scope.path) {
        *target = new_layout;
    }
    state.node_id_domain = new_node_id_domain;
    state.anchor_id_domain = new_anchor_id_domain;
    state.resettle();
    true
}

fn update_delete_button_visuals(
    pick: Res<PickState>,
    state: Res<GraphState>,
    mut button_q: Query<(&Interaction, &mut BackgroundColor, &Children), With<DeleteNodeButton>>,
    mut text_color_q: Query<&mut TextColor>,
) {
    let enabled = match state.caret_graph(&pick) {
        Some((layout, local)) => match layout.node_at(local) {
            Some(id) => !matches!(
                layout.graph.nodes.get(&id),
                Some(model::node::ENode::Sink { .. } | model::node::ENode::BranchSource { .. })
            ),
            None => false,
        },
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

/// The "Add" button does one thing: enter INSERT. What is created is typed
/// there, so the button no longer knows about node kinds at all.
fn handle_add_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<AddNodeButton>)>,
    eval: Res<EvalState>,
    mut mode: ResMut<EditorMode>,
    mut prompt: ResMut<InsertPrompt>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    if is_evaluating(&eval) {
        return;
    }
    for interaction in interaction_q.iter() {
        // Guarded so a press doesn't mark `EditorMode` changed for nothing.
        if *interaction == Interaction::Pressed && *mode != EditorMode::Insert {
            *mode = EditorMode::Insert;
            prompt.clear();
            // The caret is drawn per mode, and it is a scene entity.
            rebuild.0 = true;
        }
    }
}

/// The "Add" button's only writer: it shows in NORMAL and steps aside for the
/// prompt in INSERT, plus the usual hover tint. Folding the start menu and the
/// modals in here rather than wearing `HideDuringStartMenu` keeps it at one
/// writer, the way `sync_node_editor_ui` does for its panel.
fn sync_add_button(
    mode: Res<EditorMode>,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    mut button_q: Query<
        (&Interaction, &mut Node, &mut BackgroundColor, &Children),
        With<AddNodeButton>,
    >,
    mut text_color_q: Query<&mut TextColor>,
) {
    let visible = *mode == EditorMode::Normal
        && !start_menu.showing
        && !modal_is_open(&eval)
        && !is_evaluating(&eval);
    let desired = if visible {
        Display::Flex
    } else {
        Display::None
    };
    for (interaction, mut node, mut bg, children) in button_q.iter_mut() {
        if node.display != desired {
            node.display = desired;
        }
        let Ok(mut text_color) = text_color_q.get_mut(children[0]) else {
            continue;
        };
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

/// Clicking a suggestion is the same act as `Enter` on it — without this the
/// "Add" button would hand a mouse user a list they cannot use.
fn handle_insert_prompt_click(
    interaction_q: Query<(&Interaction, &InsertPromptOption), Changed<Interaction>>,
    eval: Res<EvalState>,
    mut state: ResMut<GraphState>,
    // Read-only: the caret keeps addressing the cell the node now fills.
    pick: Res<PickState>,
    mut prompt: ResMut<InsertPrompt>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    if is_evaluating(&eval) {
        return;
    }
    for (interaction, option) in interaction_q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if insert_node_kind(&mut state, &pick, &option.0) {
            prompt.clear();
            rebuild.0 = true;
        }
    }
}

/// The suggestions the prompt currently offers: the node kinds and every
/// declared function whose name carries the typed prefix, each with whether it
/// may be created at the caret.
///
/// The prefix filters, the legality only greys — those are two different
/// questions, and typing must not make a row silently disappear because the
/// caret happens to stand somewhere it is not allowed. The typed literal is
/// the exception that proves it: there the prefix is not something to filter
/// by, it *is* the row.
fn prompt_candidates(state: &GraphState, pick: &PickState, text: &str) -> Vec<Suggestion> {
    // Shift reports the uppercase character and the labels are CamelCase, so
    // the comparison has to ignore case in both directions.
    let prefix = text.to_ascii_lowercase();
    let scope = state.scope_of_caret(pick);
    let resolved = scope.as_ref().map(|s| {
        (
            state.root_graph().resolve_context(&s.path),
            s.path.is_empty(),
            s.local,
        )
    });
    // A HashMap has no order, so an as-it-comes iteration would reshuffle the
    // list between frames. Sorted by name, like the node editor's function
    // dropdown sorts it — byte order, which puts the symbols ahead of the
    // words and leaves `||` behind `substr`.
    let mut functions: Vec<_> = state.function_declarations.iter().collect();
    functions.sort_by(|a, b| a.1.name.cmp(&b.1.name));

    let caret_allows = |kind: &AddKind| {
        resolved.is_some_and(|(graph, is_root, local)| kind_allowed(graph, is_root, local, kind))
    };

    let mut candidates: Vec<Suggestion> = Vec::new();
    // The typed literal is not prefix-filtered — it *is* the text, and nothing
    // could hide behind it: no declared name begins with a digit or a quote.
    // It leads the list because it is the most literal reading of what stands
    // there.
    if let Some((choice, raw, _closed)) = typed_literal(text) {
        let r#type = make_etype(choice, Some(raw.clone()));
        // Checked with the one validator this codebase has, and the same one
        // the value will meet again at evaluation time — so a row that commits
        // here cannot fail there. It reads the raw text, not the typed one:
        // its `Char` arm wants exactly one character and would refuse `'c'`.
        let parses = eval::EValue::parse(&r#type, &raw).is_ok();
        let kind = AddKind::Constant(choice, Some(raw));
        let allowed = parses && caret_allows(&kind);
        candidates.push(Suggestion {
            kind,
            // Through the type's own rendering while it parses, which puts the
            // quotes back on: `'a` is offered as `'a'`, so the closing quote
            // reads as implied rather than as missing. What does not parse
            // keeps the typed text — dressing `'ab` up as `'ab'` would promise
            // something that cannot be built.
            label: if parses {
                r#type.to_string()
            } else {
                text.to_string()
            },
            detail: literal_type_name(choice),
            allowed,
        });
    }

    candidates.extend(
        NODE_KINDS
            .iter()
            .map(|(kind, label)| (kind.clone(), label.to_string(), String::new()))
            .chain(LITERAL_KEYWORDS.iter().map(|(label, choice, value)| {
                (
                    AddKind::Constant(*choice, value.map(str::to_string)),
                    label.to_string(),
                    literal_type_name(*choice),
                )
            }))
            .chain(functions.into_iter().map(|(id, declaration)| {
                (
                    AddKind::FunctionCall(id.clone()),
                    declaration.name.clone(),
                    signature_detail(declaration),
                )
            }))
            .filter(|(_, label, _)| label.to_ascii_lowercase().starts_with(&prefix))
            .map(|(kind, label, detail)| {
                let allowed = caret_allows(&kind);
                Suggestion {
                    kind,
                    label,
                    detail,
                    allowed,
                }
            }),
    );
    candidates
}

/// The literal the prompt text spells, if it spells one: its type, the raw
/// text between the quotes, and whether a closing quote has been typed.
///
/// The closing quote is optional — `'a` and `'a'` name the same character — so
/// the raw text drops a leading quote and *at most one* matching trailing one.
/// That makes `'''` the apostrophe and `''` nothing at all, which is not a
/// character and so cannot be committed. The empty *string* on the other hand
/// exists, so a lone `"` is a complete literal while a lone `'` is not.
///
/// A bare `-` or `+` is not a number, it is the name of a function, and the
/// list has to keep offering it.
fn typed_literal(text: &str) -> Option<(TypeChoice, String, bool)> {
    let first = text.chars().next()?;
    let quote = match first {
        '\'' => TypeChoice::Char,
        '"' => TypeChoice::String,
        c if c.is_ascii_digit() => return Some((TypeChoice::Int, text.to_string(), false)),
        '-' | '+' if text.len() > 1 => return Some((TypeChoice::Int, text.to_string(), false)),
        _ => return None,
    };
    // Both quotes are ASCII, so one byte is one quote and the slicing is safe.
    let body = &text[1..];
    let closed = body.ends_with(first);
    let raw = if closed {
        &body[..body.len() - 1]
    } else {
        body
    };
    Some((quote, raw.to_string(), closed))
}

/// Whether `Space` writes a space instead of making room: inside a quote that
/// has not been closed yet, and only there. Without it a string with a space
/// in it could not be typed at all. It asks `typed_literal`, so opening and
/// closing are decided in one place — were the two to drift apart, `Space`
/// would type into a literal the parser already considers finished.
fn in_open_quote(text: &str) -> bool {
    matches!(
        typed_literal(text),
        Some((TypeChoice::Char | TypeChoice::String, _, false))
    )
}

/// The parameter types of a declared function, for the prompt's muted right
/// column: it is what tells `charAt` from `concat`, and `substr` from `min`,
/// before the node exists.
///
/// Inputs only. The output type is what the finished node's own output anchor
/// shows, and spelling it out here would nearly double the widest row.
fn signature_detail(
    declaration: &model::function_declaration::FunctionDeclaration,
) -> String {
    declaration
        .inputs
        .iter()
        .map(|parameter| match &parameter.r#type {
            // An unconstrained parameter — `=` and `!=` compare any two values
            // — has no name in the type language: `EType::None` is the failed
            // result, not a top type, and the declaration's doc says `EType`
            // must not grow one. So this column says it in prose instead.
            None => "any".to_string(),
            Some(r#type) => short_type_name(r#type),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// A literal's type for that same column — through `short_type_name` as well,
/// so `Int` reads as `Int` in both halves of the list rather than as `Integer`
/// in one of them. `type_choice_label` keeps the long word for the node
/// editor's dropdown, which has the room for it.
fn literal_type_name(choice: TypeChoice) -> String {
    short_type_name(&infer::graph_type_to_eval_type(&make_etype(choice, None)))
}

/// A type as the suggestion list spells it. `EType`'s own rendering writes
/// `Integer`, which is the right word everywhere it has room; a row that has
/// to fit `Char|String,Char|String` beside a name does not.
fn short_type_name(r#type: &infer::EType) -> String {
    match r#type {
        infer::EType::Int(_) => "Int".to_string(),
        infer::EType::SumType(parts) => parts
            .iter()
            .map(short_type_name)
            .collect::<Vec<_>>()
            .join("|"),
        other => other.to_string(),
    }
}

/// Step the highlight to the next committable suggestion, wrapping around.
/// Rows that are greyed out can never be committed, so the highlight does not
/// stop on them; with nothing legal in the list it stays put.
fn step_selection(candidates: &[Suggestion], from: usize, delta: isize) -> usize {
    let len = candidates.len();
    if len == 0 {
        return 0;
    }
    let mut index = from.min(len - 1);
    for _ in 0..len {
        index = (index as isize + delta).rem_euclid(len as isize) as usize;
        if candidates[index].allowed {
            return index;
        }
    }
    from
}

/// Where the highlight lands after the text changed: the first suggestion
/// that can actually be committed.
fn first_selection(candidates: &[Suggestion]) -> usize {
    candidates
        .iter()
        .position(|suggestion| suggestion.allowed)
        .unwrap_or(0)
}

/// How many suggestions are on screen at once. There is no scrolling in this
/// codebase, so a longer list is windowed rather than clipped — the highlight
/// has to be able to walk all the way to the last entry.
const PROMPT_ROWS: usize = 8;

/// The slice of the candidate list that is on screen, always containing
/// `selected`.
///
/// Derived from `selected` rather than remembered: a stored scroll offset
/// would go stale the moment a keystroke or a caret move reshapes the list,
/// the same reason `selected` itself is re-clamped on read. The cost is that
/// the list slides by one per keypress once the highlight passes the middle,
/// instead of standing still until the highlight reaches an edge.
fn prompt_window(len: usize, selected: usize) -> std::ops::Range<usize> {
    if len <= PROMPT_ROWS {
        return 0..len;
    }
    let start = selected
        .saturating_sub(PROMPT_ROWS / 2)
        .min(len - PROMPT_ROWS);
    start..start + PROMPT_ROWS
}

#[derive(Default, PartialEq, Eq, Clone)]
struct InsertPromptFingerprint {
    visible: bool,
    text: String,
    selected: usize,
    candidates: Vec<Suggestion>,
}

/// The INSERT prompt: what was typed, and under it the node kinds that name
/// could still become. Sole writer of the panel's `display`, so the start menu
/// and the modals are folded in here rather than left to
/// `HideDuringStartMenu`; contents are respawned only when the fingerprint
/// moves, the way `sync_node_editor_ui` does it.
fn sync_insert_prompt_ui(
    mut commands: Commands,
    mode: Res<EditorMode>,
    prompt: Res<InsertPrompt>,
    state: Res<GraphState>,
    pick: Res<PickState>,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    ui_font: Res<UiFont>,
    mut panel_q: Query<(Entity, &mut Node), With<InsertPromptPanel>>,
    prompt_children_q: Query<Entity, With<InsertPromptEntity>>,
    mut cache: Local<InsertPromptFingerprint>,
) {
    let visible = *mode == EditorMode::Insert
        && !start_menu.showing
        && !modal_is_open(&eval)
        && !is_evaluating(&eval);
    let candidates = if visible {
        prompt_candidates(&state, &pick, &prompt.text)
    } else {
        Vec::new()
    };
    // Clamped on read, not on write: a caret move under a standing prompt may
    // shorten the list, and a stale index must not survive that.
    let selected = prompt.selected.min(candidates.len().saturating_sub(1));

    let fp = InsertPromptFingerprint {
        visible,
        text: prompt.text.clone(),
        selected,
        candidates: candidates.clone(),
    };
    if *cache == fp {
        return;
    }
    *cache = fp;

    for e in prompt_children_q.iter() {
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

    let font = &ui_font.0;
    // A prefix nothing answers to is shown on the text itself. An empty box
    // below would read as a broken widget instead of as a refusal.
    let text_color = if candidates.is_empty() {
        Color::srgb(0.95, 0.30, 0.30)
    } else {
        Color::srgb(0.91, 0.89, 0.87)
    };
    commands.entity(panel_entity).with_children(|panel| {
        panel
            .spawn((
                editor_text_input_node(),
                BackgroundColor(Color::srgba(0.06, 0.06, 0.12, 0.95)),
                BorderColor::all(Color::srgb(0.133, 0.827, 0.933)),
                InsertPromptEntity,
            ))
            .with_children(|row| {
                row.spawn((
                    Text::new(format!("{}|", prompt.text)),
                    text_font(font, 14.0),
                    TextColor(text_color),
                ));
            });
        if candidates.is_empty() {
            return;
        }
        panel
            .spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.08, 0.08, 0.14, 0.98)),
                InsertPromptEntity,
            ))
            .with_children(|options| {
                let window = prompt_window(candidates.len(), selected);
                // What the window hides is said, not swallowed: the list is
                // long enough that a silent cut would read as "that is all
                // there is".
                if window.start > 0 {
                    spawn_prompt_hint(options, font, format!("… {} more above", window.start));
                }
                for (index, suggestion) in candidates
                    .iter()
                    .enumerate()
                    .take(window.end)
                    .skip(window.start)
                {
                    // Only a committable row can hold the highlight, so a
                    // greyed one never looks like the answer to `Enter`.
                    let highlighted = index == selected && suggestion.allowed;
                    let label_color = if suggestion.allowed {
                        Color::srgb(0.85, 0.85, 0.9)
                    } else {
                        Color::srgb(0.35, 0.35, 0.4)
                    };
                    options
                        .spawn((
                            Button,
                            Node {
                                padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                                border_radius: BorderRadius::all(Val::Px(3.0)),
                                flex_direction: FlexDirection::Row,
                                justify_content: JustifyContent::SpaceBetween,
                                align_items: AlignItems::Center,
                                column_gap: Val::Px(12.0),
                                ..default()
                            },
                            BackgroundColor(if highlighted {
                                Color::srgba(0.2, 0.2, 0.3, 0.95)
                            } else {
                                Color::srgba(0.0, 0.0, 0.0, 0.0)
                            }),
                            InsertPromptOption(suggestion.kind.clone()),
                        ))
                        .with_children(|row| {
                            row.spawn((
                                Text::new(suggestion.label.clone()),
                                text_font(font, 14.0),
                                TextColor(label_color),
                            ));
                            // Spawned only when it says something —
                            // `SpaceBetween` already puts a lone child at the
                            // start, so a node kind needs no empty placeholder
                            // to stay left-aligned.
                            if !suggestion.detail.is_empty() {
                                row.spawn((
                                    Text::new(suggestion.detail.clone()),
                                    text_font(font, 12.0),
                                    TextColor(if suggestion.allowed {
                                        Color::srgb(0.6, 0.6, 0.7)
                                    } else {
                                        label_color
                                    }),
                                    // The signature is the row's width, not
                                    // its slack: shrinking it would wrap
                                    // `Char|String,Char|String` onto a second
                                    // line and make the rows uneven.
                                    Node {
                                        flex_shrink: 0.0,
                                        ..default()
                                    },
                                ));
                            }
                        });
                }
                if window.end < candidates.len() {
                    spawn_prompt_hint(
                        options,
                        font,
                        format!("… {} more below", candidates.len() - window.end),
                    );
                }
            });
    });
}

/// A row that counts what the window leaves off. It carries neither `Button`
/// nor `InsertPromptOption`: clicking a tally must not build anything.
fn spawn_prompt_hint(options: &mut ChildSpawnerCommands, font: &Handle<Font>, text: String) {
    options
        .spawn(Node {
            padding: UiRect::axes(Val::Px(8.0), Val::Px(2.0)),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(text),
                text_font(font, 12.0),
                TextColor(Color::srgb(0.35, 0.35, 0.4)),
            ));
        });
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
    node_id: Option<model::node::Id>,
    variant: NodeVariantKind,
    type_choice: Option<TypeChoice>,
    typecast_has_value: bool,
    func_id: Option<model::function_declaration::FunctionDeclarationId>,
    dropdown_open: Option<(model::node::Id, DropdownKind)>,
    visible: bool,
}

fn sync_node_editor_ui(
    mut commands: Commands,
    state: Res<GraphState>,
    pick: Res<PickState>,
    dropdown_state: Res<DropdownState>,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    ui_font: Res<UiFont>,
    mut panel_q: Query<(Entity, &mut Node), With<NodeEditorPanel>>,
    editor_children_q: Query<Entity, With<NodeEditorEntity>>,
    mut cache: Local<NodeEditorFingerprint>,
) {
    let caret = state.caret_graph(&pick);
    // The panel edits the property this address stands for, so it only opens
    // on a node's body cell — never on one of its anchor rows.
    let node_id = caret.and_then(|(layout, local)| {
        let id = layout.node_at(local)?;
        let ln = layout.layout_nodes.get(&id)?;
        let node_local = local - ln.pos.round().as_ivec3();
        matches!(ln.shape.role_at(node_local), Some(layout::CellRole::Body)).then_some(id)
    });
    let node = node_id
        .as_ref()
        .and_then(|id| caret.and_then(|(layout, _)| layout.graph.nodes.get(id)));
    let variant = variant_kind(node);
    let type_choice = node.and_then(|n| match n {
        model::node::ENode::Constant { r#type, .. }
        | model::node::ENode::TypeCast { r#type, .. }
        | model::node::ENode::Pattern { r#type, .. } => Some(type_choice_of(r#type)),
        _ => None,
    });
    let typecast_has_value = match node {
        Some(model::node::ENode::TypeCast { r#type, .. })
        | Some(model::node::ENode::Pattern { r#type, .. }) => value_of_etype(r#type).is_some(),
        _ => false,
    };
    let func_id = match node {
        Some(model::node::ENode::FunctionCall {
            function_declaration_id,
            ..
        }) => Some(function_declaration_id.clone()),
        _ => None,
    };

    let editable = matches!(
        variant,
        NodeVariantKind::Constant
            | NodeVariantKind::TypeCast
            | NodeVariantKind::Source
            | NodeVariantKind::FunctionCall
            | NodeVariantKind::Pattern
    );
    let visible = editable && !start_menu.showing && !is_evaluating(&eval);

    let fp = NodeEditorFingerprint {
        node_id: node_id.clone(),
        variant,
        type_choice,
        typecast_has_value,
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
            model::node::ENode::Constant { r#type, .. } => {
                spawn_editor_label(panel, font, "Constant");
                spawn_labeled_row(panel, font, "Type", |slot| {
                    spawn_type_dropdown(slot, font, &node_id, r#type, &dropdown_state.open);
                });
                if !matches!(r#type, model::r#type::EType::None { .. }) {
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
            model::node::ENode::Source { name, r#type, .. } => {
                spawn_labeled_row(panel, font, "Name", |slot| {
                    spawn_name_input(slot, font, &node_id, name);
                });
                spawn_labeled_row(panel, font, "Type", |slot| {
                    spawn_type_dropdown(slot, font, &node_id, r#type, &dropdown_state.open);
                });
            }
            model::node::ENode::TypeCast { r#type, .. } => {
                spawn_editor_label(panel, font, "TypeCast");
                spawn_labeled_row(panel, font, "Type", |slot| {
                    spawn_type_dropdown(slot, font, &node_id, r#type, &dropdown_state.open);
                });
                if !matches!(r#type, model::r#type::EType::None { .. }) {
                    spawn_labeled_row(panel, font, "Value", |slot| {
                        spawn_typecast_checkbox_and_value(
                            slot,
                            font,
                            &node_id,
                            r#type,
                            &dropdown_state.open,
                        );
                    });
                }
            }
            model::node::ENode::Pattern { r#type, .. } => {
                spawn_editor_label(panel, font, "Pattern");
                spawn_labeled_row(panel, font, "Type", |slot| {
                    spawn_type_dropdown(slot, font, &node_id, r#type, &dropdown_state.open);
                });
                if !matches!(r#type, model::r#type::EType::None { .. }) {
                    spawn_labeled_row(panel, font, "Value", |slot| {
                        spawn_typecast_checkbox_and_value(
                            slot,
                            font,
                            &node_id,
                            r#type,
                            &dropdown_state.open,
                        );
                    });
                }
            }
            model::node::ENode::FunctionCall {
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
    node_id: &model::node::Id,
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
                field: NodeEditorField::SourceName,
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
    node_id: &model::node::Id,
    current: &model::r#type::EType,
    open: &Option<(model::node::Id, DropdownKind)>,
) {
    let current_choice = type_choice_of(current);
    let label = type_choice_label(current_choice);
    spawn_dropdown_root(
        panel,
        font,
        node_id,
        DropdownKind::Type,
        label,
        open,
        |options| {
            for tc in TYPE_CHOICES {
                let is_current = tc == current_choice;
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
    node_id: &model::node::Id,
    current: &model::function_declaration::FunctionDeclarationId,
    declarations: &std::collections::HashMap<
        model::function_declaration::FunctionDeclarationId,
        model::function_declaration::FunctionDeclaration,
    >,
    open: &Option<(model::node::Id, DropdownKind)>,
) {
    let label = declarations
        .get(current)
        .map(|d| d.name.as_str())
        .unwrap_or("?");
    let mut entries: Vec<(model::function_declaration::FunctionDeclarationId, String)> =
        declarations
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
    node_id: &model::node::Id,
    current: &model::r#type::EType,
    enabled: bool,
    open: &Option<(model::node::Id, DropdownKind)>,
) {
    match current {
        model::r#type::EType::None { .. } => {}
        model::r#type::EType::Bool { value } => {
            let current_bool = value.as_deref() == Some("true");
            let label = value.as_deref().unwrap_or("Bool");
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

fn spawn_typecast_checkbox_and_value(
    row: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    node_id: &model::node::Id,
    current: &model::r#type::EType,
    open: &Option<(model::node::Id, DropdownKind)>,
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
    node_id: &model::node::Id,
    kind: DropdownKind,
    label: &str,
    open: &Option<(model::node::Id, DropdownKind)>,
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
    node_id: &model::node::Id,
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
    mut state: ResMut<GraphState>,
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
                    .layout_graph
                    .find_node_graph_mut(&option.node_id)
                    .and_then(|a| a.graph.nodes.get_mut(&option.node_id));
                match node {
                    Some(model::node::ENode::Constant { r#type, .. })
                    | Some(model::node::ENode::TypeCast { r#type, .. })
                    | Some(model::node::ENode::Source { r#type, .. })
                    | Some(model::node::ENode::Pattern { r#type, .. }) => {
                        let value = value_of_etype(r#type);
                        *r#type = make_etype(*new_choice, value);
                        // Anchor heights follow declared types, so a type
                        // change reshapes the node and its neighbours.
                        state.resettle();
                        rebuild.0 = true;
                    }
                    _ => {}
                }
            }
            DropdownChoice::BoolValue(v) => {
                let node = state
                    .layout_graph
                    .find_node_graph_mut(&option.node_id)
                    .and_then(|a| a.graph.nodes.get_mut(&option.node_id));
                match node {
                    Some(model::node::ENode::Constant { r#type, .. })
                    | Some(model::node::ENode::TypeCast { r#type, .. })
                    | Some(model::node::ENode::Source { r#type, .. })
                    | Some(model::node::ENode::Pattern { r#type, .. }) => {
                        if let model::r#type::EType::Bool { value } = r#type {
                            *value = Some(if *v { "true" } else { "false" }.to_string());
                            state.resettle();
                            rebuild.0 = true;
                        }
                    }
                    _ => {}
                }
            }
            DropdownChoice::Function(new_fn_id) => {
                let decl = state.function_declarations.get(new_fn_id).cloned();
                if let Some(new_decl) = decl {
                    let node_id_domain = state.node_id_domain.clone();
                    let anchor_id_domain = state.anchor_id_domain.clone();
                    if let Some(owning_graph) =
                        state.layout_graph.find_node_graph_mut(&option.node_id)
                    {
                        let (new_layout, new_node_id_domain, new_anchor_id_domain) = owning_graph
                            .with_function_call_replaced(
                                &option.node_id,
                                (new_fn_id.clone(), &new_decl),
                                node_id_domain,
                                anchor_id_domain,
                            );
                        *owning_graph = new_layout;
                        state.node_id_domain = new_node_id_domain;
                        state.anchor_id_domain = new_anchor_id_domain;
                        state.resettle();
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
    mut state: ResMut<GraphState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    for (interaction, cb) in interaction_q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let r#type = match state
            .layout_graph
            .find_node_graph_mut(&cb.node_id)
            .and_then(|a| a.graph.nodes.get_mut(&cb.node_id))
        {
            Some(model::node::ENode::TypeCast { r#type, .. })
            | Some(model::node::ENode::Pattern { r#type, .. }) => r#type,
            _ => continue,
        };
        let current = value_of_etype(r#type);
        let toggled: Option<String> = if current.is_some() {
            None
        } else {
            Some(String::new())
        };
        *r#type = make_etype(type_choice_of(r#type), toggled);
        // Pinning or unpinning a literal changes how the anchor renders and,
        // through a BranchSource, what its branch declares.
        state.resettle();
        rebuild.0 = true;
    }
}

fn handle_node_editor_text_input(
    input_q: Query<(&NodeEditorTextInput, &TextInput), Changed<TextInput>>,
    mut state: ResMut<GraphState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    for (editor_input, input) in input_q.iter() {
        let Some(node) = state
            .layout_graph
            .find_node_graph_mut(&editor_input.node_id)
            .and_then(|a| a.graph.nodes.get_mut(&editor_input.node_id))
        else {
            continue;
        };
        match editor_input.field {
            NodeEditorField::SourceName => {
                if let model::node::ENode::Source { name, .. } = node {
                    if *name != input.value {
                        *name = input.value.clone();
                        // The name is written along the body, so it decides
                        // how many cells that body claims — renaming reshapes
                        // the node exactly the way retyping a value does.
                        state.resettle();
                        rebuild.0 = true;
                    }
                }
            }
            NodeEditorField::Value => {
                let r#type = match node {
                    model::node::ENode::Constant { r#type, .. }
                    | model::node::ENode::TypeCast { r#type, .. }
                    | model::node::ENode::Source { r#type, .. }
                    | model::node::ENode::Pattern { r#type, .. } => r#type,
                    _ => continue,
                };
                let choice = type_choice_of(r#type);
                // Ignore the initial spawn's Added-tick: an empty input against
                // a None value is not a user edit.
                if input.value.is_empty() && value_of_etype(r#type).is_none() {
                    continue;
                }
                let new_value = Some(input.value.clone());
                if value_of_etype(r#type) != new_value {
                    *r#type = make_etype(choice, new_value);
                    // A literal reaches the branch through its BranchSource,
                    // so the shapes downstream have to be refreshed too.
                    state.resettle();
                    rebuild.0 = true;
                }
            }
        }
    }
}

/// Switch the camera between bound and free.
///
/// Bound → free matches the free camera's distance to the scale the user was
/// looking at, so the caret's own plane keeps its size across the change and
/// what happens visually is the depth of the picture opening up rather than a
/// jump. Free → bound is a journey back to the default view; the scale setting
/// is the user's and is left alone.
fn handle_camera_mode_button(
    mut interaction_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<CameraModeButton>),
    >,
    mut text_color_q: Query<&mut TextColor>,
    windows: Query<&Window>,
    pick: Res<PickState>,
    mut orbit: ResMut<camera::OrbitCamera>,
    mut tween: ResMut<camera::CameraTween>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                let height = windows
                    .single()
                    .map(|window| window.height())
                    .unwrap_or(1080.0);
                match orbit.mode {
                    camera::CameraMode::Bound => {
                        // Hand the free camera the view it is taking over:
                        // the same distance, the angles the oblique projection
                        // implies a viewer would stand at, and the field of
                        // view that keeps the caret's plane the size it
                        // already is. What is left to notice on the switch is
                        // then only the real difference — that the axes stop
                        // being exactly aligned — instead of a new viewpoint
                        // burying it.
                        let (theta, phi) = camera::oblique_view_angles();
                        let visible_world = height / orbit.cell_pixels;
                        orbit.free_fov = 2.0 * (visible_world * 0.5 / orbit.radius).atan();
                        orbit.mode = camera::CameraMode::Free;
                        tween.to_view(&orbit, theta, phi, orbit.radius, orbit.target);
                    }
                    camera::CameraMode::Free => {
                        let radius = camera::bound_radius(&orbit, height);
                        orbit.mode = camera::CameraMode::Bound;
                        tween.to_view(
                            &orbit,
                            camera::RESET_THETA,
                            camera::RESET_PHI,
                            radius,
                            render::cell_center_world(pick.selected_pos.as_vec3()),
                        );
                    }
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

/// Keep the mode button's caption on the mode it will show, not the one it
/// switches to.
fn sync_camera_mode_button(
    orbit: Res<camera::OrbitCamera>,
    button_q: Query<&Children, With<CameraModeButton>>,
    mut text_q: Query<&mut Text>,
) {
    let label = camera_mode_label(orbit.mode);
    for children in button_q.iter() {
        if let Ok(mut text) = text_q.get_mut(children[0]) {
            if text.0 != label {
                text.0 = label.to_string();
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
    state: Res<GraphState>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                if is_evaluating(&eval) {
                    // Already showing a modal or running — ignore.
                    continue;
                }
                if !infer::sink_has_input(&state.root_graph().graph) {
                    eval.phase = EvalPhase::ErrorModal(
                        "Cannot evaluate, because no node is connected to the sink".to_string(),
                    );
                    continue;
                }
                // Flatten pattern sub-scenes in so eval and var-decl collection
                // see every node, not just the program-level ones.
                let graph = state.root_graph().flattened_graph();
                let sources = infer::collect_sources(&graph);
                if !sources.is_empty() {
                    eval.phase = EvalPhase::SourcePrompt {
                        inputs: sources
                            .into_iter()
                            .map(|(id, _name)| (id, String::new()))
                            .collect(),
                    };
                } else {
                    let user_source_values = std::collections::HashMap::new();
                    match eval::State::new(
                        &graph,
                        &user_source_values,
                        &state.function_declarations,
                    ) {
                        Ok(initial) => {
                            eval.phase = EvalPhase::Running {
                                states: vec![initial],
                                current: 0,
                                user_source_values,
                            };
                        }
                        Err(errors) => {
                            eval.phase = EvalPhase::ErrorModal(errors.join("\n"));
                        }
                    }
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
    SourcePrompt,
}

fn modal_kind(phase: &EvalPhase) -> ModalKind {
    match phase {
        EvalPhase::ErrorModal(_) => ModalKind::Error,
        EvalPhase::ControlsModal => ModalKind::Controls,
        EvalPhase::SourcePrompt { .. } => ModalKind::SourcePrompt,
        _ => ModalKind::None,
    }
}

fn sync_modal_ui(
    mut commands: Commands,
    eval: Res<EvalState>,
    state: Res<GraphState>,
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
        EvalPhase::SourcePrompt { inputs } => {
            let graph = state.root_graph().flattened_graph();
            let rows: Vec<(model::node::Id, String)> = inputs
                .iter()
                .map(|(id, _)| {
                    let name = match graph.nodes.get(id) {
                        Some(model::node::ENode::Source { name, .. }) => name.clone(),
                        _ => "?".to_string(),
                    };
                    (id.clone(), name)
                })
                .collect();
            spawn_source_modal(&mut commands, &ui_font.0, rows);
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
        ("Ctrl + Left drag", "Free camera: orbit"),
        ("Ctrl + Right drag", "Free camera: pan"),
        ("Ctrl + Scroll", "Zoom (bound: cell size, free: distance)"),
    ];
    let key_bindings: &[(&str, &str)] = &[
        ("Arrow keys", "NORMAL: move selection along the grid"),
        ("Shift + Up/Down", "NORMAL: move selection vertically"),
        ("Ctrl + Arrow", "NORMAL: move the selected node"),
        (
            "Ctrl + Shift + Up/Down",
            "NORMAL: move the selected node vertically",
        ),
        ("i", "Enter INSERT mode"),
        ("Space", "INSERT: open a cell behind the caret"),
        ("Return", "INSERT: open a column (X)"),
        ("Shift + Return", "INSERT: open a row (Y)"),
        ("Escape", "Unfocus text input, then leave INSERT"),
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

fn spawn_source_modal(
    commands: &mut Commands,
    font: &Handle<Font>,
    rows: Vec<(model::node::Id, String)>,
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
                    Text::new("Enter values for the sources:"),
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
                                ModalSourceInput { node_id },
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
    mut state: ResMut<GraphState>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut start_menu: ResMut<StartMenu>,
    mut pick: ResMut<PickState>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                let node_id_domain = state.node_id_domain.clone();
                let anchor_id_domain = state.anchor_id_domain.clone();
                let (fresh, new_node_id_domain, new_anchor_id_domain) =
                    layout::LayoutGraph::new(node_id_domain, anchor_id_domain);
                *state.root_graph_mut() = fresh;
                state.node_id_domain = new_node_id_domain;
                state.anchor_id_domain = new_anchor_id_domain;
                pick.selected_pos = IVec3::ZERO;
                state.resettle();
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
    state: Res<GraphState>,
    input_q: Query<(&ModalSourceInput, &TextInput)>,
) {
    for (interaction, mut bg, children) in interaction_q.iter_mut() {
        let mut color = text_color_q.get_mut(children[0]).unwrap();
        match *interaction {
            Interaction::Pressed => {
                let graph = state.root_graph().flattened_graph();
                let mut user_source_values: std::collections::HashMap<
                    model::node::Id,
                    eval::EValue,
                > = std::collections::HashMap::new();
                let mut parse_errors: Vec<String> = Vec::new();
                for (m, input) in input_q.iter() {
                    if let Some(model::node::ENode::Source { r#type, name, .. }) =
                        graph.nodes.get(&m.node_id)
                    {
                        match eval::EValue::parse(r#type, &input.value) {
                            Ok(value) => {
                                user_source_values.insert(m.node_id.clone(), value);
                            }
                            Err(error) => parse_errors.push(format!("{}: {}", name, error)),
                        }
                    }
                }
                if !parse_errors.is_empty() {
                    eval.phase = EvalPhase::ErrorModal(parse_errors.join("\n"));
                } else {
                    match eval::State::new(
                        &graph,
                        &user_source_values,
                        &state.function_declarations,
                    ) {
                        Ok(initial) => {
                            eval.phase = EvalPhase::Running {
                                states: vec![initial],
                                current: 0,
                                user_source_values,
                            };
                        }
                        Err(errors) => {
                            eval.phase = EvalPhase::ErrorModal(errors.join("\n"));
                        }
                    }
                }
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
    state: Res<GraphState>,
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
        let graph = state.root_graph().flattened_graph();
        // Compute the next step first (immutable borrow of `eval.phase`), then
        // apply it — `Err` reassigns `eval.phase`, which the borrow would block.
        let step_result = if let EvalPhase::Running {
            states,
            current,
            user_source_values,
        } = &eval.phase
        {
            graph
                .nodes
                .get(&graph.sink_node_id)
                .cloned()
                .map(|sink_node| {
                    states[*current].eval_next_step(
                        &graph,
                        user_source_values,
                        (graph.sink_node_id.clone(), sink_node),
                        &state.function_declarations,
                    )
                })
        } else {
            None
        };
        match step_result {
            Some(Ok(next_state)) => {
                if let EvalPhase::Running {
                    states, current, ..
                } = &mut eval.phase
                {
                    // Only record a new snapshot if the step actually resolved
                    // more nodes, so a dead `Next` press does not grow history.
                    if next_state.node_ids_to_values.len()
                        > states[*current].node_ids_to_values.len()
                    {
                        states.truncate(*current + 1);
                        states.push(next_state);
                        *current += 1;
                    }
                }
            }
            Some(Err(errors)) => {
                eval.phase = EvalPhase::ErrorModal(errors.join("\n"));
            }
            None => {}
        }
    }
}

fn update_step_button_visuals(
    eval: Res<EvalState>,
    state: Res<GraphState>,
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
        EvalPhase::Running {
            states, current, ..
        } => {
            let next_possible = !states[*current].is_evaluated(&state.root_graph().graph);
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
    state: Res<GraphState>,
    ui_font: Res<UiFont>,
    mut existing_q: Query<(Entity, &ValueLabel, &mut Text)>,
) {
    let snapshot: Option<&eval::State> = match &eval.phase {
        EvalPhase::Running {
            states, current, ..
        } => Some(&states[*current]),
        _ => None,
    };
    let Some(snapshot) = snapshot else {
        for (entity, _, _) in existing_q.iter() {
            commands.entity(entity).despawn();
        }
        return;
    };

    let mut kept: std::collections::HashSet<model::node::Id> = std::collections::HashSet::new();
    for (entity, label, mut text) in existing_q.iter_mut() {
        if let Some(value) = snapshot.node_ids_to_values.get(&label.node_id) {
            let rendered = value.to_string();
            if text.0 != rendered {
                text.0 = rendered;
            }
            kept.insert(label.node_id.clone());
        } else {
            commands.entity(entity).despawn();
        }
    }

    for (id, value) in snapshot.node_ids_to_values.iter() {
        if kept.contains(id) {
            continue;
        }
        let Some(layout_node) = state.root_graph().layout_nodes.get(id) else {
            continue;
        };
        let world_pos = render::cell_center_world(layout_node.pos);
        commands.spawn((
            Text::new(value.to_string()),
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
fn animate_nodes(time: Res<Time>, mut query: Query<(&NodeEntity, &mut Transform)>) {
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
    state: Res<GraphState>,
    rebuild: ResMut<NeedsRebuild>,
    query_ast_entities: Query<Entity, With<SceneEntity>>,
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
    state: Res<GraphState>,
    ui_font: Res<UiFont>,
    pick: Res<PickState>,
    editor_mode: Res<EditorMode>,
    mut rebuild: ResMut<NeedsRebuild>,
    _query_scene_entities: Query<Entity, With<SceneEntity>>,
) {
    if rebuild.0 {
        spawn_graph_nodes(
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
            editor_mode,
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

/// Spawn one type-marker stack: the coloured rect and its letter per leaf,
/// plus the gizmo line and value label when the anchor carries a literal.
///
/// Shared by the per-anchor markers and the node-level ones a Pattern uses, so
/// a Pattern's band is built exactly like an anchor's.
fn spawn_type_markers(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    font: &Handle<Font>,
    markers: Vec<render::RenderTypeMarker>,
) {
    for marker in markers {
        let render::RenderTypeMarker {
            rect,
            label,
            value_line,
            value_label,
        } = marker;
        if let Some(rect) = rect {
            commands.spawn((
                Mesh3d(meshes.add(rect.mesh)),
                MeshMaterial3d(materials.add(rect.material)),
                rect.transform,
                SceneEntity,
            ));
        }
        if let Some(label) = label {
            spawn_world_label(commands, font, label, SceneEntity);
        }
        if let Some(line) = value_line {
            commands.spawn((
                Mesh3d(meshes.add(line.mesh)),
                MeshMaterial3d(materials.add(line.material)),
                line.transform,
                SceneEntity,
            ));
        }
        if let Some(vlabel) = value_label {
            spawn_world_label(commands, font, vlabel, SceneEntity);
        }
    }
}
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
        // The orthographic view box reaches behind the camera on purpose (see
        // `camera::bound_projection`), so a point behind it still projects to a
        // perfectly plausible screen position instead of being rejected. Ask
        // which side of the camera it is on directly.
        let offset = label.world_pos - cam_gt.translation();
        let in_front = cam_gt.forward().dot(offset) > 0.0;
        let screen = camera.world_to_viewport(cam_gt, label.world_pos);
        if let (true, Ok(screen_pos)) = (in_front, screen) {
            let size = computed.size();
            node.left = Val::Px(screen_pos.x - size.x / 2.0 + label.offset.x);
            node.top = Val::Px(screen_pos.y - size.y / 2.0 + label.offset.y);
            *vis = Visibility::Visible;
        } else {
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

fn spawn_mode_display(mut commands: Commands, ui_font: Res<UiFont>) {
    commands.spawn((
        Text::new("NORMAL"),
        text_font(&ui_font.0, 14.0),
        TextColor(Color::srgb(0.6, 0.6, 0.7)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(16.0),
            right: Val::Px(14.0),
            ..default()
        },
        ModeDisplay,
        HideDuringStartMenu,
    ));
}

/// Blink the INSERT-mode caret on a fixed wall-clock cycle, so it keeps its
/// rhythm across scene rebuilds instead of restarting on every caret move.
fn blink_caret(time: Res<Time>, mut caret_q: Query<&mut Visibility, With<CaretBlink>>) {
    let lit = (time.elapsed_secs() / CARET_BLINK_SECONDS) as i64 % 2 == 0;
    let desired = if lit {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut visibility in caret_q.iter_mut() {
        if *visibility != desired {
            *visibility = desired;
        }
    }
}

fn update_mode_display(
    mode: Res<EditorMode>,
    orbit: Res<camera::OrbitCamera>,
    mut text_q: Query<(&mut Text, &mut TextColor), With<ModeDisplay>>,
) {
    let Ok((mut text, mut color)) = text_q.single_mut() else {
        return;
    };
    let (label, tint) = match *mode {
        EditorMode::Normal => ("NORMAL", Color::srgb(0.6, 0.6, 0.7)),
        // Green, and brighter than NORMAL: INSERT is the state that changes
        // the graph, so it should be the one that catches the eye.
        EditorMode::Insert => ("INSERT", Color::srgb(0.35, 0.85, 0.55)),
    };
    // The free camera suspends the guarantees the bound one gives, so it is
    // worth saying out loud next to the editing mode.
    let label = match orbit.mode {
        camera::CameraMode::Bound => label.to_string(),
        camera::CameraMode::Free => format!("{} · FREE", label),
    };
    if text.0 != label {
        text.0 = label;
    }
    *color = TextColor(tint);
}

fn spawn_breadcrumb_display(mut commands: Commands, ui_font: Res<UiFont>) {
    commands.spawn((
        Text::new("Root"),
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
    state: Res<GraphState>,
    mut text_q: Query<&mut Text, With<BreadcrumbDisplay>>,
) {
    let Ok(mut text) = text_q.single_mut() else {
        return;
    };
    let Some(scope) = state.scope_of_caret(&pick) else {
        text.0 = "—".to_string();
        return;
    };
    let mut parts = vec!["Root".to_string()];
    for id in &scope.path {
        parts.push(format!("Pattern({})", id));
    }
    text.0 = parts.join(" > ");
}

fn pick_nodes(
    camera_q: Query<(&Camera, &GlobalTransform), With<camera::OrbitCameraTag>>,
    windows: Query<&Window>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut pick: ResMut<PickState>,
    node_q: Query<(&NodeEntity, &Transform)>,
    grid_q: Query<(Entity, &ScopeGridEntity)>,
    state: Res<GraphState>,
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

    // Ray-sphere test against nodes, at a fraction of a cell.
    let radius = render::CELL * 0.35;
    let mut closest: Option<(model::node::Id, f32)> = None;
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

    // Ray-plane test against each spawned graph grid. Each graph grid sits at
    // its own Y (root scope Y=0; Pattern sub-graph at Pattern's world Y) and
    // spans a rectangle in local grid coords. Pick the closest rect hit.
    //
    // Every conversion goes through `render::layout_to_world` /
    // `world_to_layout` because LAYOUT_SCALE negates Y and Z — dividing by a
    // bare 3.0 here would mirror the picking against the rendering.
    let mut grid_hit: Option<HoveredGrid> = None;
    let mut best_t = f32::INFINITY;
    if !over_ui && closest.is_none() {
        for (entity, ag) in grid_q.iter() {
            let origin_world = render::layout_to_world(ag.origin_offset);
            let denom = ray.direction.y;
            if denom.abs() < 1e-4 {
                continue;
            }
            let t = (origin_world.y - ray.origin.y) / denom;
            if t <= 0.0 || t >= best_t {
                continue;
            }
            let hit = ray.origin + *ray.direction * t;
            // Cells are corner-anchored — cell N covers [N, N+1) — so the
            // containing cell is the floor, not the nearest address.
            let local = render::world_to_layout(hit - origin_world);
            let local_x = local.x.floor() as i32;
            let local_z = local.z.floor() as i32;
            if local_x < ag.min.x || local_x > ag.max.x {
                continue;
            }
            if local_z < ag.min.z || local_z > ag.max.z {
                continue;
            }
            let cell_local = IVec3::new(local_x, 0, local_z);
            let center_world = render::cell_center_world(cell_local.as_vec3() + ag.origin_offset);
            best_t = t;
            grid_hit = Some(HoveredGrid {
                global_pos: cell_local + ag.origin_offset.round().as_ivec3(),
                entity,
                world_center: Vec2::new(center_world.x, center_world.z),
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
                // Lift the node's scope-local position into a global address.
                if let Some(ctx) = state.root_graph().context_of_node(&node_id) {
                    let owning_graph = state.root_graph().resolve_context(&ctx);
                    if let Some(ln) = owning_graph.layout_nodes.get(&node_id) {
                        let new_pos =
                            ln.pos.round().as_ivec3() + state.root_graph().scope_offset(&ctx);
                        if pick.selected_pos != new_pos {
                            pick.selected_pos = new_pos;
                            rebuild.0 = true;
                        }
                    }
                }
            } else if let Some(hit) = grid_hit {
                if pick.selected_pos != hit.global_pos {
                    pick.selected_pos = hit.global_pos;
                    rebuild.0 = true;
                }
            }
            // Click into truly empty space (no graph grid hit) leaves the
            // selection unchanged.
        }
    }
}

fn highlight_hovered(
    pick: Res<PickState>,
    state: Res<GraphState>,
    node_q: Query<(&NodeEntity, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let selected_node = state
        .caret_graph(&pick)
        .and_then(|(layout, local)| layout.node_at(local));
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
    state: Res<GraphState>,
    eval: Res<EvalState>,
    mut display_q: Query<(&mut Text, &mut TextColor), With<SelectionDisplay>>,
) {
    let Ok((mut text, mut color)) = display_q.single_mut() else {
        return;
    };
    let caret = state.caret_graph(&pick);
    if let Some(id) = caret.and_then(|(layout, local)| layout.node_at(local)) {
        if let Some(node) = caret.and_then(|(layout, _)| layout.graph.nodes.get(&id)) {
            // A `none` says only that no value was produced; the why lives in
            // the run's trace, and this is where the addressed node gets to
            // tell it. It is read from the snapshot on screen, so stepping
            // back drops the line again.
            let reason = match &eval.phase {
                EvalPhase::Running {
                    states, current, ..
                } => states[*current].trace.get(&id),
                _ => None,
            };
            text.0 = format!(
                "{} : {}{}",
                render::label_for_node(node, &state.function_declarations),
                match infer::node_output_type(
                    &state.root_graph().flattened_graph(),
                    &id,
                    &state.function_declarations,
                ) {
                    Some(r#type) => r#type.to_string(),
                    // Sink and Root produce nothing at all.
                    None => "-".to_string(),
                },
                reason
                    .map(|reason| format!("\n{}", reason))
                    .unwrap_or_default()
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

fn update_grid_material(
    pick: Res<PickState>,
    state: Res<GraphState>,
    grid_q: Query<(
        Entity,
        &ScopeGridEntity,
        &MeshMaterial3d<grid::GridMaterial>,
    )>,
    mut materials: ResMut<Assets<grid::GridMaterial>>,
) {
    // The bordered grid is the one the caret addresses.
    let caret_path = state.scope_of_caret(&pick).map(|s| s.path);
    let hit_entity = pick.hovered_grid.as_ref().map(|h| h.entity);
    let hit_center = pick
        .hovered_grid
        .as_ref()
        .map(|h| h.world_center)
        .unwrap_or(Vec2::ZERO);
    for (entity, scope_grid, mat_handle) in grid_q.iter() {
        let Some(mat) = materials.get_mut(&mat_handle.0) else {
            continue;
        };
        if Some(entity) == hit_entity {
            mat.hover_pos = hit_center;
            mat.hover_active = 1.0;
        } else {
            mat.hover_active = 0.0;
        }
        mat.border_active = if Some(scope_grid.context.as_slice()) == caret_path.as_deref() {
            1.0
        } else {
            0.0
        };
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
    mut key_events: MessageReader<KeyboardInput>,
    eval: Res<EvalState>,
) {
    let clicked_outside = mouse.just_pressed(MouseButton::Left);
    let evaluating = is_evaluating(&eval);
    // By the layout's reckoning, not the key's position — see
    // `handle_editor_keys`.
    let escaped = key_events.read().any(|ev| {
        ev.state == bevy::input::ButtonState::Pressed
            && matches!(ev.logical_key, bevy::input::keyboard::Key::Escape)
    });

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

        if escaped {
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
    }
}

/// Detect a change in `PickState::selected_pos` and start a camera
/// auto-focus tween toward the new position. The first observation
/// (fresh `Local`) does not trigger, so app startup doesn't jump.
fn trigger_camera_focus_on_selection_change(
    pick: Res<PickState>,
    orbit: Res<camera::OrbitCamera>,
    mut tween: ResMut<camera::CameraTween>,
    mut last_selection: Local<Option<IVec3>>,
    start_menu: Res<StartMenu>,
) {
    if start_menu.showing {
        return;
    }
    let current = pick.selected_pos;
    if *last_selection != Some(current) {
        if last_selection.is_some() {
            // The caret is already a global address, so it converts to world
            // space directly.
            tween.focus_on(&orbit, render::cell_center_world(current.as_vec3()));
        }
        *last_selection = Some(current);
    }
}

/// True while something other than the graph owns the keyboard: a focused
/// text field, a modal, the start menu, or a running evaluation. Mode
/// switching and the INSERT-mode inserts stay out of the way then — otherwise
/// `i` would both type an `i` and change the mode.
fn keyboard_captured(
    text_inputs: &Query<&TextInput>,
    start_menu: &StartMenu,
    eval: &EvalState,
) -> bool {
    start_menu.showing
        || modal_is_open(eval)
        || is_evaluating(eval)
        || text_inputs.iter().any(|input| input.focused)
}

/// Apply an insert that makes room and let the caret ride it, so it keeps
/// addressing whatever it pointed at. An insert the layout refuses — it would
/// have to cut a node in half — leaves the graph untouched, which is what the
/// `None` stands for.
fn apply_room_insert(
    state: &mut GraphState,
    pick: &mut PickState,
    make: impl FnOnce(&layout::LayoutGraph, &CaretScope) -> Option<layout::LayoutGraph>,
    delta: IVec3,
) -> bool {
    let Some(scope) = state.scope_of_caret(pick) else {
        return false;
    };
    let scope_graph = state.root_graph().resolve_context(&scope.path);
    let Some(new_layout) = make(scope_graph, &scope) else {
        return false;
    };
    if let Some(target) = state.root_graph_mut().resolve_context_mut(&scope.path) {
        *target = new_layout;
    }
    state.resettle();
    pick.selected_pos = state
        .root_graph()
        .clamp_to_volume(pick.selected_pos + delta);
    true
}

/// Everything the keyboard means to the editor itself, in one pass over the
/// batch: the mode switch (`i` enters INSERT, `Esc` returns to NORMAL), and
/// then INSERT's two jobs — the room-makers `Space`, `Return` and
/// `Shift+Return`, which act on the scope the caret addresses and in that
/// scope's local coordinates, and the prompt, into which every other key
/// writes what is to be created: a node kind's name, a function's name, or a
/// literal value. `Space` belongs to the prompt rather than to the room-makers
/// for the length of an unclosed quote, and only then.
///
/// Mode switch and prompt have to be **one** system: the key that enters
/// INSERT is consumed at the very point that flips the mode, so it cannot also
/// be typed into the prompt. Two systems could only approximate that by
/// throwing away a whole frame's batch, losing the first real keystroke
/// whenever a hitch queues it together with the `i`.
///
/// Keys are read from `logical_key`, the key the layout produced, not from
/// `KeyCode`, which names the physical position. A CapsLock remapped to Escape
/// still reports the CapsLock position, so a `KeyCode::Escape` binding would
/// never fire for it — and a command letter like `i` would sit wherever QWERTY
/// puts it, whatever the user actually types. Only Shift, which no message
/// carries, still comes from `ButtonInput`.
///
/// While something else owns the keyboard the batch is dropped rather than
/// left standing: a message survives two updates, so keys struck under the
/// guard would otherwise fire once it lifts — an `Esc` that unfocused a field
/// would leave INSERT on the next frame, and a field's text would land in the
/// prompt.
fn handle_editor_keys(
    mut key_events: MessageReader<KeyboardInput>,
    keys: Res<ButtonInput<KeyCode>>,
    text_inputs: Query<&TextInput>,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    mut mode: ResMut<EditorMode>,
    mut prompt: ResMut<InsertPrompt>,
    mut state: ResMut<GraphState>,
    mut pick: ResMut<PickState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    if keyboard_captured(&text_inputs, &start_menu, &eval) {
        key_events.clear();
        return;
    }
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    for ev in key_events.read() {
        if ev.state != bevy::input::ButtonState::Pressed {
            continue;
        }
        match (*mode, &ev.logical_key) {
            (_, bevy::input::keyboard::Key::Escape) => {
                // Escape leaves INSERT outright and drops what was typed —
                // there is no half-written name worth keeping.
                prompt.clear();
                if *mode != EditorMode::Normal {
                    *mode = EditorMode::Normal;
                    // The caret is drawn per mode, and it is a scene entity.
                    rebuild.0 = true;
                }
            }
            (EditorMode::Normal, bevy::input::keyboard::Key::Character(s)) if s.as_str() == "i" => {
                prompt.clear();
                *mode = EditorMode::Insert;
                rebuild.0 = true;
            }
            // NORMAL's own keys belong to `handle_arrow_keys`.
            (EditorMode::Normal, _) => {}
            (EditorMode::Insert, key) => match key {
                bevy::input::keyboard::Key::Character(s) => {
                    // A single press can carry more than one character when a
                    // dead key resolves; control characters are not a name.
                    if !s.is_empty() && !s.chars().any(|c| c.is_control()) {
                        prompt.text.push_str(s.as_str());
                        let selected =
                            first_selection(&prompt_candidates(&state, &pick, &prompt.text));
                        prompt.selected = selected;
                    }
                }
                bevy::input::keyboard::Key::Backspace => {
                    prompt.text.pop();
                    let selected = first_selection(&prompt_candidates(&state, &pick, &prompt.text));
                    prompt.selected = selected;
                }
                bevy::input::keyboard::Key::ArrowDown | bevy::input::keyboard::Key::ArrowUp => {
                    let delta = if matches!(key, bevy::input::keyboard::Key::ArrowDown) {
                        1
                    } else {
                        -1
                    };
                    let candidates = prompt_candidates(&state, &pick, &prompt.text);
                    prompt.selected = step_selection(&candidates, prompt.selected, delta);
                }
                // Inside an unclosed quote the space bar writes a space —
                // `Key::Space` is its own variant, so the `Character` arm
                // never sees one and a string could otherwise not hold one.
                // The guard needs a non-empty text, the room-maker below an
                // empty one, so the two can never both fire.
                bevy::input::keyboard::Key::Space if in_open_quote(&prompt.text) => {
                    prompt.text.push(' ');
                    let selected = first_selection(&prompt_candidates(&state, &pick, &prompt.text));
                    prompt.selected = selected;
                }
                // The room-makers are bounded to one insert per press: a held
                // key auto-repeats, and the repeats must not each open a cell.
                bevy::input::keyboard::Key::Space if prompt.text.is_empty() && !ev.repeat => {
                    if apply_room_insert(
                        &mut state,
                        &mut pick,
                        |graph, scope| graph.plus_empty_cell(scope.local),
                        IVec3::Z,
                    ) {
                        rebuild.0 = true;
                    }
                    break;
                }
                bevy::input::keyboard::Key::Enter if prompt.text.is_empty() && !ev.repeat => {
                    let inserted = if shift {
                        apply_room_insert(
                            &mut state,
                            &mut pick,
                            |graph, scope| graph.plus_empty_slab(layout::Axis::Y, scope.local.y),
                            IVec3::Y,
                        )
                    } else {
                        apply_room_insert(
                            &mut state,
                            &mut pick,
                            |graph, scope| graph.plus_empty_slab(layout::Axis::X, scope.local.x),
                            IVec3::X,
                        )
                    };
                    if inserted {
                        rebuild.0 = true;
                    }
                    break;
                }
                bevy::input::keyboard::Key::Enter if !ev.repeat => {
                    // Commit what is written. The caret stays on the cell the
                    // node now fills, and INSERT stays on, so the next name
                    // can be typed straight away.
                    let candidates = prompt_candidates(&state, &pick, &prompt.text);
                    let selected = prompt.selected.min(candidates.len().saturating_sub(1));
                    if let Some(suggestion) =
                        candidates.get(selected).filter(|suggestion| suggestion.allowed)
                    {
                        if insert_node_kind(&mut state, &pick, &suggestion.kind) {
                            prompt.clear();
                            rebuild.0 = true;
                        }
                    }
                    break;
                }
                // A space outside an open quote is in no name the list holds,
                // and there is no undo: a stray one mid-typing must neither
                // land in the text nor reshape the graph.
                _ => {}
            },
        }
    }
}

/// Caret navigation in NORMAL: the arrow keys and the vim letters `hjkl`,
/// which are the same four directions under two names.
fn handle_arrow_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut key_events: MessageReader<KeyboardInput>,
    text_inputs: Query<&TextInput>,
    start_menu: Res<StartMenu>,
    mut state: ResMut<GraphState>,
    mut pick: ResMut<PickState>,
    mut rebuild: ResMut<NeedsRebuild>,
    eval: Res<EvalState>,
    mode: Res<EditorMode>,
) {
    // INSERT mode freezes the caret: the keys belong to NORMAL, and moving
    // a node (Ctrl+key) is a NORMAL operation too. The letters are dropped
    // along with the guard, so a key struck under it doesn't fire once it
    // lifts.
    if is_evaluating(&eval) || *mode == EditorMode::Insert {
        key_events.clear();
        return;
    }
    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    // Direction of each key in layout coordinates, named by its vim letter —
    // the arrows carry the same four names. Layout `+Y` renders downward and
    // `+Z` runs source-to-sink, so both are inverted relative to the pre-flip
    // bindings — the keys still move the caret the same way on screen.
    fn caret_delta(dir: char, shift: bool) -> Option<IVec3> {
        match (dir, shift) {
            ('k', false) => Some(IVec3::new(-1, 0, 0)),
            ('k', true) => Some(IVec3::new(0, -1, 0)),
            ('j', false) => Some(IVec3::new(1, 0, 0)),
            ('j', true) => Some(IVec3::new(0, 1, 0)),
            ('h', false) => Some(IVec3::new(0, 0, -1)),
            ('l', false) => Some(IVec3::new(0, 0, 1)),
            _ => None,
        }
    }

    // `hjkl` rides alongside the arrows. Like the mode letters in
    // `handle_editor_keys` they are read from `logical_key`, the character the
    // layout produced, so the binding follows the letter rather than its
    // QWERTY position. A focused field or an open menu swallows them — the
    // arrows keep working there, since nothing else claims them for the
    // caret. Repeats are dropped so a held key steps once, the way
    // `just_pressed` bounds the arrows.
    let letter = if keyboard_captured(&text_inputs, &start_menu, &eval) {
        key_events.clear();
        None
    } else {
        key_events
            .read()
            .filter(|ev| ev.state == bevy::input::ButtonState::Pressed && !ev.repeat)
            .find_map(|ev| match &ev.logical_key {
                bevy::input::keyboard::Key::Character(s) => {
                    let mut chars = s.chars();
                    // Shift reports the uppercase character; the shifted
                    // meaning comes from the modifier, as it does for the
                    // arrows.
                    let c = chars.next()?.to_ascii_lowercase();
                    (chars.next().is_none() && matches!(c, 'h' | 'j' | 'k' | 'l')).then_some(c)
                }
                _ => None,
            })
    };

    let arrow = if keys.just_pressed(KeyCode::ArrowUp) {
        Some('k')
    } else if keys.just_pressed(KeyCode::ArrowDown) {
        Some('j')
    } else if keys.just_pressed(KeyCode::ArrowLeft) {
        Some('h')
    } else if keys.just_pressed(KeyCode::ArrowRight) {
        Some('l')
    } else {
        None
    };

    let delta = arrow.or(letter).and_then(|dir| caret_delta(dir, shift));

    let Some(delta) = delta else {
        return;
    };

    if ctrl {
        // Move the node under the current selection (if any) and keep
        // the selection anchored to it — the effective position may differ
        // from `selected + delta` when the move jumped over a match.
        // The node lives in whichever scope the caret addresses.
        let Some(scope) = state.scope_of_caret(&pick) else {
            return;
        };
        let scope_graph = state.root_graph().resolve_context(&scope.path);
        if let Some(node_id) = scope_graph.node_at(scope.local) {
            let (new_layout, effective_local) =
                scope_graph.move_node_delta(node_id, delta.as_vec3());
            let scope_origin = state.root_graph().scope_offset(&scope.path);
            if let Some(target) = state.root_graph_mut().resolve_context_mut(&scope.path) {
                *target = new_layout;
            }
            state.resettle();
            // `move_node_delta` may report a different cell than
            // `local + delta` when the move jumped a match footprint.
            pick.selected_pos = effective_local + scope_origin;
            rebuild.0 = true;
        }
    } else {
        // Plain move: navigate the selection between grid crossings. The
        // caret stays inside the graph volume — its faces are where the grid
        // ends, and only an explicit action may push them outward — so a move
        // against a face is simply no move at all.
        let target = state
            .root_graph()
            .clamp_to_volume(pick.selected_pos + delta);
        if target != pick.selected_pos {
            pick.selected_pos = target;
            rebuild.0 = true;
        }
    }
}

fn anchor_hover_system(
    mut commands: Commands,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    anchors: Query<(Entity, &GlobalTransform), With<EAnchor>>,
    existing_hovers: Query<Entity, With<AnchorHovered>>,
    ui_interactions: Query<&Interaction, With<Button>>,
) {
    // Alle vorherigen Hovers entfernen
    for e in &existing_hovers {
        commands.entity(e).remove::<AnchorHovered>();
    }

    // Anchor-Hover ist rein screen-space (Distanz zum projizierten Anchor) und
    // weiß nichts von davorliegenden UI-Panels. Ohne diesen Guard startet ein
    // Klick auf einen Button/eine Dropdown-Option einen Drag, sobald zufällig
    // ein Anchor in Cursor-Nähe projiziert wird. Gleiches Muster wie in
    // `pick_nodes`.
    let over_ui = ui_interactions
        .iter()
        .any(|i| matches!(*i, Interaction::Hovered | Interaction::Pressed));
    if over_ui {
        return;
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
        // Same reason as in `update_world_labels`: under the orthographic
        // projection a point behind the camera still projects, so it would
        // otherwise become a hover target.
        if cam_tf
            .forward()
            .dot(global_tf.translation() - cam_tf.translation())
            <= 0.0
        {
            continue;
        }
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

fn draw_drag_preview(drag: Res<DragState>, mut gizmos: Gizmos) {
    let Some(ref info) = drag.active else { return };
    let color = if info.target_anchor_id.is_some() {
        Color::srgb(0.3, 1.0, 0.4) // grün = eingeschnappt
    } else {
        Color::srgb(1.0, 0.9, 0.3) // gelb = dragging
    };
    gizmos.line(info.source_pos, info.current_end, color);
}

fn drag_start_system(
    mouse: Res<ButtonInput<MouseButton>>,
    hovered: Query<(&GlobalTransform, &EAnchor), With<AnchorHovered>>,
    mut drag: ResMut<DragState>,
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    if mouse.just_pressed(MouseButton::Left) {
        if let Ok((tf, anchor)) = hovered.single() {
            let pos = tf.translation();
            drag.active = Some(DragInfo {
                source_anchor_id: anchor.id(),
                source_is_output: matches!(anchor, EAnchor::Output { .. }),
                source_pos: pos,
                current_end: pos,
                target_anchor_id: None,
            });
        }
    }
}

fn drag_update_system(
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    hovered: Query<(&GlobalTransform, &EAnchor), With<AnchorHovered>>,
    mut drag: ResMut<DragState>,
    eval: Res<EvalState>,
    state: Res<GraphState>,
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

    // Snap zu hovering target. Muss jeden Frame zurückgesetzt werden — sonst
    // hält drag_end an einem längst verlassenen Ziel fest und legt beim Drop
    // ins Leere trotzdem eine Edge an.
    info.target_anchor_id = None;
    if let Ok((tf, anchor)) = hovered.single() {
        let target_id = anchor.id();
        // Vergleich über AnchorId, nicht Entity: Entities werden bei jedem
        // Rebuild neu gespawnt, ein Entity-Vergleich würde den Quell-Anchor
        // nach einem Rebuild mitten im Drag als gültiges Ziel durchlassen und
        // eine Self-Edge erzeugen.
        let is_self = target_id == info.source_anchor_id;
        // Nur output → input (oder umgekehrt) verbinden.
        let is_opposite_kind = matches!(anchor, EAnchor::Output { .. }) != info.source_is_output;
        // Schon verbundene Paare snappen nicht ein, damit die Preview-Linie
        // gelb bleibt statt eine Verbindung zu versprechen, die drag_end
        // ohnehin als Duplikat verwirft.
        let is_duplicate =
            anchors_already_connected(state.root_graph(), &info.source_anchor_id, &target_id);
        if !is_self && is_opposite_kind && !is_duplicate {
            info.target_anchor_id = Some(target_id);
            info.current_end = tf.translation();
        }
    }
}

/// True if `a` and `b` are already joined by an edge, in either stored
/// direction.
///
/// `TermGraph::plus_edge` appends unconditionally, so without this guard reconnecting
/// the same pair stacks a second, perfectly coincident ribbon on the first —
/// invisible until one of them is deleted. The reverse direction is checked too
/// because edges recorded before drag-end started normalising to output → input
/// may still sit the other way around, and `eval::neighbours_of_anchor` treats
/// both orientations as connected.
fn anchors_already_connected(
    layout_graph: &layout::LayoutGraph,
    a: &model::anchor::Id,
    b: &model::anchor::Id,
) -> bool {
    let joined = |from: &model::anchor::Id, to: &model::anchor::Id| {
        layout_graph
            .graph
            .edges
            .get(from)
            .is_some_and(|edges| edges.iter().any(|e| e.to == *to))
    };
    joined(a, b) || joined(b, a)
}

fn drag_end_system(
    mouse: Res<ButtonInput<MouseButton>>,
    mut drag: ResMut<DragState>,
    mut commands: Commands,
    mut rebuild: ResMut<NeedsRebuild>,
    mut state: ResMut<GraphState>,
    eval: Res<EvalState>,
) {
    if is_evaluating(&eval) {
        drag.active = None;
        return;
    }
    if mouse.just_released(MouseButton::Left) {
        if let Some(info) = drag.active.take() {
            if let Some(target_id) = info.target_anchor_id {
                // Edges immer output → input speichern: EdgeCurve::from_endpoints
                // leitet die Tangenten aus dieser Richtung ab, und der Renderer
                // stapelt die Leaves nach Quelle/Ziel.
                let (from, to) = if info.source_is_output {
                    (info.source_anchor_id, target_id)
                } else {
                    (target_id, info.source_anchor_id)
                };
                // Defensiv: eine Self-Edge kollabiert die Kurve zu einer
                // Schlaufe am Anchor. drag_update lässt das nicht zu, aber die
                // Invariante hier nochmal festnageln.
                if from != to && !anchors_already_connected(state.root_graph(), &from, &to) {
                    let updated = state.root_graph().plus_edge(from, to);
                    *state.root_graph_mut() = updated;
                    // A new edge can grow the target's anchor, which changes
                    // its footprint.
                    state.resettle();
                    rebuild.0 = true;
                }
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
                    title: "Expression Visualizer 3D".into(),
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
        .init_resource::<GraphState>()
        .init_resource::<NeedsRebuild>()
        .init_resource::<PickState>()
        .init_resource::<DragState>()
        .init_resource::<EvalState>()
        .init_resource::<StartMenu>()
        .init_resource::<DropdownState>()
        .init_resource::<EditorMode>()
        .init_resource::<InsertPrompt>()
        .add_systems(
            Startup,
            (
                load_ui_font,
                setup_scene,
                spawn_graph_nodes,
                spawn_ui,
                spawn_selection_display,
                spawn_node_editor_panel,
                spawn_insert_prompt_panel,
                spawn_fps_display,
                spawn_breadcrumb_display,
                spawn_mode_display,
                spawn_start_menu,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                (
                    draw_drag_preview,
                    animate_nodes,
                    (
                        handle_delete_node_button,
                        handle_add_button,
                        handle_insert_prompt_click,
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
                    update_grid_material,
                    update_cursor,
                    // Editor keys before the text field: while a field has
                    // focus, Escape belongs to it, and `text_input_focus`
                    // clears that focus in the same frame.
                    (handle_editor_keys, text_input_focus, text_input_keyboard).chain(),
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
                update_fps_display,
                update_breadcrumb_display,
                update_mode_display,
                blink_caret,
            ),
        )
        .add_systems(
            Update,
            (
                handle_evaluate_button,
                handle_camera_mode_button,
                sync_camera_mode_button,
                handle_semi_ortho_checkbox,
                sync_semi_ortho_checkbox,
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
                sync_add_button,
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
                sync_insert_prompt_ui,
            )
                .chain(),
        )
        .run();
}
