mod camera;
mod common;
mod depth_cue;
mod edge;
mod eval;
mod grid;
mod infer;
mod layout;
mod lint;
mod lod;
mod mesh;
mod model;
mod render;

use bevy::core_pipeline::oit::OrderIndependentTransparencySettings;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::{
    input::keyboard::KeyboardInput,
    input::mouse::{MouseMotion, MouseWheel},
    math::VectorSpace,
    prelude::*,
};

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

/// Everything the editor draws about itself is off, and what is left standing
/// is the program. It takes no picture of its own — the screen is simply clear
/// for whatever does.
///
/// It ends at the first sign of life, and the input that ends it also does
/// whatever it normally would: leaving is not an act of its own, it is what
/// *using the editor again* means.
#[derive(Resource, Default)]
struct ScreenshotMode {
    /// `None` while it is off; otherwise when it was entered.
    ///
    /// The time is kept rather than a bare flag because of how it is entered:
    /// with the mouse, on a button. Without a moment's grace the very movement
    /// that carries the hand off that button would end it again, and the mode
    /// would be unreachable by the one gesture that reaches it.
    entered_at: Option<f32>,
}

impl ScreenshotMode {
    fn active(&self) -> bool {
        self.entered_at.is_some()
    }

    /// Whether input should be listened to yet.
    fn listening(&self, now: f32) -> bool {
        self.entered_at
            .is_some_and(|entered| now - entered >= SCREENSHOT_GRACE_SECONDS)
    }
}

/// How long the screenshot mode ignores input after being entered. Long enough
/// for a hand to leave the mouse, short enough that nobody waits for it.
const SCREENSHOT_GRACE_SECONDS: f32 = 0.5;

/// What has been typed in INSERT so far, and which suggestion it stands on.
/// One resource for every prompt — the one that creates and the one that edits
/// whichever property the caret's cell names — because the caret decides which
/// of them is live, so no two can be live at once and a second resource would
/// only add the question of which one to believe.
///
/// Deliberately **not** a `TextInput`: a focused text field makes
/// `keyboard_captured` true, and that is what gives `Space`, `Return` and
/// `Escape` their INSERT meanings. A field here would turn `Space` into a
/// blank and `Escape` into a mere unfocus — the three behaviours that have to
/// survive. So it grows the parts of a text field that are worth having
/// instead: `cursor` is a byte offset into `text`, and `Left`/`Right`/`Home`/
/// `End`/`Delete` move and edit at it. They are affordable because the list
/// only ever claimed `Up`/`Down`, and they are *needed* because the prompt now
/// opens on the value that is already there — without a cursor, correcting one
/// character of `charAt` would mean deleting back to it.
///
/// `selected` indexes the *candidate* list — not the window of it that fits on
/// screen, which is derived from `selected` rather than the other way round —
/// and is re-clamped on every rebuild rather than trusted.
///
/// It is an `Option` because "no row is the answer yet" is a state the prompt
/// really has, and one the list has to show: while nothing is typed on a
/// *create* prompt, `Return` still means the editor's newline, so a highlight
/// standing there would promise a commit the key does not perform.
#[derive(Resource, Default)]
struct InsertPrompt {
    text: String,
    /// Byte offset into `text`, always on a character boundary.
    cursor: usize,
    selected: Option<usize>,
}

impl InsertPrompt {
    fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.selected = None;
    }

    /// Replace the text wholesale and put the cursor behind it — what opening
    /// the prompt on an existing value does. The caller sets `selected`, which
    /// depends on candidates this type knows nothing about.
    fn set_text(&mut self, text: String) {
        self.cursor = text.len();
        self.text = text;
    }

    fn insert_str(&mut self, s: &str) {
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    /// Byte offset of the character boundary one step before the cursor, and
    /// one step after it. `None` at the respective end.
    fn prev_boundary(&self) -> Option<usize> {
        self.text[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(index, _)| index)
    }

    fn next_boundary(&self) -> Option<usize> {
        self.text[self.cursor..]
            .chars()
            .next()
            .map(|c| self.cursor + c.len_utf8())
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
/// The pair is also what `PromptAction::SetType` carries, for the same reason
/// and into the same `make_etype`.
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
    Tunnel,
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
/// `mod`, `max` and `min` — but a typed letter puts the highlight on the first
/// committable row, so `m` still settles on `Match`. (The *empty* prompt
/// highlights nothing at all; the letter is what starts the list answering.)
const NODE_KINDS: [(AddKind, &str); 5] = [
    (AddKind::Source, "Source"),
    (AddKind::Match, "Match"),
    (AddKind::Pattern, "Pattern"),
    (AddKind::TypeCast, "TypeCast"),
    (AddKind::Tunnel, "Tunnel"),
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
    (model::r#type::NONE_LITERAL, TypeChoice::None, None),
];

/// What committing a prompt row does. The prompt is one widget with as many
/// jobs as there are editable cells, because INSERT means whatever the caret's
/// cell is: on a free cell it builds a node, on a property cell it changes that
/// property.
///
/// One type rather than one prompt per job is what keeps the rest small — the
/// typing, the highlight, the window and the click path never learn which job
/// is on. Which node each setter acts on comes from the caret, not from the
/// row, for the same reason `Create` does: a row can never point at something
/// that has moved on since it was drawn.
#[derive(Clone, PartialEq, Eq)]
enum PromptAction {
    Create(AddKind),
    SetName(String),
    /// A type and, where it is a literal, that literal's text. One action for
    /// four properties — a Source's declared type (never a literal), a
    /// Constant's value (always one, bar `none`), a TypeCast's target and a
    /// Pattern's arm — because all four are the same question asked of
    /// different cells.
    SetType(TypeChoice, Option<String>),
}

/// A node the prompt has built whose one mandatory property has not been given
/// yet.
///
/// It stands in the graph while it is being answered, so what is being built
/// can be seen — but it is not finished, and Escape unmakes it rather than
/// leaving behind a placeholder nobody chose. Five kinds need this: a
/// `TypeCast`, a `Pattern`, a `Match`, a `Source` and a `Tunnel`, whose
/// property is not part of the name that was typed to create it. A `Constant`
/// carries its literal and a `FunctionCall` its function already, so both
/// arrive finished.
///
/// A Source's *name* is not that property — it may be empty, and a Source is
/// told apart by its index. Its declared type is, and a Tunnel's is for the
/// same reason: every value that arrives at one from outside has to be a value
/// of something, so neither may stand there declaring nothing.
#[derive(Clone, PartialEq, Eq)]
struct PendingEdit {
    /// Removed whole on Escape. For a Match this is its one `Pattern`, not the
    /// Match: `minus_node` takes a parent Match with its last Pattern, so
    /// naming the Pattern removes both — and the Pattern is what the caret
    /// stands on anyway.
    node: model::node::Id,
    /// Where the caret goes back to, so a cancelled insert leaves no trace.
    caret_before: IVec3,
}

#[derive(Resource, Default)]
struct PendingNode(Option<PendingEdit>);

/// What building a node left behind.
enum Inserted {
    /// The node is finished; nothing is owed.
    Done,
    /// The caret has been moved onto the cell that names the missing property,
    /// and INSERT stays on to collect it.
    Pending(PendingEdit),
}

/// One row of the INSERT prompt: what it would do, what it is called, and the
/// muted right column that tells `charAt` from `concat` before the node
/// exists.
///
/// It owns its strings instead of borrowing the catalogue on purpose: the list
/// is built from a `&GraphState` and the picked row then goes to
/// `apply_prompt_action(&mut state, …)`, which a borrowed declaration in here
/// would keep from compiling.
#[derive(Clone, PartialEq, Eq)]
struct Suggestion {
    action: PromptAction,
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
/// Asks for a fresh graph. What it opens is the confirmation, not the reset —
/// the reset itself is the only editor action that cannot be undone, so it is
/// never one click away.
#[derive(Component)]
struct NewButton;

/// Opens the controls list. Nothing is destroyed on the way, so it acts at once.
#[derive(Component)]
struct HelpButton;

/// A clickable suggestion row, so the mouse path into INSERT does not dead-end
/// at a keyboard-only list.
#[derive(Component)]
struct InsertPromptOption(PromptAction);

/// Marker for everything the editor draws about *itself* — buttons, panels, the
/// HUD readouts — as opposed to what it draws about the program.
///
/// Carrying it is a standing offer to be taken off screen whenever the editor
/// has nothing to say: while a modal owns the screen, and in screenshot mode,
/// where the point is that only the program is left. One system writes their
/// `display`, and a widget that manages its own must not also wear this — two
/// writers flicker on the transition frame. For the same reason it goes on the
/// container of a group rather than on each member.
#[derive(Component)]
struct EditorChrome;

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

/// The scope a walked node stands in, written the way every scope path is:
/// relative to the root graph, empty at the program's own scope.
///
/// `LayoutGraph::walk_all` is started one level further out than that — at the
/// shell holding the Root node — so its contexts carry the root id in front.
/// Scope paths from `scope_at` do not, and the two have to be comparable or the
/// level of detail would grade every volume against the wrong one.
fn scope_of_walk<'a>(state: &GraphState, context: &'a [model::node::Id]) -> &'a [model::node::Id] {
    match context.split_first() {
        Some((first, rest)) if *first == state.root_id => rest,
        _ => &[],
    }
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

/// The node the caret addresses and which of its cells. Purely mechanical —
/// whether that cell *names* anything is `insert_target`'s question, and asking
/// it in one place is what keeps the panel and INSERT mode from disagreeing
/// about which cell edits what.
fn addressed_cell(
    state: &GraphState,
    pick: &PickState,
) -> Option<(model::node::Id, layout::CellRole)> {
    let (layout, local) = state.caret_graph(pick)?;
    let id = layout.node_at(local)?;
    let ln = layout.layout_nodes.get(&id)?;
    let role = ln.shape.role_at(local - ln.pos.round().as_ivec3())?.clone();
    Some((id, role))
}

/// Which property of a node one of its cells stands for: one variant per
/// (kind, cell) pair, and none for a cell that names nothing.
///
/// This is the whole editing model. A cell names at most one property, so
/// standing on it and choosing what to change are one act. The panel does show
/// a node's other properties beside the addressed one, but only to read: which
/// one an edit acts on is still the caret's answer and nothing else's.
#[derive(Clone, PartialEq, Eq)]
enum EditTarget {
    /// Printed along the body, so the body is where it is typed.
    SourceName,
    /// Declared on the Source's first cell, in front of the name — the cell of
    /// its own that every declared type gets, a TypeCast's and a Pattern's
    /// included. It used to hang off the output anchor; an anchor is where an
    /// edge begins and has no room for a second job.
    SourceType,
    /// A Constant's body cell. The value *is* the node, so the type is not
    /// carried separately — it is whatever the literal spells.
    ///
    /// Its output anchor used to answer for the literal too, saying the same
    /// thing twice. Only the body says it now, and the anchor is free for the
    /// edge that starts there.
    ConstantValue,
    /// The cell between a TypeCast's two anchors: a base type to cast to, or a
    /// literal to produce. Its anchors name nothing.
    CastType,
    /// The type a Pattern's arm matches, on the Pattern's one cell.
    PatternType,
    /// The type a Tunnel lets through, on the cell between its two anchors —
    /// the only one of the three that is a cell of the branch at all, its
    /// input being outside the volume and its output an anchor like any other.
    TunnelType,
    // A FunctionCall is deliberately absent. Which function a call calls is
    // fixed when it is built: a different function has a different arity, so
    // re-pointing a call is re-wiring it, and every edge into it would have to
    // go somewhere. Rather than answer that, the question is not asked — the
    // call is deleted and another built. Its body cells therefore name nothing.
}

/// What INSERT mode does at the caret. The mode is one key, but it means
/// whatever the addressed cell is.
///
/// `Create` is what a cell that names no property falls back to — an empty cell,
/// an anchor row, a hole in a multi-column node. On an occupied cell that is
/// already inert, because `kind_allowed` greys every row there; what it keeps
/// alive is the room-makers, which are the reason to stand on such a cell.
#[derive(Clone, PartialEq, Eq)]
enum InsertTarget {
    /// Build something: the prompt offers node kinds, functions and literals.
    Create,
    /// Change something that already stands there.
    Edit(model::node::Id, EditTarget),
}

/// Which property one cell of a node stands for.
///
/// Every row here names a `Body` or a `Name` cell, and none of them an anchor.
/// That is the rule rather than how it happens to have come out: an anchor is
/// where an edge begins, so it cannot also be where a property is answered, and
/// a cell that names one property is what makes standing on it and choosing
/// what to change the same act.
///
/// Out here rather than inside `insert_target` because the panel asks it of
/// every cell a node has, not only of the one the caret stands on.
fn edit_target_of(node: &model::node::ENode, role: &layout::CellRole) -> Option<EditTarget> {
    match (node, role) {
        (model::node::ENode::Source { .. }, layout::CellRole::Body) => Some(EditTarget::SourceType),
        (model::node::ENode::Source { .. }, layout::CellRole::Name) => Some(EditTarget::SourceName),
        (model::node::ENode::Constant { .. }, layout::CellRole::Body) => {
            Some(EditTarget::ConstantValue)
        }
        (model::node::ENode::TypeCast { .. }, layout::CellRole::Body) => Some(EditTarget::CastType),
        (model::node::ENode::Pattern { .. }, layout::CellRole::Body) => {
            Some(EditTarget::PatternType)
        }
        (model::node::ENode::Tunnel { .. }, layout::CellRole::Body) => Some(EditTarget::TunnelType),
        // A FunctionCall's body cells fall through with everything else: the
        // function is not changeable, so they name nothing. See `EditTarget`.
        _ => None,
    }
}

fn insert_target(state: &GraphState, pick: &PickState) -> InsertTarget {
    let Some((id, role)) = addressed_cell(state, pick) else {
        return InsertTarget::Create;
    };
    let Some((layout, _)) = state.caret_graph(pick) else {
        return InsertTarget::Create;
    };
    let Some(node) = layout.graph.nodes.get(&id) else {
        return InsertTarget::Create;
    };
    match edit_target_of(node, &role) {
        Some(property) => InsertTarget::Edit(id, property),
        None => InsertTarget::Create,
    }
}

/// The property the target names, spelled the way it would be typed — which is
/// what the prompt opens on, so that changing a value starts from the value.
///
/// `EType`'s own `Display` is exactly that spelling already: `Integer` for a
/// bare type, `42`, `'a'`, `"word"` for a literal.
fn property_text(state: &GraphState, node_id: &model::node::Id, target: &EditTarget) -> String {
    let Some(node) = state
        .layout_graph
        .find_node_graph(node_id)
        .and_then(|graph| graph.graph.nodes.get(node_id))
    else {
        return String::new();
    };
    match (target, node) {
        (EditTarget::SourceName, model::node::ENode::Source { name, .. }) => name.clone(),
        // The type's name and never a literal's spelling: what a Source and a
        // Tunnel declare is a base type, so that is what the prompt opens on.
        // Nothing declared yet is nothing to open on, the same as below.
        (EditTarget::SourceType, model::node::ENode::Source { r#type, .. })
        | (EditTarget::TunnelType, model::node::ENode::Tunnel { r#type, .. }) => r#type
            .as_ref()
            .map(|t| type_choice_label(type_choice_of(t)).to_string())
            .unwrap_or_default(),
        (EditTarget::ConstantValue, model::node::ENode::Constant { r#type, .. }) => {
            r#type.to_string()
        }
        (EditTarget::CastType, model::node::ENode::TypeCast { r#type, .. })
        | (EditTarget::PatternType, model::node::ENode::Pattern { r#type, .. }) => r#type
            .as_ref()
            .map(ToString::to_string)
            // Nothing chosen yet is nothing to open on, so the prompt starts
            // empty and the first keystroke is the whole answer.
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// What a row of the panel addresses — and what the caret is turned into, so
/// that finding the row it stands on is a comparison and not a second decision.
#[derive(Clone, PartialEq, Eq)]
enum RowAddress {
    /// A property of a node: the pair `insert_target` answers with.
    Property(model::node::Id, EditTarget),
    /// The cell a new arm would be built at — the gap cell of the arm it would
    /// go below. Named by its cell, because there is no node there yet.
    ArmGap(IVec3),
}

/// One row of the panel: what it says, and the cell the caret stands on to
/// address it. The two are the same address seen from two sides, which is what
/// lets `TAB` walk the rows by moving the caret.
#[derive(Clone, PartialEq, Eq)]
struct PanelRow {
    address: RowAddress,
    /// Global. This is where `TAB` puts the caret.
    cell: IVec3,
    /// Left column. A gap row has none — it is a line, not a property.
    label: String,
    /// Right column. A gap row has none.
    value: String,
}

/// What the panel is about: one node, and everything about it that can be
/// answered.
#[derive(Clone, PartialEq, Eq)]
struct PanelSubject {
    heading: String,
    rows: Vec<PanelRow>,
}

/// What the panel calls the node it is about.
///
/// The kind and not the node: what a Source is called is its `Name` row's
/// business, and saying it twice would make the heading jump while the row is
/// being typed into. A call is the exception that proves it — which function it
/// calls is the one thing about a call that cannot be changed, so its name is a
/// fact about the kind rather than a property.
fn panel_heading(state: &GraphState, node: &model::node::ENode) -> String {
    match node {
        model::node::ENode::Source { .. } => "Source".to_string(),
        model::node::ENode::Constant { .. } => "Constant".to_string(),
        model::node::ENode::TypeCast { .. } => "TypeCast".to_string(),
        model::node::ENode::Tunnel { .. } => "Tunnel".to_string(),
        model::node::ENode::Match { .. } => "Match".to_string(),
        model::node::ENode::Pattern { .. } => "Pattern".to_string(),
        model::node::ENode::Sink { .. } => "Sink".to_string(),
        model::node::ENode::BranchSource { .. } => "Branch source".to_string(),
        model::node::ENode::Root {} => "Root".to_string(),
        model::node::ENode::FunctionCall {
            function_declaration_id,
            ..
        } => state
            .function_declarations
            .get(function_declaration_id)
            .map(|declaration| declaration.name.clone())
            .unwrap_or_else(|| "Call".to_string()),
    }
}

/// What the panel calls a property. One word, because the node's own name is
/// already the heading above it.
fn property_label(target: &EditTarget) -> &'static str {
    match target {
        EditTarget::SourceName => "Name",
        EditTarget::SourceType
        | EditTarget::CastType
        | EditTarget::PatternType
        | EditTarget::TunnelType => "Type",
        EditTarget::ConstantValue => "Value",
    }
}

/// The caret, said the way a row says it.
fn caret_row_address(state: &GraphState, pick: &PickState) -> RowAddress {
    match insert_target(state, pick) {
        InsertTarget::Edit(id, target) => RowAddress::Property(id, target),
        InsertTarget::Create => RowAddress::ArmGap(pick.selected_pos),
    }
}

/// What the caret's cell makes the panel about: the node it stands on and all
/// of that node's properties — or, where it stands on an arm of a Match, the
/// Match and all of its arms, because a Match declares nothing of its own and
/// its arms are the whole of what there is to say about it.
///
/// `None` where the caret stands on no node at all. There is nothing to show
/// there but the create prompt, and the create prompt is about no node.
///
/// One function and two readers: the panel draws these rows and `TAB` walks
/// them. Were there two lists, `TAB` would land where no row stands.
fn panel_subject(state: &GraphState, pick: &PickState) -> Option<PanelSubject> {
    let scope = state.scope_of_caret(pick)?;
    let root = state.root_graph();
    let graph = root.resolve_context(&scope.path);
    let origin = root.scope_offset(&scope.path);
    // A Pattern's gap cell belongs to the Pattern, so the caret in the space
    // between two arms lands here too — and is answered with its Match.
    let id = graph.node_at(scope.local)?;
    let node = graph.graph.nodes.get(&id)?;
    match node {
        model::node::ENode::Pattern { parent_match, .. } => {
            match_subject(state, graph, origin, parent_match)
        }
        _ => node_subject(state, graph, origin, &id, node),
    }
}

/// A node and its properties, in the order its cells run — the order the caret
/// walks them in with `l`, so that `TAB` and the arrow keys agree about what
/// comes next.
fn node_subject(
    state: &GraphState,
    graph: &layout::LayoutGraph,
    origin: IVec3,
    id: &model::node::Id,
    node: &model::node::ENode,
) -> Option<PanelSubject> {
    let ln = graph.layout_nodes.get(id)?;
    let node_origin = ln.pos.round().as_ivec3() + origin;
    let mut rows: Vec<PanelRow> = Vec::new();
    let mut seen: Vec<EditTarget> = Vec::new();
    for (local, role) in ln.shape.cells() {
        let Some(target) = edit_target_of(node, role) else {
            continue;
        };
        // A Source's name runs across as many cells as it needs, every one of
        // them a `Name`. They answer one question between them, so the first is
        // the row and the rest are that same row.
        if seen.contains(&target) {
            continue;
        }
        seen.push(target.clone());
        rows.push(PanelRow {
            label: property_label(&target).to_string(),
            value: property_text(state, id, &target),
            address: RowAddress::Property(id.clone(), target),
            cell: *local + node_origin,
        });
    }
    Some(PanelSubject {
        heading: panel_heading(state, node),
        rows,
    })
}

/// A Match and its arms, each followed by the place the next one would go.
///
/// The order is `ENode::Match.patterns` and nothing else — the layout hands out
/// the rows from that list, so what stands higher on screen is what is asked
/// first, and the panel says the same.
///
/// That the gap cell of arm `i` is the place between `i` and `i + 1` is not an
/// arrangement of the panel's own: `kind_allowed` admits a Pattern only on a
/// cell belonging to a Pattern, and `plus_pattern_below` puts what is built
/// there directly below that arm. Which is also why there is no place above the
/// first arm — there is no arm to be below.
fn match_subject(
    state: &GraphState,
    graph: &layout::LayoutGraph,
    origin: IVec3,
    match_id: &model::node::Id,
) -> Option<PanelSubject> {
    let model::node::ENode::Match { patterns, .. } = graph.graph.nodes.get(match_id)? else {
        return None;
    };
    let mut rows: Vec<PanelRow> = Vec::new();
    for (index, pattern_id) in patterns.iter().enumerate() {
        let Some(ln) = graph.layout_nodes.get(pattern_id) else {
            continue;
        };
        let pattern_origin = ln.pos.round().as_ivec3() + origin;
        let cell_of = |wanted: layout::CellRole| {
            ln.shape
                .cells()
                .iter()
                .find(|(_, role)| *role == wanted)
                .map(|(local, _)| *local + pattern_origin)
        };
        if let Some(cell) = cell_of(layout::CellRole::Body) {
            rows.push(PanelRow {
                address: RowAddress::Property(pattern_id.clone(), EditTarget::PatternType),
                cell,
                label: format!("Pattern {}", index + 1),
                value: property_text(state, pattern_id, &EditTarget::PatternType),
            });
        }
        if let Some(cell) = cell_of(layout::CellRole::Gap) {
            rows.push(PanelRow {
                address: RowAddress::ArmGap(cell),
                cell,
                label: String::new(),
                value: String::new(),
            });
        }
    }
    Some(PanelSubject {
        heading: "Match".to_string(),
        rows,
    })
}

/// The cell of the row `TAB` moves to, wrapping at both ends. Standing on no
/// row at all, the first one is next — that is the way into a panel the caret
/// is beside rather than on.
fn tab_step(subject: &PanelSubject, here: &RowAddress, back: bool) -> Option<IVec3> {
    let len = subject.rows.len();
    if len == 0 {
        return None;
    }
    let next = match subject.rows.iter().position(|row| &row.address == here) {
        Some(index) if back => (index + len - 1) % len,
        Some(index) => (index + 1) % len,
        None => 0,
    };
    Some(subject.rows[next].cell)
}

/// Marker for the FPS counter text in the top-right corner.
#[derive(Component)]
struct FpsDisplay;

/// Which of the two lines above the panel a text node is.
///
/// One component with two values rather than two markers, because one system
/// writes both: the relative line and the absolute one are the same walk of
/// the caret's scope path, read twice.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum AddressLine {
    /// The way to the caret as a chain of offsets, each measured in the frame
    /// its segment names. The chain sums to `Absolute` — that is what makes it
    /// readable as a path rather than as a list of numbers.
    Relative,
    /// Where the caret stands, full stop.
    Absolute,
}

/// One half of the mode control on the bottom edge: the mode a click on it
/// asks for. Which half is on is `EditorMode`'s answer, never the button's.
#[derive(Component)]
struct ModeToggle(EditorMode);

/// What stands in the mode control's place while an evaluation runs, when the
/// editing mode is nobody's to set.
#[derive(Component)]
struct EvalModeLabel;

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
    /// The question before a New. It carries nothing: the reset reads the
    /// graph state when it runs, and the only thing this phase remembers is
    /// that the question is on screen.
    ConfirmNew,
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
    /// A standing wish to reach the end of the run, rather than a phase of it.
    ///
    /// Its own field because the wish has to survive the way a run *starts*: a
    /// graph with Sources goes through the values modal first, and the run only
    /// begins when that is answered. Asking for the end is therefore said at one
    /// moment and carried out at another, and between the two there is nothing
    /// in `phase` that could remember it.
    ///
    /// Cleared by `apply_run_to_end` — on carrying it out, and on any phase that
    /// is neither running nor on its way to running, so a cancelled modal leaves
    /// no wish lying in wait for the next run.
    run_to_end: bool,
}

impl Default for EvalState {
    fn default() -> Self {
        Self {
            phase: EvalPhase::Idle,
            run_to_end: false,
        }
    }
}

fn is_evaluating(eval: &EvalState) -> bool {
    !matches!(eval.phase, EvalPhase::Idle)
}

fn modal_is_open(eval: &EvalState) -> bool {
    matches!(
        eval.phase,
        EvalPhase::ErrorModal(_)
            | EvalPhase::ControlsModal
            | EvalPhase::ConfirmNew
            | EvalPhase::SourcePrompt { .. }
    )
}

#[derive(Component)]
struct EvaluateButton;

/// The standing container the evaluation controls live in, centred on the
/// bottom edge. It never despawns, so exactly one system writes its `display`
/// and its contents can be rebuilt underneath without that question arising.
#[derive(Component)]
struct PlayerControls;

/// Tags the buttons and the counter inside `PlayerControls`, which are
/// respawned whenever the phase changes.
#[derive(Component)]
struct PlayerControlsEntity;

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

/// The affirmative half of the New question. Its own marker because it is the
/// only modal button that does something: `ModalCancelButton` beside it, and
/// `ModalOkButton` everywhere else, do no more than close what they are on.
#[derive(Component)]
struct ConfirmNewButton;

/// Marker on a TextInputBox inside the Source modal so we can collect
/// typed values per Source when the user confirms.
#[derive(Component)]
struct ModalSourceInput {
    node_id: model::node::Id,
}

#[derive(Component)]
struct PrevStepButton;
#[derive(Component)]
struct NextStepButton;
#[derive(Component)]
struct ExitEvaluationButton;

/// Takes the run to its end in one press — the same steps `NextStepButton`
/// takes, taken until there are none left.
#[derive(Component)]
struct FullRunButton;

/// The step the bar is standing on, written out beside the buttons that move
/// it. A snapshot carries no number of its own — `current` indexes the history
/// — so this is the one place the run says where in itself it is.
#[derive(Component)]
struct StepCounterText;

/// World-space text node showing a node's current evaluated value.
#[derive(Component)]
struct ValueLabel {
    node_id: model::node::Id,
}

// ── Editor panel ────────────────────────────────────────────

/// The left column: where the caret stands, and then the panel about what it
/// stands on.
///
/// A column and not three absolutely-placed nodes, because the relative line
/// grows with the caret's depth and will fold once it runs out of screen — and
/// a panel at a fixed `top` cannot answer that. Here it simply flows under
/// however many lines there turned out to be, and takes no space at all on the
/// frames it is not shown.
///
/// The column wears `EditorChrome` and the panel inside it must not: the panel
/// writes its own `display`, and being a child is enough to disappear with the
/// rest of the chrome.
#[derive(Component)]
struct EditorColumn;

/// The one panel on the left edge: the node the caret stands on, and what can
/// be answered about it. It never despawns — `sync_editor_panel` rebuilds its
/// contents and writes its `display`.
#[derive(Component)]
struct EditorPanel;

/// Tag on each row the panel respawns — the heading, a property's row, an
/// arm's gap, a loose prompt. Only those carry it, never anything inside one:
/// `despawn` takes a row's children with it, and an entity despawned twice
/// warns.
#[derive(Component, Clone)]
struct EditorPanelEntity;

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

/// What the OIT layer budget actually is on this machine, said once at startup.
///
/// The layer count is not a free choice: the layers buffer is one `vec2<u32>`
/// per pixel per layer over the whole viewport, bound whole, so it is measured
/// against `max_storage_buffer_binding_size` — and that number is the adapter's,
/// not wgpu's 128 MiB default. Bevy's default priority is `Functionality`, and
/// `initialize_renderer` throws the requested limits away and asks the adapter
/// for its own, so what is in force here is whatever this GPU reports. Which is
/// exactly why it is worth printing rather than reasoning about: raising
/// `layer_count` is a one-line change, and this is the line that says whether
/// there is room for it.
///
/// The ceiling it prints is the binding limit alone. It is not advice — the
/// resolve pass keeps `array<OitFragment, LAYER_COUNT>` in private memory and
/// bubble-sorts it, so the practical ceiling is well under the one the buffer
/// allows.
fn log_oit_budget(
    device: Option<Res<bevy::render::renderer::RenderDevice>>,
    adapter: Option<Res<bevy::render::renderer::RenderAdapterInfo>>,
    windows: Query<&Window>,
    settings: Query<&OrderIndependentTransparencySettings>,
) {
    /// One layer entry: the `vec2<u32>` `oit_draw` packs colour and depth into.
    const ENTRY_BYTES: u64 = 8;

    // A diagnostic must not be the thing that takes the app down, so every one
    // of these is a question rather than an assertion.
    let (Some(device), Some(adapter), Ok(window)) = (device, adapter, windows.single()) else {
        return;
    };
    let size = window.physical_size();
    let pixels = u64::from(size.x) * u64::from(size.y);
    if pixels == 0 {
        return;
    }

    let limit = device.limits().max_storage_buffer_binding_size as u64;
    let per_layer = pixels * ENTRY_BYTES;
    let layers = settings.iter().next().map(|s| s.layer_count.max(0) as u64);
    let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);

    info!(
        "oit: {}x{} px costs {:.1} MiB per layer, limit {:.0} MiB -> room for {} layers; \
         in use: {}. adapter: {:?} / {}",
        size.x,
        size.y,
        mib(per_layer),
        mib(limit),
        limit / per_layer,
        match layers {
            Some(n) => format!(
                "{} layers, {:.0} MiB ({:.0}%)",
                n,
                mib(per_layer * n),
                100.0 * (per_layer * n) as f64 / limit as f64,
            ),
            None => "no OIT camera".to_string(),
        },
        adapter.backend,
        adapter.name,
    );
}

/// Initial scene setup: camera, lights, ambient.
fn setup_scene(
    mut commands: Commands,
    pick: Res<PickState>,
    mut orbit: ResMut<camera::OrbitCamera>,
) {
    // Where the camera looks before anything has asked it to move: the centre
    // of the caret's cell, which is what every focus after this one aims at.
    //
    // `OrbitCamera::default` can only answer `Vec3::ZERO`, and zero is a
    // cell's *corner* — a cell is anchored at its address by the face turned
    // toward the origin, which is what `layout_to_world` says and
    // `cell_center_world` exists to correct. Left at the default the view
    // opened half a cell off on every axis and came right only once the caret
    // had moved and `trigger_camera_focus_on_selection_change` had had its
    // say: the first framing was the one framing not expressed in the same
    // terms as the rest.
    orbit.target = render::cell_center_world(pick.selected_pos.as_vec3());

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
        // `layer_count` stays at Bevy's eight, and not because eight is
        // comfortable. The layer buffer is one `vec2<u32>` per pixel per layer
        // across the whole viewport, bound whole, so it is spent against
        // `max_storage_buffer_binding_size` — 127 MiB of it at 1920x1080.
        //
        // What that budget *is* is not wgpu's 128 MiB default: Bevy's default
        // priority is `Functionality`, and `initialize_renderer` discards the
        // requested limits and takes the adapter's own, so the ceiling is
        // whatever this GPU reports. `log_oit_budget` prints it at startup
        // rather than leaving it to be assumed.
        //
        // The buffer is not the binding constraint either way. The resolve pass
        // holds `array<OitFragment, LAYER_COUNT>` in private memory — 32 bytes
        // an entry — and bubble-sorts it, so doubling the layers quadruples the
        // sort and doubles a spill that is already costly. Bevy warns past 32;
        // the useful ceiling is well under it.
        //
        // `alpha_threshold` does not stay. Bevy's default is `0.0`, and
        // `oit_draw` tests `color.a < alpha_threshold` — a test `0.0` can never
        // pass. Every blended fragment therefore claims a slot, including the
        // ones a scope surface has already faded to nothing at its outer edge,
        // and slots are the scarce thing here: a ray through a nested program
        // crosses four surfaces per volume before it reaches the caret, and
        // what does not fit is dropped by draw order rather than by depth. One
        // step of the eight-bit alpha a layer entry stores is 1/255, and
        // anything under it packs to zero regardless — so this drops exactly
        // the fragments that could not have shown up, and nothing else.
        OrderIndependentTransparencySettings {
            alpha_threshold: 1.0 / 255.0,
            ..default()
        },
        Msaa::Off,
        // Reads the depth buffer back in a full-screen pass. OIT is what makes
        // that possible without a depth prepass: it already marks the depth
        // texture as bindable, so the cue samples the one the main pass wrote.
        // Off until F9 says otherwise.
        depth_cue::DepthCue::default(),
        camera::OrbitCameraTag,
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
    state: Res<GraphState>,
    eval: Res<EvalState>,
    ui_font: Res<UiFont>,
    pick: Res<PickState>,
    editor_mode: Res<EditorMode>,
    screenshot: Res<ScreenshotMode>,
    clipping: Res<lod::Clipping>,
) {
    // Which volume the caret is in — which is to say which scope, the two being
    // the same question. Fixed for the whole pass: every volume below is graded
    // by how far it sits from this one, outward to nothing and inward to a
    // closed box.
    //
    // Baked in at spawn rather than followed per frame, because a caret move
    // already rebuilds the scene — the caret's own mesh is built from
    // `pick.selected_pos` further down, and would not move otherwise.
    let grading = lod::Lod::new(
        state.scope_of_caret(&pick).map(|scope| scope.path),
        clipping.0,
    );
    // Where the grid surfaces' distance fade hangs from — the same reading as
    // above, one step finer. `grading` asks which volume the caret is in and
    // grades a whole scope by the answer; this asks which cell, so the fade can
    // fall off smoothly around it instead of in steps. Baked in for the same
    // reason, and it is the caret's point for every volume alike: a fade that
    // re-centred per volume would say nothing about where the work is.
    let fog_origin = render::cell_center_world(pick.selected_pos.as_vec3());
    let mut node_entites = std::collections::HashMap::<model::node::Id, Entity>::new();
    let mut anchor_entities = std::collections::HashMap::<model::anchor::Id, Entity>::new();
    let mut anchor_world_positions = std::collections::HashMap::<model::anchor::Id, Vec3>::new();
    // What the level of detail leaves of each anchor's own scope. The edge pass
    // below reads it at both ends of a ribbon: an edge into a Tunnel crosses one
    // wall, so the two ends may be graded a step apart and the strand has to
    // pick one.
    let mut anchor_opacity = std::collections::HashMap::<model::anchor::Id, f32>::new();
    // A Pattern and a TypeCast both hang their declared type on a cell of their
    // own rather than on an anchor, so those cells are recorded here — the link
    // pass has to meet them and would otherwise have no way to ask where they
    // are. One table for both: node ids are unique across kinds, and every
    // lookup site already knows which kind it is holding.
    let mut declared_band_positions = std::collections::HashMap::<model::node::Id, Vec3>::new();
    // Type inference resolves edges, and every edge (pattern branches included)
    // lives in the program-level edge table — so flatten once here instead of
    // per anchor, and hand the same view to the renderer and the edge pass.
    let flat_graph = state.root_graph().flattened_graph();
    // What the run standing on screen has narrowed, and nothing while none is.
    // Asked once here and handed down, for the same reason the flattened graph
    // is: every anchor and every ribbon has to read the same answer or the two
    // ends of a strand disagree about their shape.
    let known = match &eval.phase {
        EvalPhase::Running {
            states,
            current,
            user_source_values,
        } => states[*current]
            .known()
            .offering(eval::literal_types(user_source_values)),
        _ => infer::Known::nothing(),
    };
    // Rasterising the names printed on body faces needs a font synchronously —
    // see `edge::FONT_BYTES` for why that one bypasses the asset server.
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
        // Nothing of a scope the grading has closed over or taken to nothing —
        // not the node, not its anchors, not its labels. Its anchors therefore
        // never reach `anchor_world_positions`, and every edge and link that
        // would have met one falls away on the lookup that expects it there.
        let scope = scope_of_walk(&state, &walked.context);
        if grading.hidden(scope) {
            continue;
        }
        let opacity = grading.content(scope);
        let layout_node = walked.layout_node;
        let node_id = &layout_node.node_id;
        let node = walked.layout_graph.graph.nodes.get(node_id).unwrap();
        match node {
            model::node::ENode::Pattern { .. } => {
                declared_band_positions.insert(
                    node_id.clone(),
                    render::pattern_band_world(layout_node, walked.extra_offset),
                );
            }
            model::node::ENode::TypeCast { .. } => {
                declared_band_positions.insert(
                    node_id.clone(),
                    render::cast_band_world(layout_node, walked.extra_offset),
                );
            }
            _ => {}
        }
        let render_node = render::layoutnode_to_rendernode(
            layout_node,
            walked.layout_graph,
            &flat_graph,
            &state.function_declarations,
            &known,
            walked.extra_offset,
        );
        // A node drawn only as strands or loose objects has no mesh of its own;
        // the entity is still spawned so picking and selection keep working.
        let node_entity = match render_node.node {
            Some(obj) => commands
                .spawn((
                    Mesh3d(meshes.add(obj.mesh)),
                    MeshMaterial3d(materials.add(lod::faded(obj.material, opacity))),
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

        // Meshes a node owns without their being its body or one of its
        // anchors: the line and the point a Constant is drawn as. They carry no
        // `NodeEntity`, so nothing picks them and nothing hovers them — the
        // node's own entity, spawned above, answers for all of that.
        for object in render_node.objects {
            commands.spawn((
                Mesh3d(meshes.add(object.mesh)),
                MeshMaterial3d(materials.add(lod::faded(object.material, opacity))),
                object.transform,
                SceneEntity,
            ));
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
                MeshMaterial3d(materials.add(lod::faded(
                    StandardMaterial {
                        base_color_texture: Some(texture),
                        ..face.material
                    },
                    opacity,
                ))),
                face.transform,
                SceneEntity,
            ));
        }

        // Markers the node owns directly rather than through an anchor: a
        // Pattern is drawn as the band of the type its arm matches.
        spawn_anchor_strands(
            &mut commands,
            &mut meshes,
            &mut materials,
            &ui_font.0,
            render_node.strands,
            opacity,
        );

        render_node
            .anchors
            .into_iter()
            .for_each(|(anchor_id, render_anchor)| {
                let render::RenderAnchor {
                    pick_center,
                    strands,
                    plain_body,
                } = render_anchor;

                spawn_anchor_strands(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &ui_font.0,
                    strands,
                    opacity,
                );

                // Neutral cuboid for anchors without strands.
                if let Some(body) = plain_body {
                    commands.spawn((
                        Mesh3d(meshes.add(body.mesh)),
                        MeshMaterial3d(materials.add(lod::faded(body.material, opacity))),
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
                anchor_opacity.insert(anchor_id.clone(), opacity);
                anchor_world_positions.insert(anchor_id, pick_center);
            });

        node_entites.insert(node_id.clone(), node_entity.clone());

        render_node.labels.into_iter().for_each(|l| {
            spawn_world_label(&mut commands, &ui_font.0, l, SceneEntity, opacity);
        });
    }

    for e in state.root_graph().edges() {
        let src_id = &e.from_anchor.anchor_id;
        let tgt_id = &e.to_anchor.anchor_id;

        let Some(&from_world) = anchor_world_positions.get(src_id) else {
            continue;
        };
        let Some(&to_world) = anchor_world_positions.get(tgt_id) else {
            continue;
        };
        // The dimmer of the two ends. An edge reaching out of a scope goes into
        // a Tunnel one wall away, so the two gradings differ by at most a step,
        // and taking the lesser keeps a strand from being the brightest thing
        // in the volume it is leaving.
        let opacity = anchor_opacity
            .get(src_id)
            .copied()
            .unwrap_or(1.0)
            .min(anchor_opacity.get(tgt_id).copied().unwrap_or(1.0));

        let src_type = infer::anchor_type(&flat_graph, src_id, &state.function_declarations)
            .unwrap_or(infer::EType::Pending);
        // The row a run settled this edge's source on, if one has. Read once
        // and used at both ends: the same narrowing has to reach the target's
        // rows, or a ribbon would leave a row that is no longer drawn or land
        // on one that is not there.
        let taken = known.at_output(&flat_graph, src_id);
        let source_rows = render::drawn_rows(&src_type, taken);

        let curve = edge::EdgeCurve::from_endpoints(from_world, to_world);

        // Spawned before the strands and before anything may leave the loop:
        // an edge the user wired exists whatever the inferer has to say about
        // it, and the entity that stands for it should not depend on that
        // either.
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

        // A source type with no rows — `Pending`, or an anchor the inferer
        // could not resolve at all — leaves no row for a strand to run along.
        // Drawn all the same: what is undecided is the type, not the wiring,
        // and an edge left out reads as an edge never made. One band on the
        // anchors' own row, wearing the grey `plain_anchor_body` gives either
        // end, cut across by the pattern that says the type is still open.
        if source_rows.is_empty() {
            spawn_pending_ribbon(
                &mut commands,
                &mut meshes,
                &mut materials_edge,
                from_world,
                to_world,
                // A band at both ends: there is nothing to taper into, and a
                // hairline would spell a value where there is not even a type
                // yet.
                whole_row_end(from_world.y, false),
                whole_row_end(to_world.y, false),
                Some(edge_root),
                opacity,
            );
            continue;
        }

        // A target that constrains nothing renders as tall as what arrives,
        // so the ribbons must use that same type — otherwise every leaf
        // would collapse onto the anchor's first row.
        let tgt_type = infer::anchor_type(&flat_graph, tgt_id, &state.function_declarations)
            .or_else(|| {
                infer::incoming_anchor_type(&flat_graph, tgt_id, &state.function_declarations)
            });
        let target_rows = tgt_type
            .as_ref()
            .map(|t| render::drawn_rows(t, taken))
            .unwrap_or_default();

        // graph-level literal on the source anchor. When present, the sole
        // rendered leaf swaps to the thin "value line" style — same rule
        // the anchor strands follow, via the same lookup.
        let src_graph_value = infer::anchor_literal(&flat_graph, src_id, &known);

        for (k, leaf) in source_rows.iter() {
            // Same row offsets the anchor strands use, so each ribbon meets
            // the strand it continues exactly — the index comes from
            // `drawn_rows` rather than from the loop, so a narrowed row is met
            // where the layout still keeps it.
            let y_src = render::leaf_row_offset(*k);
            // Which row at the target will accept this strand: the one that
            // admits it. Asked as subsumption rather than by kind because a
            // row may be a literal now — a `1` strand belongs on the `1`
            // row of a `1|2` anchor, not merely on some Integer row.
            let y_tgt = match target_rows
                .iter()
                .find(|(_, target_leaf)| infer::subsumes(target_leaf, leaf))
            {
                Some((idx, _)) => render::leaf_row_offset(*idx),
                // No matching leaf at the target: aim at its first row.
                None => 0.0,
            };
            // A leaf that claims no row of its own — a sum type, or `Pending` —
            // has no strand to draw.
            if edge::leaf_kind_of(leaf).is_none() {
                continue;
            }
            // A strand carrying a value is a hairline, a strand carrying a type
            // is a band. Literally the same question the anchor's own strands ask,
            // asked through the same function and of the same leaf, so a strand
            // and the row it lands on cannot end up wearing different shapes.
            let (height, line_mode) =
                if render::leaf_is_drawn_as_line(leaf, src_graph_value.as_deref()) {
                    (edge::RIBBON_LINE_HEIGHT, 1.0)
                } else {
                    (render::STRAND_BAND_HEIGHT, 0.0)
                };
            let (mesh, arc_total) =
                edge::build_ribbon_mesh(&curve, from_world.y + y_src, to_world.y + y_tgt, height);
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(materials_edge.add(edge::EdgeMaterial {
                    // An edge carries one type from end to end, so both
                    // colours are the one colour.
                    band_color_start: render::strand_color(leaf).to_linear(),
                    band_color_end: render::strand_color(leaf).to_linear(),
                    time: 0.0,
                    // Both ends of an ordinary edge wear the same shape, so
                    // there is nothing for the two line modes to interpolate
                    // between. The arc length is the real one all the same —
                    // it is where a fragment *is*, not merely what the modes
                    // are read against.
                    line_mode_start: line_mode,
                    line_half_thickness: edge::RIBBON_LINE_HALF_THICKNESS_UV,
                    line_mode_end: line_mode,
                    // Built at a constant height, so the width has nothing to
                    // ramp between.
                    height_start: height,
                    height_end: height,
                    arc_total,
                    dash_period: 0.0,
                    dash_duty: 0.0,
                    opacity,
                })),
                ChildOf(edge_root),
                SceneEntity,
            ));
        }
    }

    // A link is the same substance as an edge — same ribbon, same taper between
    // band and line — so these passes share everything but the table they read
    // from.
    spawn_match_links(
        &mut commands,
        &mut meshes,
        &mut materials_edge,
        &state,
        &flat_graph,
        &known,
        &anchor_world_positions,
        &declared_band_positions,
        &grading,
    );
    spawn_cast_links(
        &mut commands,
        &mut meshes,
        &mut materials_edge,
        &state,
        &flat_graph,
        &known,
        &anchor_world_positions,
        &declared_band_positions,
        &grading,
    );

    for walked_graph in state.root_graph().walk_all_graphs() {
        let Some(bounds) = walked_graph.layout_graph.grid_bounds() else {
            continue;
        };
        // A volume inside one already drawn as a solid body is not reached at
        // all, its own box included. A sealed volume itself still is — the box
        // is the whole of what is left of it, and `spawn_volume_surfaces` leaves
        // out the four surfaces once nothing of the volume survives.
        if grading.dropped(&walked_graph.context) {
            continue;
        }
        let offset = walked_graph.extra_offset;
        // `layout_range_to_world` re-normalises min/max: LAYOUT_SCALE negates
        // Z, so scaling the corners individually would yield an inverted rect
        // and the shader would draw no border at all.
        let (border_lo, border_hi) = render::layout_range_to_world(
            bounds.min.as_vec3() + offset,
            bounds.max.as_vec3() + offset,
            0.0,
        );
        // Collect multi-cell node footprints in this LayoutGraph and convert
        // to world-space XZ rects. Fed to the grid shader to suppress
        // interior grid lines inside merged fields.
        //
        // A Match is not one of those. Its footprint is an envelope and not a
        // field — the arms standing in it have their own volumes and the space
        // between them is the scope's own floor — so flattening it would leave
        // a blank rectangle with boxes floating in it. It is the one node whose
        // `node_footprint` reaches past its own cells, and this is the one
        // place that has to say so.
        let mut footprints = [Vec4::ZERO; grid::MAX_FOOTPRINTS];
        let mut footprint_count: u32 = 0;
        for id in walked_graph.layout_graph.layout_nodes.keys() {
            if matches!(
                walked_graph.layout_graph.graph.nodes.get(id),
                Some(model::node::ENode::Match { .. })
            ) {
                continue;
            }
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

        spawn_volume_surfaces(
            &mut commands,
            &mut meshes,
            &mut materials_grid,
            &mut materials,
            bounds.min,
            bounds.max,
            offset,
            grading.content(&walked_graph.context),
            grading.shell(&walked_graph.context),
            fog_origin,
            InteractiveFloor {
                scope: ScopeGridEntity {
                    context: walked_graph.context.clone(),
                    origin_offset: offset,
                    min: bounds.min,
                    max: bounds.max,
                },
                border_min: Vec2::new(border_lo.x, border_lo.z),
                border_max: Vec2::new(border_hi.x, border_hi.z),
                footprints,
                footprint_count,
            },
        );
    }

    // Selection caret, enclosing the addressed cell volume from the caret
    // address to address + (1,1,1). Rebuild-driven, like every other scene
    // entity — caret moves and mode switches already flag a rebuild.
    //
    // It shows which mode it is in: NORMAL outlines the cell, INSERT fills the
    // two faces the next insert would open along and blinks like a text caret.
    //
    // In screenshot mode there is none. It is the editor pointing at the
    // program, and pointing is the one thing that mode is for leaving out — so
    // it is not spawned rather than hidden, the way every other scene entity
    // that is not wanted simply is not built.
    match *editor_mode {
        _ if screenshot.active() => {}
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

/// The top edge: the two actions that are about the document rather than about
/// the graph, and the name of the thing they belong to.
///
/// Both wear `EditorChrome` on their container and not on their parts — the
/// system that writes that `display` blanks every carrier it finds, so a part
/// that carried it too would be a second writer on the same node.
fn spawn_ui(mut commands: Commands, ui_font: Res<UiFont>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(8.0),
                ..default()
            },
            EditorChrome,
        ))
        .with_children(|row| {
            row.spawn((
                Button,
                chrome_button_node(),
                BackgroundColor(CONTROL_REST),
                NewButton,
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new("New"),
                    text_font(&ui_font.0, 14.0),
                    TextColor(INK_BRIGHT),
                ));
            });
            row.spawn((
                Button,
                chrome_button_node(),
                BackgroundColor(CONTROL_REST),
                HelpButton,
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new("Help"),
                    text_font(&ui_font.0, 14.0),
                    TextColor(INK_BRIGHT),
                ));
            });
        });

    // Centred by a full-width row rather than by a guessed offset, so it stays
    // centred as the window resizes. The same arrangement the mode toggle has.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(14.0),
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::Center,
                ..default()
            },
            EditorChrome,
        ))
        .with_children(|row| {
            row.spawn((
                Text::new("Expression Visualizer"),
                text_font(&ui_font.0, 18.0),
                TextColor(Color::srgb(0.95, 0.95, 1.0)),
            ));
        });
}

/// The shape of a button that stands on the scene rather than inside a panel:
/// the same padding and radius `modal_button` has, without the margin a
/// button in a row of buttons needs — the row here sets its own gap.
fn chrome_button_node() -> Node {
    Node {
        padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
        border_radius: BorderRadius::all(Val::Px(6.0)),
        ..default()
    }
}

/// What a control on the scene rests in, and what it turns under the pointer.
///
/// This pair was the de-facto theme long before it had a name — the resting
/// fill alone stood as a literal in eleven places. Named here because
/// `paint_hover` has to say it once, and a colour that eleven sites agree on
/// by coincidence is a colour that will one day disagree.
const CONTROL_REST: Color = Color::srgba(0.16, 0.16, 0.22, 0.9);
const CONTROL_HOT: Color = Color::srgba(0.2, 0.2, 0.3, 0.95);
/// A button inside a modal sits on the modal's own backdrop rather than on the
/// scene, so it rests and lifts a shade brighter to keep the same contrast.
const MODAL_REST: Color = Color::srgba(0.18, 0.18, 0.28, 0.95);
const MODAL_HOT: Color = Color::srgba(0.25, 0.25, 0.35, 0.95);
/// What a row of a list turns under the pointer. A row rests invisible — the
/// panel behind it is its colour — so only the lit half is worth a name.
const ROW_HOT: Color = Color::srgba(0.2, 0.2, 0.3, 0.95);
/// Nothing at all, spelt as a colour. A row's resting fill, and the one value
/// that would read as a mistake written as `srgba(0.0, 0.0, 0.0, 0.0)`.
const FILL_NONE: Color = Color::srgba(0.0, 0.0, 0.0, 0.0);

/// What a control that is on screen but cannot be used is filled and written
/// in. Being switched off outranks the pointer, which is why this pair is not
/// part of `HoverFill`: a control with a state needs one writer deciding the
/// state and the pointer together, and that is what `sync_chrome_buttons` and
/// `update_step_button_visuals` are.
const CONTROL_OFF: Color = Color::srgba(0.10, 0.10, 0.13, 0.9);
const INK_OFF: Color = Color::srgb(0.35, 0.35, 0.4);

/// The three inks, dimmest first: a label at rest beside a lit one, a label
/// that is the point of its button, and a label under the pointer.
const INK_DIM: Color = Color::srgb(0.6, 0.6, 0.7);
const INK_BRIGHT: Color = Color::srgb(0.85, 0.85, 0.9);
const INK_HOT: Color = Color::srgb(1.0, 1.0, 1.0);

/// What a control is filled with, by whether the pointer stands on it.
///
/// A component carrying two colours rather than two constants inside the
/// system, because the same painter has to serve a bar button that rests
/// opaque and a list row that rests invisible, and what separates those is two
/// colours and nothing else.
///
/// **Only for controls whose look depends on nothing but the pointer.**
/// `update_step_button_visuals` and `sync_mode_toggles` decide a fill from a
/// state *and* the pointer, in one writer each, and say why. Hanging this on
/// their nodes would put a second writer on the same colour, and the two would
/// take turns — which is the arrangement those two systems were written to end.
#[derive(Component, Clone, Copy)]
struct HoverFill {
    rest: Color,
    hot: Color,
}

/// The same for the text under it, and separate from it on purpose: a row
/// whose ink says something of its own — a severity, a greyed-out suggestion,
/// a set of tallies in three colours — keeps its ink and takes only the fill.
#[derive(Component, Clone, Copy)]
struct HoverInk {
    rest: Color,
    hot: Color,
}

/// Paint every `HoverFill` by where the pointer is.
///
/// `Pressed` reads as `Hovered` and not as a third look: the press is already
/// answered by whatever the button does, and a control that darkened for the
/// duration of a click would be reporting the click twice.
///
/// The ink goes to *every* child that has one rather than to `children[0]`.
/// That is what lets the checkbox share this system — its first child is the
/// swatch, which has no `TextColor` at all — and it is why a control may hold
/// one label or two without the painter being told which.
///
/// Nothing is ordered around this. `Interaction` is written in `PreUpdate`, so
/// a row spawned during `Update` is at `None` for the rest of its first frame
/// and takes its colour the moment the focus pass first sees it.
fn paint_hover(
    mut control_q: Query<
        (
            &Interaction,
            &HoverFill,
            &mut BackgroundColor,
            Option<&HoverInk>,
            Option<&Children>,
        ),
        Changed<Interaction>,
    >,
    mut ink_q: Query<&mut TextColor>,
) {
    for (interaction, fill, mut bg, ink, children) in control_q.iter_mut() {
        let hot = matches!(*interaction, Interaction::Hovered | Interaction::Pressed);
        let wanted = if hot { fill.hot } else { fill.rest };
        if bg.0 != wanted {
            bg.0 = wanted;
        }
        let (Some(ink), Some(children)) = (ink, children) else {
            continue;
        };
        let wanted = if hot { ink.hot } else { ink.rest };
        for child in children.iter() {
            let Ok(mut color) = ink_q.get_mut(child) else {
                continue;
            };
            if color.0 != wanted {
                color.0 = wanted;
            }
        }
    }
}

/// Marker on the bottom-right box that holds the problem list and the run
/// controls. It owns nothing but the frame; the two zones inside it are their
/// own nodes.
#[derive(Component)]
struct RunPanel;

/// The rule between the two zones. Shown only while there is a list above it.
#[derive(Component)]
struct RunPanelRule;

/// One of the four ways of looking at the program.
///
/// Three of them are camera states and the fourth is not, which is the whole
/// reason they are one control: what the user is choosing between is *what the
/// screen shows*, and "everything but the program" belongs in that list beside
/// "from here" and "from anywhere". The three controls this replaced —
/// `Camera: bound/free`, a `semi ortho` checkbox and a `Screenshot` button —
/// spread one question over three widgets and two idioms.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewEntry {
    /// The bound camera, parallel. The default, and the exact picture the
    /// layout is specified in.
    EditOrtho,
    /// The bound camera with the convergence eased in: the same oblique
    /// picture, the same standoff, depth that converges a little.
    EditPersp,
    /// The free camera. Orbit, pan and zoom by hand, and the guarantees of the
    /// bound mode deliberately suspended.
    Explore,
    /// Everything the editor draws about itself, gone. Not a camera state at
    /// all — it changes nothing about where the view is, only what is left
    /// standing in it.
    Screenshot,
}

impl ViewEntry {
    const ALL: [ViewEntry; 4] = [
        ViewEntry::EditOrtho,
        ViewEntry::EditPersp,
        ViewEntry::Explore,
        ViewEntry::Screenshot,
    ];

    fn label(self) -> &'static str {
        match self {
            ViewEntry::EditOrtho => "Edit Ortho",
            ViewEntry::EditPersp => "Edit Persp",
            ViewEntry::Explore => "Explore",
            ViewEntry::Screenshot => "Screenshot",
        }
    }
}

/// Which entry the live state *is*, read rather than remembered.
///
/// Nothing stores the selection, and that is what makes the screenshot mode
/// return by itself: it never touches the camera, so the moment
/// `end_screenshot_mode` clears `entered_at` the answer falls back to whatever
/// the camera was already doing. A remembered selection would have to be put
/// back by hand, and would be wrong the first time something else moved it.
///
/// `semi_ortho` is only read in the bound mode — `apply_projection` takes the
/// free branch whole — so the free camera answers `Explore` whatever it is set
/// to, and carries the value untouched until the user comes back.
fn view_entry(orbit: &camera::OrbitCamera, screenshot: &ScreenshotMode) -> ViewEntry {
    if screenshot.active() {
        return ViewEntry::Screenshot;
    }
    match (orbit.mode, orbit.semi_ortho) {
        (camera::CameraMode::Free, _) => ViewEntry::Explore,
        (camera::CameraMode::Bound, true) => ViewEntry::EditPersp,
        (camera::CameraMode::Bound, false) => ViewEntry::EditOrtho,
    }
}

/// Whether the view list is unfolded.
#[derive(Resource, Default)]
struct ViewMenuOpen(bool);

/// The row at the bottom-left holding the view control and the clipping
/// checkbox. Wears `EditorChrome` for everything in it.
#[derive(Component)]
struct ViewBar;

/// The button that opens the list, and shows what is chosen.
#[derive(Component)]
struct ViewMenuButton;

/// The text inside that button, which is what a sync writes.
#[derive(Component)]
struct ViewMenuLabel;

/// The list itself. Carried so a click on its own padding does not read as a
/// click somewhere else and fold it.
#[derive(Component)]
struct ViewMenuPopup;

/// On every node the list is built from, so the sweep that clears it cannot
/// reach anything else. The same arrangement `PlayerControlsEntity` has.
#[derive(Component)]
struct ViewMenuEntity;

/// One row of the list.
#[derive(Component)]
struct ViewMenuOption(ViewEntry);

/// Go to a view.
///
/// The two camera arms are the ones the camera-mode button used to carry, and
/// they are unchanged: bound to free matches the free camera's distance to the
/// scale being looked at, so the caret's plane keeps its size and what happens
/// is the depth opening up rather than a jump; free to bound is a journey back
/// to the default view and leaves the scale setting alone.
///
/// What is new is that the target is *named* rather than toggled to, so coming
/// back from `Explore` says which of the two bound pictures it is coming back
/// to. Going *out* to `Explore` leaves `semi_ortho` alone on purpose: the
/// blend's near half is built from it, so the transition fades out of the
/// picture that was actually on screen.
#[allow(clippy::too_many_arguments)]
fn apply_view_entry(
    entry: ViewEntry,
    orbit: &mut camera::OrbitCamera,
    tween: &mut camera::CameraTween,
    screenshot: &mut ScreenshotMode,
    rebuild: &mut NeedsRebuild,
    height: f32,
    caret: Vec3,
    now: f32,
) {
    if entry == ViewEntry::Screenshot {
        // Not a camera state: the camera is left exactly where it is, and the
        // rebuild is for the caret, which goes away by not being spawned.
        screenshot.entered_at = Some(now);
        rebuild.0 = true;
        return;
    }
    let wanted_semi = match entry {
        ViewEntry::EditPersp => Some(true),
        ViewEntry::EditOrtho => Some(false),
        // Inert while free, and what the return trip fades out of.
        ViewEntry::Explore | ViewEntry::Screenshot => None,
    };
    if let Some(semi) = wanted_semi {
        orbit.semi_ortho = semi;
    }
    let wanted_mode = match entry {
        ViewEntry::Explore => camera::CameraMode::Free,
        _ => camera::CameraMode::Bound,
    };
    if orbit.mode == wanted_mode {
        return;
    }
    match wanted_mode {
        camera::CameraMode::Free => {
            let (theta, phi) = camera::oblique_view_angles();
            let visible_world = height / orbit.cell_pixels;
            orbit.free_fov = 2.0 * (visible_world * 0.5 / orbit.radius).atan();
            orbit.mode = camera::CameraMode::Free;
            tween.to_view(&*orbit, theta, phi, orbit.radius, orbit.target);
        }
        camera::CameraMode::Bound => {
            let radius = camera::bound_radius(&*orbit, height);
            orbit.mode = camera::CameraMode::Bound;
            tween.to_view(
                &*orbit,
                camera::RESET_THETA,
                camera::RESET_PHI,
                radius,
                caret,
            );
        }
    }
}

/// The trigger, a row of options, and folding the list again.
///
/// Folding on a click elsewhere is the pattern `text_input_focus` uses, and it
/// needs the same care: Bevy marks only the topmost node `Pressed`, so a press
/// on a row leaves the trigger and the list itself at `None`. Everything that
/// counts as "inside" is therefore asked before the fall-through, and each of
/// the three answers returns rather than dropping into the next.
#[allow(clippy::too_many_arguments)]
fn handle_view_menu_click(
    trigger_q: Query<&Interaction, (Changed<Interaction>, With<ViewMenuButton>)>,
    popup_q: Query<&Interaction, (Changed<Interaction>, With<ViewMenuPopup>)>,
    option_q: Query<(&Interaction, &ViewMenuOption), Changed<Interaction>>,
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    pick: Res<PickState>,
    time: Res<Time>,
    mut open: ResMut<ViewMenuOpen>,
    mut orbit: ResMut<camera::OrbitCamera>,
    mut tween: ResMut<camera::CameraTween>,
    mut screenshot: ResMut<ScreenshotMode>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    if let Some(entry) = option_q
        .iter()
        .find(|(interaction, _)| **interaction == Interaction::Pressed)
        .map(|(_, option)| option.0)
    {
        let height = windows
            .single()
            .map(|window| window.height())
            .unwrap_or(1080.0);
        apply_view_entry(
            entry,
            &mut orbit,
            &mut tween,
            &mut screenshot,
            &mut rebuild,
            height,
            render::cell_center_world(pick.selected_pos.as_vec3()),
            time.elapsed_secs(),
        );
        open.0 = false;
        return;
    }
    if trigger_q.iter().any(|i| *i == Interaction::Pressed) {
        open.0 = !open.0;
        return;
    }
    // The list's own padding is inside it, and a press there is not a press
    // somewhere else.
    if popup_q.iter().any(|i| *i == Interaction::Pressed) {
        return;
    }
    if open.0 && mouse.just_pressed(MouseButton::Left) {
        open.0 = false;
    }
}

/// Write the trigger's caption, and build or clear the list.
///
/// The list is a despawned subtree rather than a node whose `display` is
/// toggled, and that is not a style choice: `sync_editor_chrome` writes
/// `Display::Flex` on every `EditorChrome` node whenever a modal closes, so a
/// folded list that wore the marker would be torn open by something that knows
/// nothing about it. Under a trigger that wears it, a subtree that is simply
/// not there cannot be revealed. `PlayerControls` holds its row the same way.
fn sync_view_menu(
    mut commands: Commands,
    open: Res<ViewMenuOpen>,
    orbit: Res<camera::OrbitCamera>,
    screenshot: Res<ScreenshotMode>,
    ui_font: Res<UiFont>,
    trigger_q: Query<Entity, With<ViewMenuButton>>,
    content_q: Query<Entity, With<ViewMenuEntity>>,
    mut label_q: Query<&mut Text, With<ViewMenuLabel>>,
    mut cache: Local<Option<(bool, bool, bool, bool)>>,
) {
    let current = view_entry(&orbit, &screenshot);
    let caption = format!("View: {} \u{25BE}", current.label());
    for mut text in label_q.iter_mut() {
        if text.0 != caption {
            text.0 = caption.clone();
        }
    }

    // `ViewEntry` is not hashable and there are four of them, so the latch is
    // the open flag beside the three bits that decide which row is lit.
    let fp = (
        open.0,
        orbit.mode == camera::CameraMode::Free,
        orbit.semi_ortho,
        screenshot.active(),
    );
    if *cache == Some(fp) {
        return;
    }
    *cache = Some(fp);

    for e in content_q.iter() {
        commands.entity(e).despawn();
    }
    if !open.0 {
        return;
    }
    let Ok(trigger) = trigger_q.single() else {
        return;
    };
    let font = &ui_font.0;
    commands.entity(trigger).with_children(|parent| {
        parent
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(0.0),
                    // Standing on the trigger's top edge, opening upward —
                    // there is no room below it.
                    bottom: Val::Percent(100.0),
                    margin: UiRect::bottom(Val::Px(6.0)),
                    min_width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.08, 0.08, 0.14, 0.98)),
                BorderColor::all(Color::srgb(0.25, 0.25, 0.4)),
                // Above everything that stacks by spawn order — the run panel
                // is drawn after this one and would otherwise cover it — and
                // well below the modals, which own the screen outright at 50.
                GlobalZIndex(10),
                Button,
                ViewMenuPopup,
                ViewMenuEntity,
            ))
            .with_children(|list| {
                for entry in ViewEntry::ALL {
                    let chosen = entry == current;
                    // The chosen row rests lit, so its hover has to lift from
                    // *there* rather than from nothing: a painter that sent it
                    // back to transparent would unchoose it on the way out.
                    let rest = if chosen { ROW_HOT } else { FILL_NONE };
                    let ink_rest = if chosen { INK_BRIGHT } else { INK_DIM };
                    list.spawn((
                        Node {
                            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                            border_radius: BorderRadius::all(Val::Px(3.0)),
                            ..default()
                        },
                        BackgroundColor(rest),
                        HoverFill {
                            rest,
                            hot: if chosen {
                                Color::srgba(0.26, 0.26, 0.36, 0.95)
                            } else {
                                ROW_HOT
                            },
                        },
                        HoverInk {
                            rest: ink_rest,
                            hot: INK_BRIGHT,
                        },
                        Button,
                        ViewMenuOption(entry),
                        ViewMenuEntity,
                    ))
                    .with_children(|row| {
                        row.spawn((
                            Text::new(entry.label()),
                            text_font(font, 14.0),
                            TextColor(ink_rest),
                        ));
                    });
                }
            });
    });
}

/// The bottom-left row: how to look, and how much to look at.
///
/// Both are about the picture rather than about the program, which is why they
/// stand together and why neither is near the run controls. A flex row rather
/// than two corner addresses, so the widths settle themselves — the view
/// caption changes length as the entry does.
fn spawn_view_bar(mut commands: Commands, ui_font: Res<UiFont>) {
    let font = ui_font.0.clone();
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(12.0),
                bottom: Val::Px(12.0),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                ..default()
            },
            ViewBar,
            EditorChrome,
        ))
        .with_children(|bar| {
            bar.spawn((
                Button,
                Node {
                    padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
                    border_radius: BorderRadius::all(Val::Px(6.0)),
                    ..default()
                },
                BackgroundColor(CONTROL_REST),
                HoverFill {
                    rest: CONTROL_REST,
                    hot: CONTROL_HOT,
                },
                HoverInk {
                    rest: INK_DIM,
                    hot: INK_BRIGHT,
                },
                ViewMenuButton,
            ))
            .with_children(|button| {
                button.spawn((
                    Text::new("View: Edit Ortho \u{25BE}"),
                    text_font(&font, 14.0),
                    TextColor(INK_DIM),
                    ViewMenuLabel,
                ));
            });
            spawn_inline_checkbox(
                bar,
                &font,
                "Clipping",
                ClippingCheckbox,
                ClippingCheckboxBox,
            );
        });
}

/// The flex twin of the old corner checkbox: the same row, the same swatch, but
/// a child of a container rather than an address of its own. It carries no
/// `EditorChrome` — the bar wears that for everything in it.
fn spawn_inline_checkbox<C: Bundle, B: Bundle>(
    parent: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    label: &str,
    component: C,
    swatch: B,
) {
    parent
        .spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(10.0), Val::Px(8.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                ..default()
            },
            BackgroundColor(CONTROL_REST),
            // The swatch is the first child and carries no `TextColor`, which
            // is the case `paint_hover` walks every child for: only the label
            // beside it takes the ink.
            HoverFill {
                rest: CONTROL_REST,
                hot: CONTROL_HOT,
            },
            HoverInk {
                rest: INK_DIM,
                hot: INK_BRIGHT,
            },
            component,
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
                swatch,
            ));
            parent.spawn((Text::new(label), text_font(font, 14.0), TextColor(INK_DIM)));
        });
}

/// The two states a checkbox swatch is painted in.
const CHECKED_COLOR: Color = Color::srgb(0.133, 0.827, 0.933);
const UNCHECKED_COLOR: Color = Color::srgba(0.06, 0.06, 0.12, 0.95);

/// Marker for the checkbox that turns the level-of-detail grading on.
#[derive(Component)]
struct ClippingCheckbox;

/// Marker on that checkbox's swatch.
#[derive(Component)]
struct ClippingCheckboxBox;

/// Toggle the grading. Every opacity it decides is baked into a material at
/// spawn — a caret move already rebuilds the scene, so nothing follows it per
/// frame — which is why this has to ask for a rebuild itself, the way
/// `apply_view_entry` does for the caret the screenshot mode removes.
fn handle_clipping_checkbox(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<ClippingCheckbox>)>,
    mut clipping: ResMut<lod::Clipping>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    for interaction in interaction_q.iter() {
        if *interaction == Interaction::Pressed {
            clipping.0 = !clipping.0;
            rebuild.0 = true;
        }
    }
}

fn sync_clipping_checkbox(
    clipping: Res<lod::Clipping>,
    mut box_q: Query<&mut BackgroundColor, With<ClippingCheckboxBox>>,
) {
    let wanted = if clipping.0 {
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

/// The bottom-right box: what is wrong with the graph, and the controls that
/// run it — one frame around both.
///
/// They used to stand apart, the list a full-width bar near the top of the
/// bottom edge and the controls centred below it. Which read as two unrelated
/// things, and they are not: a run does not start while an error stands
/// (`Diagnostics::blocking`), so the list is the *precondition* of the row
/// under it. A shared frame with a rule between the two says that; adjacency
/// alone did not.
///
/// Anchored at the bottom, so the box grows upward. The controls therefore keep
/// their place whatever the list does, and the last error going away does not
/// move the button the user is reaching for.
///
/// One `display` writer per node, which is the whole reason this is three nodes
/// and not one: `EditorChrome` takes the outer box, and `sync_diagnostics_ui`
/// takes the section and the rule. `PlayerControls` writes none — its children
/// come and go instead (`sync_player_controls`).
fn spawn_run_panel(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(14.0),
                bottom: Val::Px(12.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Stretch,
                padding: UiRect::all(Val::Px(10.0)),
                row_gap: Val::Px(6.0),
                // Fixed, not content-sized. A box that grew with the longest
                // message would change width every time the list did, and the
                // controls under it would slide with it — the one thing a row
                // of buttons must never do. Long messages wrap into it instead.
                width: Val::Px(420.0),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.10, 0.10, 0.16, 0.9)),
            BorderColor::all(Color::srgb(0.25, 0.25, 0.4)),
            // Carried with no handler of its own, so that a click landing on the
            // box does not fall through and move the caret to whatever cell is
            // behind it. The two standing panels do the same.
            Button,
            RunPanel,
            EditorChrome,
        ))
        .with_children(|panel| {
            panel.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    ..default()
                },
                DiagnosticsPanel,
            ));
            // The rule between the two zones. Drawn as a one-pixel box rather
            // than as a border on either neighbour, because it belongs to the
            // seam and to neither side of it.
            //
            // Always there, like the heading above it: the two zones are what
            // the box *is*, and a frame that lost a line when the list emptied
            // would be a different frame.
            panel.spawn((
                Node {
                    height: Val::Px(1.0),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.25, 0.25, 0.4)),
                RunPanelRule,
            ));
            panel.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::FlexEnd,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                },
                PlayerControls,
            ));
        });
}

/// One control in that row: a square button wearing a glyph or two.
///
/// A child of the row rather than an address of its own, and wide enough that
/// the glyphs sit in the middle of it rather than filling it. Same colours and
/// radius as the view bar's controls, which is what keeps the two ends of the
/// bottom edge reading as one set. It carries no `EditorChrome`: the run panel
/// wears that for the whole row. Colour is not its
/// business either; `update_step_button_visuals` writes the whole row's.
fn spawn_control_button<C: Bundle>(
    parent: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    glyph: &str,
    component: C,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(40.0),
                height: Val::Px(32.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(CONTROL_REST),
            component,
            PlayerControlsEntity,
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(glyph),
                text_font(font, 18.0),
                TextColor(Color::srgb(0.6, 0.6, 0.7)),
            ));
        });
}

/// Take out what the caret stands on, where it stands on something that may
/// go. Says whether the graph changed, like the `remove_node` it ends in;
/// flagging the rebuild is the caller's.
///
/// Sink and BranchSource are constitutive parts of their scope, not
/// user-placed nodes — neither can be deleted.
fn delete_node_at_caret(state: &mut GraphState, pick: &PickState) -> bool {
    let Some((caret_graph, local)) = state.caret_graph(pick) else {
        return false;
    };
    let Some(node_id) = caret_graph.node_at(local) else {
        return false;
    };
    let is_fixture = matches!(
        caret_graph.graph.nodes.get(&node_id),
        Some(model::node::ENode::Sink { .. } | model::node::ENode::BranchSource { .. })
    );
    if is_fixture {
        return false;
    }
    remove_node(state, &node_id)
}

/// Take a node out of the graph it lives in, wherever that is. Returns whether
/// the graph changed, so the caller knows whether to flag a rebuild.
///
/// Found through the node's own scope rather than through the caret's. The
/// caret is where two of the three callers got the node from, but it is not
/// where the third one is: walking away from an unfinished node is what unmakes
/// it, and by the time that is noticed the caret is already standing somewhere
/// else — possibly in another scope entirely.
fn remove_node(state: &mut GraphState, node_id: &model::node::Id) -> bool {
    let Some(context) = state.root_graph().context_of_node(node_id) else {
        return false;
    };
    let updated = state
        .root_graph()
        .resolve_context(&context)
        .minus_node(node_id);
    let Some(owning_graph) = state.root_graph_mut().resolve_context_mut(&context) else {
        return false;
    };
    *owning_graph = updated;
    // `minus_node` took the edges its own graph held. Every other edge in the
    // scene is the root's — `plus_edge` is only ever called there — so a node
    // removed from a branch leaves its wiring behind, pointing at anchors that
    // are gone. Swept here, at the one removal in the program, rather than left
    // for `edges()` to walk into.
    let swept = state.root_graph().minus_dangling_edges();
    *state.root_graph_mut() = swept;
    // Removing a node can shrink a constraint-less input that fed off it, so
    // shapes have to be recomputed, not just the layout.
    state.resettle();
    true
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
        // The one kind that *needs* an occupied cell: a Pattern is added below
        // the Pattern the caret stands on. In practice that is always its gap
        // cell — the other one names the arm's type, so INSERT there edits
        // instead of building — and the gap belongs to the Pattern precisely so
        // that this test can find it.
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
        // The mirror image, on the same row one scope in. A Tunnel is what a
        // Source is to the root: where a value enters. The root has no
        // enclosing graph to be tunnelled from, which is why this is the one
        // kind refused *because* the scope is the root — and `node_at`
        // refuses the branch's own (0,0,0), since the BranchSource already
        // stands on it.
        AddKind::Tunnel => {
            graph.node_at(local).is_none() && !is_root_scope && local.z == 0 && local.y == 0
        }
        _ => graph.node_at(local).is_none() && local.z != 0,
    }
}

/// Create a node of this kind at the caret. `None` when nothing was built.
///
/// For most kinds the caret does not follow: it keeps addressing the cell,
/// which now holds the new node. The five kinds that come into the world
/// unfinished are the exception — there the caret ends up on the cell that
/// names the missing property, because that is where it has to be answered, and
/// the `PendingEdit` that comes back is what an Escape undoes. For three of
/// them that means moving; a Source and a Tunnel are built on that cell
/// already.
fn insert_node_kind(
    state: &mut GraphState,
    pick: &mut PickState,
    kind: &AddKind,
) -> Option<Inserted> {
    // The caret's scope is the editing target — there is nothing else to
    // agree with, so no context guard is needed here.
    let scope = state.scope_of_caret(pick)?;
    let caret_before = pick.selected_pos;
    let node_id_domain = state.node_id_domain.clone();
    let anchor_id_domain = state.anchor_id_domain.clone();
    let scope_graph = state.root_graph().resolve_context(&scope.path);
    if !kind_allowed(scope_graph, scope.path.is_empty(), scope.local, kind) {
        return None;
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
        AddKind::Tunnel => scope_graph.plus_tunnel(new_pos, node_id_domain, anchor_id_domain),
        AddKind::FunctionCall(function_declaration_id) => {
            // The declaration decides the call's arity — `plus_function_call`
            // mints one input anchor per parameter — so a kind naming a
            // function the catalogue does not hold builds nothing.
            let declaration = state.function_declarations.get(function_declaration_id)?;
            scope_graph.plus_function_call(
                (function_declaration_id.clone(), declaration),
                new_pos,
                node_id_domain,
                anchor_id_domain,
            )
        }
        // Built without a type at all, rather than with a placeholder one: the
        // caret lands on the cell that names it and INSERT stays on, so it is
        // either typed in the next keystrokes or the node goes away with the
        // Escape. Nothing that was never chosen is ever shown.
        AddKind::TypeCast => {
            scope_graph.plus_type_cast(None, new_pos, node_id_domain, anchor_id_domain)
        }
        AddKind::Match => scope_graph.plus_match(new_pos, node_id_domain, anchor_id_domain),
        AddKind::Pattern => {
            // `kind_allowed` already established that the caret stands on a
            // Pattern — on its gap cell, which is the address for this. The
            // selected arm keeps its row and the new one goes below it, which
            // is where the caret follows to.
            let id = scope_graph.node_at(scope.local)?;
            scope_graph.plus_pattern_below(&id, node_id_domain, anchor_id_domain)
        }
    };
    if let Some(target) = state.root_graph_mut().resolve_context_mut(&scope.path) {
        *target = new_layout;
    }
    state.node_id_domain = new_node_id_domain;
    state.anchor_id_domain = new_anchor_id_domain;
    state.resettle();

    // Where the property that is still missing is edited, relative to the cell
    // the node was built on. All four sit in the scope it was built in, so no
    // descent into a branch is needed.
    //
    // Two of the steps are `2 * Z` for the same reason: both a TypeCast and a
    // Match hold a gap open after their input anchor, so what names the type is
    // the second cell behind it, not the first. A new Pattern goes one row
    // below the gap the caret was standing on, and one cell further back again
    // to reach its own type. A Source and a Tunnel need no step at all: each is
    // built on the very cell that declares its type, so the caret is already
    // standing on the question.
    let step = match kind {
        AddKind::TypeCast | AddKind::Match => IVec3::Z * 2,
        AddKind::Pattern => IVec3::Y + IVec3::Z,
        AddKind::Source | AddKind::Tunnel => IVec3::ZERO,
        _ => return Some(Inserted::Done),
    };
    pick.selected_pos = state.root_graph().clamp_to_volume(caret_before + step);
    // Read back rather than threaded out of the builders: every one of them
    // mints its ids internally, and the node standing on the property cell is
    // the node that was just built — for a Match that is its Pattern, which is
    // both what the caret addresses and what removing takes the Match with.
    //
    // If it cannot be found the node is still there and the graph still
    // changed; only the offer to unmake it is lost, and saying `Done` is the
    // reading that leaves nothing dangling.
    let Some(node) = state
        .caret_graph(pick)
        .and_then(|(layout, local)| layout.node_at(local))
    else {
        return Some(Inserted::Done);
    };
    Some(Inserted::Pending(PendingEdit { node, caret_before }))
}

/// Declare a new type — and, where the row named a literal, that literal — on
/// whichever node carries one. Returns whether the graph changed, so the caller
/// knows whether to flag a rebuild.
///
/// Type and literal are written together, which is what lets one keystroke say
/// both: `42` on a TypeCast makes it produce the integer 42, `Integer` makes it
/// cast to integers. The two used to be two acts, a dropdown and a checkbox,
/// and the half-state between them — a literal pinned but not yet typed — is
/// gone with them.
fn set_node_type(
    state: &mut GraphState,
    node_id: &model::node::Id,
    choice: TypeChoice,
    value: Option<String>,
) -> bool {
    let node = state
        .layout_graph
        .find_node_graph_mut(node_id)
        .and_then(|a| a.graph.nodes.get_mut(node_id));
    // Two shapes of the same field: a Constant always carries a type, because
    // the literal that built it *is* one, while a Source, a TypeCast and a
    // Pattern carry one only once it has been chosen. Committing a row is
    // exactly the act that makes the second into the first.
    match node {
        Some(model::node::ENode::Constant { r#type, .. }) => {
            *r#type = make_etype(choice, value);
            // Anchor heights follow declared types, so a type change reshapes
            // the node and its neighbours.
            state.resettle();
            true
        }
        Some(model::node::ENode::Source { r#type, .. })
        | Some(model::node::ENode::TypeCast { r#type, .. })
        | Some(model::node::ENode::Pattern { r#type, .. })
        | Some(model::node::ENode::Tunnel { r#type, .. }) => {
            *r#type = Some(make_etype(choice, value));
            state.resettle();
            true
        }
        _ => false,
    }
}

/// Rename a Source. The name is written along the body, so it decides how many
/// cells that body claims — renaming reshapes the node exactly the way retyping
/// a value does.
fn set_source_name(state: &mut GraphState, node_id: &model::node::Id, name: &str) -> bool {
    let node = state
        .layout_graph
        .find_node_graph_mut(node_id)
        .and_then(|a| a.graph.nodes.get_mut(node_id));
    match node {
        Some(model::node::ENode::Source { name: current, .. }) => {
            *current = name.to_string();
            state.resettle();
            true
        }
        _ => false,
    }
}

/// Commit a prompt row. `None` when nothing changed.
///
/// Every setter takes the node from the **caret**, not from the row, for the
/// same reason `Create` does: the caret is what the prompt is about, so a row
/// can never point at something that has moved on since it was drawn. A row
/// whose action does not fit the cell the caret now stands on commits nothing.
///
/// Only `Create` can come back `Pending` — answering a property is what *ends*
/// a pending node, never what starts one.
fn apply_prompt_action(
    state: &mut GraphState,
    pick: &mut PickState,
    action: &PromptAction,
) -> Option<Inserted> {
    let target = insert_target(state, pick);
    let changed = match (action, target) {
        (PromptAction::Create(kind), _) => return insert_node_kind(state, pick, kind),
        (PromptAction::SetName(name), InsertTarget::Edit(id, EditTarget::SourceName)) => {
            set_source_name(state, &id, name)
        }
        (
            PromptAction::SetType(choice, value),
            InsertTarget::Edit(
                id,
                EditTarget::SourceType
                | EditTarget::ConstantValue
                | EditTarget::CastType
                | EditTarget::PatternType
                | EditTarget::TunnelType,
            ),
        ) => set_node_type(state, &id, *choice, value.clone()),
        _ => false,
    };
    changed.then_some(Inserted::Done)
}

/// The mode control's two halves are the mouse's way between the modes, and
/// each half is the key it stands for: entering is what `i` does, leaving is
/// what `Escape` does, down to the placeholder a cancelled insert drops.
fn handle_mode_toggle(
    interaction_q: Query<(&Interaction, &ModeToggle), Changed<Interaction>>,
    eval: Res<EvalState>,
    mut state: ResMut<GraphState>,
    mut pick: ResMut<PickState>,
    mut prompt: ResMut<InsertPrompt>,
    mut pending: ResMut<PendingNode>,
    mut mode: ResMut<EditorMode>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    if is_evaluating(&eval) {
        return;
    }
    for (interaction, toggle) in interaction_q.iter() {
        // Guarded so a press on the half that is already on doesn't mark
        // `EditorMode` changed for nothing.
        if *interaction != Interaction::Pressed || toggle.0 == *mode {
            continue;
        }
        match toggle.0 {
            EditorMode::Insert => {
                prompt.clear();
                *mode = EditorMode::Insert;
                // The caret is drawn per mode, and it is a scene entity.
                rebuild.0 = true;
            }
            EditorMode::Normal => {
                leave_insert_mode(
                    &mut state,
                    &mut pick,
                    &mut prompt,
                    &mut pending,
                    &mut mode,
                    &mut rebuild,
                );
            }
        }
    }
}

/// Clicking a suggestion is the same act as `Enter` on it — without this the
/// mode control would hand a mouse user a list they cannot use.
fn handle_insert_prompt_click(
    interaction_q: Query<(&Interaction, &InsertPromptOption), Changed<Interaction>>,
    eval: Res<EvalState>,
    mut state: ResMut<GraphState>,
    // Written, not read: a kind that comes into the world unfinished moves the
    // caret onto the cell that finishes it.
    mut pick: ResMut<PickState>,
    mut prompt: ResMut<InsertPrompt>,
    mut pending: ResMut<PendingNode>,
    mut mode: ResMut<EditorMode>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    if is_evaluating(&eval) {
        return;
    }
    for (interaction, option) in interaction_q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if let Some(outcome) = apply_prompt_action(&mut state, &mut pick, &option.0) {
            commit_outcome(
                &state,
                &mut pick,
                outcome,
                &mut prompt,
                &mut pending,
                &mut mode,
                &mut rebuild,
            );
        }
    }
}

/// Leaving INSERT without committing anything — the one act behind `Escape`
/// and behind a click on the control's NORMAL half.
///
/// What was typed is dropped: there is no half-written name worth keeping. And
/// where what stands under the caret was built a moment ago and is still
/// waiting for its one mandatory property, dropping what was typed means
/// dropping the node: a cancelled insert leaves no placeholder behind, and the
/// caret goes back to the cell it was standing on.
///
/// Says whether it did drop one, which is the one case where the graph a
/// caller was still holding questions about is gone underneath it.
fn leave_insert_mode(
    state: &mut GraphState,
    pick: &mut PickState,
    prompt: &mut InsertPrompt,
    pending: &mut PendingNode,
    mode: &mut EditorMode,
    rebuild: &mut NeedsRebuild,
) -> bool {
    if let Some(edit) = pending.0.take() {
        if remove_node(state, &edit.node) {
            pick.selected_pos = state.root_graph().clamp_to_volume(edit.caret_before);
            prompt.clear();
            *mode = EditorMode::Normal;
            rebuild.0 = true;
            return true;
        }
    }
    prompt.clear();
    if *mode != EditorMode::Normal {
        *mode = EditorMode::Normal;
        // The caret is drawn per mode, and it is a scene entity.
        rebuild.0 = true;
    }
    false
}

/// The next property of the panel after the one the caret stands on, and
/// nothing else: gaps are skipped, and there is no wrap.
///
/// `TAB` walks every row and comes round again, because walking is what it is
/// for. This is the other thing — what is left to answer after an answer — and
/// a place to build is not something left to answer, nor is the row that was
/// just filled in.
fn next_property_cell(state: &GraphState, pick: &PickState) -> Option<IVec3> {
    let subject = panel_subject(state, pick)?;
    let here = caret_row_address(state, pick);
    let index = subject.rows.iter().position(|row| row.address == here)?;
    subject
        .rows
        .iter()
        .skip(index + 1)
        .find(|row| matches!(row.address, RowAddress::Property(..)))
        .map(|row| row.cell)
}

/// What follows a committed row, whichever key or click committed it.
///
/// INSERT stays on while there is anything left to say about the node the
/// caret stands on — first whatever the new node still owes, then every
/// property of it that has not been reached yet. A Source is the whole rule in
/// one: built, it owes a type; typed, the name is next; named, there is nothing
/// further and INSERT ends. It is `TAB` without the wrap, which is what makes
/// it an end rather than a round.
fn commit_outcome(
    state: &GraphState,
    pick: &mut PickState,
    outcome: Inserted,
    prompt: &mut InsertPrompt,
    pending: &mut PendingNode,
    mode: &mut EditorMode,
    rebuild: &mut NeedsRebuild,
) {
    prompt.clear();
    rebuild.0 = true;
    // A node that came into the world unfinished has already had the caret put
    // on the cell that names what it owes, and that cell is where the answer
    // has to be given — so there is nowhere else to go first.
    let stay = matches!(outcome, Inserted::Pending(_));
    // Answering a property is what finishes a node, so any commit clears the
    // mark — and a new one only ever comes from a `Create`.
    pending.0 = match outcome {
        Inserted::Pending(edit) => Some(edit),
        Inserted::Done => None,
    };
    if stay {
        return;
    }
    // Read before the caret is written, so the answer is about where it stood.
    let next = next_property_cell(state, pick);
    match next {
        Some(cell) => pick.selected_pos = state.root_graph().clamp_to_volume(cell),
        // Nothing further to say about it: a Constant *is* its literal and a
        // call *is* its function, both already given by the row that built
        // them, and the last property of a node is the last thing it has.
        None => *mode = EditorMode::Normal,
    }
}

/// The suggestions the prompt currently offers — which is a question about the
/// caret's cell first and about the typed text second. Each cell names one
/// property, so each answers with the rows that property can take.
///
/// The prefix filters, the legality only greys — those are two different
/// questions, and typing must not make a row silently disappear because the
/// caret happens to stand somewhere it is not allowed. The typed literal is
/// the exception that proves it: there the prefix is not something to filter
/// by, it *is* the row.
fn prompt_candidates(state: &GraphState, pick: &PickState, text: &str) -> Vec<Suggestion> {
    match insert_target(state, pick) {
        InsertTarget::Create => create_candidates(state, pick, text),
        // A name answers to no list. The one row is the text itself, which
        // keeps what `Return` commits visible in the place it is visible for
        // every other property.
        InsertTarget::Edit(_, EditTarget::SourceName) => vec![Suggestion {
            action: PromptAction::SetName(text.to_string()),
            label: text.to_string(),
            detail: "Name".to_string(),
            // An empty name is a name: a Source is identified by its index,
            // and clearing the name has to stay possible.
            allowed: true,
        }],
        // A declared type is a closed set of five, all of them always legal —
        // the node already stands there, so nothing about the caret can forbid
        // one. No literal, and for the same reason on both: what a Source and
        // a Tunnel declare is the *shape* of a value that arrives from
        // somewhere else, never the value.
        InsertTarget::Edit(_, EditTarget::SourceType | EditTarget::TunnelType) => type_rows(text),
        // A Constant *is* its literal, so only literals are offered. Naming a
        // bare type here would build a constant with nothing in it, which
        // `eval_value_for_type` refuses anyway.
        InsertTarget::Edit(_, EditTarget::ConstantValue) => literal_rows(text),
        // Both at once, and that is the whole point of the cell: `Integer`
        // casts to integers, `42` produces the integer 42. A cast has one
        // answer it cannot give, which is `none`.
        InsertTarget::Edit(_, EditTarget::CastType) => refuse_none_cast(type_or_literal_rows(text)),
        InsertTarget::Edit(_, EditTarget::PatternType) => type_or_literal_rows(text),
    }
}

/// A base type or a literal, in one list — what a TypeCast's and a Pattern's
/// cell can both be set to.
fn type_or_literal_rows(text: &str) -> Vec<Suggestion> {
    literal_rows(text)
        .into_iter()
        .chain(type_rows(text))
        .collect()
}

/// Refuse `none` as a cast target.
///
/// A cast to `none` would ignore whatever flows in and hand back a fixed
/// `none` — which is a Constant, spelled the long way round and with an input
/// anchor that means nothing. A Pattern is the opposite case and keeps it: an
/// arm that matches `none` is how the sad path is caught.
///
/// Greyed rather than dropped, because those are two different questions: the
/// prefix filters and the legality only greys. A row that vanished as the word
/// was typed would leave the user looking for a typo.
fn refuse_none_cast(rows: Vec<Suggestion>) -> Vec<Suggestion> {
    rows.into_iter()
        .map(|mut row| {
            if matches!(row.action, PromptAction::SetType(TypeChoice::None, _)) {
                row.allowed = false;
                row.detail = "casts nothing".to_string();
            }
            row
        })
        .collect()
}

/// Shift reports the uppercase character and the labels are CamelCase, so every
/// prefix test has to ignore case in both directions.
fn matches_prefix(label: &str, text: &str) -> bool {
    label
        .to_ascii_lowercase()
        .starts_with(&text.to_ascii_lowercase())
}

/// The five base types, as rows that declare one.
fn type_rows(text: &str) -> Vec<Suggestion> {
    TYPE_CHOICES
        .iter()
        .filter(|choice| matches_prefix(type_choice_label(**choice), text))
        .map(|choice| Suggestion {
            action: PromptAction::SetType(*choice, None),
            label: type_choice_label(*choice).to_string(),
            detail: String::new(),
            allowed: true,
        })
        .collect()
}

/// The literal the text spells, if any, and the three that are written as a
/// word — as rows that *set* a property rather than build a node. The create
/// prompt spells the same two sources of literals its own way, because there
/// they build a Constant instead.
fn literal_rows(text: &str) -> Vec<Suggestion> {
    let typed = typed_literal(text).map(|(choice, raw, _closed)| {
        let r#type = make_etype(choice, Some(raw.clone()));
        let parses = eval::EValue::parse(&r#type, &raw).is_ok();
        Suggestion {
            action: PromptAction::SetType(choice, Some(raw)),
            label: if parses {
                r#type.to_string()
            } else {
                text.to_string()
            },
            detail: literal_type_name(choice),
            allowed: parses,
        }
    });
    typed
        .into_iter()
        .chain(
            LITERAL_KEYWORDS
                .iter()
                .filter(|(label, _, _)| matches_prefix(label, text))
                .map(|(label, choice, value)| Suggestion {
                    action: PromptAction::SetType(*choice, value.map(str::to_string)),
                    label: label.to_string(),
                    detail: literal_type_name(*choice),
                    allowed: true,
                }),
        )
        .collect()
}

/// What the prompt offers on a cell that names nothing: the node kinds, the
/// literals and every declared function, each with whether it may be created
/// where the caret stands.
fn create_candidates(state: &GraphState, pick: &PickState, text: &str) -> Vec<Suggestion> {
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
    // list between frames. Sorted by name — byte order, which puts the symbols
    // ahead of the words and leaves `||` behind `substr`.
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
            action: PromptAction::Create(kind),
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
                    action: PromptAction::Create(kind),
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
fn signature_detail(declaration: &model::function_declaration::FunctionDeclaration) -> String {
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
/// in one of them. `type_choice_label` keeps the long word for the rows that
/// *are* a type, where it is the thing being named rather than an aside.
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

/// Step the highlight to the next committable suggestion. Rows that are greyed
/// out can never be committed, so the highlight does not stop on them.
///
/// It walks in one direction and **does not wrap**: stepping off either end
/// leaves the list, and with nothing highlighted `Return` is the editor's
/// newline again. That is the way back out, and a wrapping list would have none
/// short of `Escape`. From outside the list the two ends are one keypress away —
/// down enters at the first row, up at the last.
fn step_selection(candidates: &[Suggestion], from: Option<usize>, delta: isize) -> Option<usize> {
    let len = candidates.len();
    if len == 0 {
        return None;
    }
    let mut index = match from {
        Some(index) => index.min(len - 1) as isize + delta,
        None if delta > 0 => 0,
        None => len as isize - 1,
    };
    while (0..len as isize).contains(&index) {
        if candidates[index as usize].allowed {
            return Some(index as usize);
        }
        index += delta;
    }
    None
}

/// Where the highlight lands after the text changed: the first suggestion that
/// can actually be committed — and none at all while nothing is typed, because
/// nothing typed is nothing chosen. On a create prompt a highlight there would
/// also take `Return` away from the column it opens; on a node that was built a
/// keystroke ago and is waiting for its type, it would answer the question with
/// whatever row happened to sort first.
///
/// The one exception is a row that *is* the text. A Source's name answers to no
/// list: its single row carries whatever has been typed, and an empty name is a
/// name — a Source is told apart by its index, so clearing the name has to stay
/// committable. So an empty prompt highlights a row whose label is empty too,
/// and nothing else.
///
/// The rule lives here rather than in the callers, so there is one place that
/// decides it.
fn first_selection(text: &str, candidates: &[Suggestion]) -> Option<usize> {
    let index = candidates
        .iter()
        .position(|suggestion| suggestion.allowed)?;
    (!text.is_empty() || candidates[index].label.is_empty()).then_some(index)
}

/// Re-ask the list after the text changed. Every key that edits the text ends
/// in this, and it is a function rather than six copies because the list
/// depends on the caret as much as on the text, and that pair is easy to get
/// half right.
fn reselect(prompt: &mut InsertPrompt, state: &GraphState, pick: &PickState) {
    let candidates = prompt_candidates(state, pick, &prompt.text);
    prompt.selected = first_selection(&prompt.text, &candidates);
}

/// The highlight as it stands against *this* list. Clamped on read rather than
/// on write: a caret move under a standing prompt can shorten the list, and a
/// stale index must not survive that. Sole reader of `InsertPrompt::selected`,
/// so the row that is drawn highlighted and the row `Return` commits can never
/// come apart.
fn clamped_selection(candidates: &[Suggestion], selected: Option<usize>) -> Option<usize> {
    selected
        .filter(|_| !candidates.is_empty())
        .map(|index| index.min(candidates.len() - 1))
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

/// The prompt's suggestion rows, windowed around the highlight. Shared by the
/// prompt that creates nodes and the ones that change a property: the rows are
/// the same rows, only the container differs — which is why this takes no
/// marker of its own and the caller tags its own container.
fn spawn_prompt_rows(
    options: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    candidates: &[Suggestion],
    selected: Option<usize>,
) {
    // With no highlight the window has nothing to follow, so it sits where the
    // list starts — which is where the first arrow keypress enters it.
    let window = prompt_window(candidates.len(), selected.unwrap_or(0));
    // What the window hides is said, not swallowed: the list can be long
    // enough that a silent cut would read as "that is all there is".
    if window.start > 0 {
        spawn_prompt_hint(options, font, format!("… {} more above", window.start));
    }
    for (index, suggestion) in candidates
        .iter()
        .enumerate()
        .take(window.end)
        .skip(window.start)
    {
        // Only a committable row can hold the highlight, so a greyed one never
        // looks like the answer to `Enter`.
        let highlighted = selected == Some(index) && suggestion.allowed;
        let label_color = if suggestion.allowed {
            Color::srgb(0.85, 0.85, 0.9)
        } else {
            Color::srgb(0.35, 0.35, 0.4)
        };
        let mut entity = options.spawn((
            Node {
                padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                column_gap: Val::Px(12.0),
                ..default()
            },
            BackgroundColor(if highlighted { ROW_HOT } else { FILL_NONE }),
        ));
        // A greyed row is not clickable at all, for the same reason the "… N
        // more" tally is not: what cannot be committed must not be committable
        // by another route. `Enter` filters on `allowed`, but the click path
        // commits whatever the row carries — and only the *create* actions are
        // checked a second time, inside `insert_node_kind`.
        if suggestion.allowed {
            // A quieter lift than the highlight, and deliberately so: in this
            // list the full `ROW_HOT` already means *this is what `Enter`
            // takes*, and a pointer that said the same thing would be a second
            // answer to a question with one. The highlighted row still lifts,
            // just barely, so that hovering it is not the one place the
            // pointer goes unanswered.
            entity.insert((
                Button,
                InsertPromptOption(suggestion.action.clone()),
                HoverFill {
                    rest: if highlighted { ROW_HOT } else { FILL_NONE },
                    hot: if highlighted {
                        Color::srgba(0.24, 0.24, 0.34, 0.95)
                    } else {
                        Color::srgba(0.16, 0.16, 0.24, 0.7)
                    },
                },
            ));
        }
        entity.with_children(|row| {
            row.spawn((
                Text::new(suggestion.label.clone()),
                text_font(font, 14.0),
                TextColor(label_color),
            ));
            // Spawned only when it says something — `SpaceBetween` already
            // puts a lone child at the start, so a row without a detail
            // needs no empty placeholder to stay left-aligned.
            if !suggestion.detail.is_empty() {
                row.spawn((
                    Text::new(suggestion.detail.clone()),
                    text_font(font, 12.0),
                    TextColor(if suggestion.allowed {
                        Color::srgb(0.6, 0.6, 0.7)
                    } else {
                        label_color
                    }),
                    // The signature is the row's width, not its slack:
                    // shrinking it would wrap `Char|String,Char|String`
                    // onto a second line and make the rows uneven.
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

// ── Editor panel UI ─────────────────────────────────────────

/// Build the left column: the two address lines, then the panel under them.
///
/// `top: 56.0` clears the New/Help row above — 12 px down, and about 33 high
/// with `chrome_button_node`'s padding around 14 px text.
///
/// Spanned from edge to edge rather than given the panel's width, because the
/// relative line is as long as the caret is deep and there is no reason to
/// fold it while the screen is still going. The panel states its own 280 and
/// so keeps it; the lines take what is left, and wrap at the far margin.
fn spawn_editor_column(mut commands: Commands, ui_font: Res<UiFont>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(56.0),
                left: Val::Px(12.0),
                right: Val::Px(12.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                ..default()
            },
            EditorColumn,
            EditorChrome,
        ))
        .with_children(|column| {
            // Dimmed: the path is the context the address is read in.
            column.spawn((
                Text::new(""),
                text_font(&ui_font.0, 12.0),
                TextColor(Color::srgba(0.55, 0.55, 0.65, 0.9)),
                AddressLine::Relative,
            ));
            column.spawn((
                Text::new(""),
                text_font(&ui_font.0, 12.0),
                TextColor(Color::srgb(0.85, 0.85, 0.9)),
                AddressLine::Absolute,
            ));
            // The panel's frame, spawned once and never taken down. Empty at
            // startup — `sync_editor_panel` fills it. It flows under the lines
            // rather than standing at a `top` of its own, so however far they
            // wrapped is how far down it begins. `Button` on the root so
            // `pick_nodes`' `over_ui` test covers it and a click on the panel
            // doesn't move the caret to whatever cell lies behind it.
            column.spawn((
                Node {
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
                EditorPanel,
            ));
        });
}

#[derive(Default, PartialEq, Eq, Clone)]
struct EditorPanelFingerprint {
    /// The whole drawn view, values included — every row's text is in here, so
    /// a commit that changes one under a standing panel moves the fingerprint.
    subject: Option<PanelSubject>,
    /// Which row the caret stands on. Moving between two cells of the *same*
    /// node changes only this, which is why it is not folded into the subject.
    focus: Option<usize>,
    /// The prompt's text, cursor and highlight, but only while the panel is
    /// drawing it. That is also the bit that catches the NORMAL→INSERT switch,
    /// where nothing else in here moves.
    prompt: Option<(String, usize, Option<usize>)>,
    candidates: Vec<Suggestion>,
    visible: bool,
}

/// The panel beside the caret: the node the caret stands on, every property
/// that node has, and the addressed one picked out — outlined in NORMAL, typed
/// into in INSERT.
///
/// One panel and not two. It also carries the create prompt, because a cell
/// that names no property is still a cell of the same volume, and standing
/// there is still standing somewhere: on the gap between two arms of a Match it
/// is the row that would open, and on an empty cell it is the panel's whole
/// content.
///
/// Sole writer of the panel's `display`, so the modals are folded in here
/// rather than left to `EditorChrome`; contents are respawned only when the
/// fingerprint moves.
fn sync_editor_panel(
    mut commands: Commands,
    state: Res<GraphState>,
    pick: Res<PickState>,
    eval: Res<EvalState>,
    screenshot: Res<ScreenshotMode>,
    mode: Res<EditorMode>,
    prompt: Res<InsertPrompt>,
    ui_font: Res<UiFont>,
    mut panel_q: Query<(Entity, &mut Node), With<EditorPanel>>,
    panel_children_q: Query<Entity, With<EditorPanelEntity>>,
    mut cache: Local<EditorPanelFingerprint>,
) {
    let subject = panel_subject(&state, &pick);
    let here = caret_row_address(&state, &pick);
    let focus = subject
        .as_ref()
        .and_then(|s| s.rows.iter().position(|row| row.address == here));
    // Where the caret's cell is not a row the panel drew — an empty cell, or a
    // hole in a node that names nothing — the create prompt stands on its own
    // at the foot of the panel instead of inside a row.
    let loose_prompt =
        *mode == EditorMode::Insert && focus.is_none() && matches!(here, RowAddress::ArmGap(_));
    let visible = (subject.is_some() || loose_prompt)
        && !modal_is_open(&eval)
        && !is_evaluating(&eval)
        && !screenshot.active();
    let editing = visible && *mode == EditorMode::Insert;
    let candidates = if editing {
        prompt_candidates(&state, &pick, &prompt.text)
    } else {
        Vec::new()
    };
    let selected = clamped_selection(&candidates, prompt.selected);

    let fp = EditorPanelFingerprint {
        subject: subject.clone(),
        focus,
        prompt: editing.then(|| (prompt.text.clone(), prompt.cursor, selected)),
        candidates: candidates.clone(),
        visible,
    };
    if *cache == fp {
        return;
    }
    *cache = fp;

    for e in panel_children_q.iter() {
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
    commands.entity(panel_entity).with_children(|panel| {
        if let Some(subject) = &subject {
            spawn_editor_label(panel, font, &subject.heading);
            for (index, row) in subject.rows.iter().enumerate() {
                let focused = focus == Some(index);
                if matches!(row.address, RowAddress::ArmGap(_)) {
                    // A gap is a row at every arm for `TAB`'s sake; for the eye
                    // it opens only where the caret is standing in it.
                    if focused {
                        spawn_arm_gap(panel, font, editing, &prompt, &candidates, selected);
                    }
                    continue;
                }
                spawn_labeled_row(panel, font, &row.label, |slot| {
                    if focused && editing {
                        spawn_prompt_body(
                            slot,
                            font,
                            &prompt.text,
                            prompt.cursor,
                            &candidates,
                            selected,
                        );
                    } else if focused {
                        // The same box INSERT types in, only quiet: nothing
                        // moves on the switch, the border lights up and the
                        // text starts answering.
                        slot.spawn((
                            editor_text_input_node(),
                            BackgroundColor(Color::srgba(0.06, 0.06, 0.12, 0.95)),
                            BorderColor::all(Color::srgb(0.24, 0.40, 0.46)),
                        ))
                        .with_children(|boxed| {
                            boxed.spawn((
                                Text::new(row.value.clone()),
                                text_font(font, 14.0),
                                TextColor(Color::srgb(0.91, 0.89, 0.87)),
                            ));
                        });
                    } else {
                        // Dimmer than the addressed row, and no box: it is here
                        // to be read, and reaching it is what `TAB` is for.
                        slot.spawn((
                            Text::new(row.value.clone()),
                            text_font(font, 14.0),
                            TextColor(Color::srgb(0.64, 0.63, 0.62)),
                        ));
                    }
                });
            }
        }
        if loose_prompt {
            panel
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        ..default()
                    },
                    EditorPanelEntity,
                ))
                .with_children(|column| {
                    spawn_prompt_body(
                        column,
                        font,
                        &prompt.text,
                        prompt.cursor,
                        &candidates,
                        selected,
                    );
                });
        }
    });
}

/// The place a new arm would go: the line it would open on, and in INSERT the
/// prompt that builds it — which on this cell offers nothing but `Pattern`.
fn spawn_arm_gap(
    panel: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    editing: bool,
    prompt: &InsertPrompt,
    candidates: &[Suggestion],
    selected: Option<usize>,
) {
    panel
        .spawn((
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..default()
            },
            EditorPanelEntity,
        ))
        .with_children(|gap| {
            gap.spawn((
                Node {
                    height: Val::Px(1.0),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.133, 0.827, 0.933)),
            ));
            if editing {
                spawn_prompt_body(gap, font, &prompt.text, prompt.cursor, candidates, selected);
            }
        });
}

fn spawn_editor_label(panel: &mut ChildSpawnerCommands, font: &Handle<Font>, text: &str) {
    panel.spawn((
        Text::new(text),
        text_font(font, 15.0),
        TextColor(Color::srgb(0.75, 0.75, 0.9)),
        EditorPanelEntity,
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
            EditorPanelEntity,
        ))
        .with_children(|row| {
            row.spawn((
                Text::new(label.to_string()),
                text_font(font, 13.0),
                TextColor(Color::srgb(0.6, 0.6, 0.7)),
                // A label is one word and stays one line. `min_width` and not
                // `width`, because an arm's number grows with the arm count and
                // no fixed column is provably wide enough: the short labels
                // still line up on the same edge they always did, and a long
                // one takes the room it needs instead of breaking in half.
                TextLayout::new_with_no_wrap(),
                Node {
                    min_width: Val::Px(70.0),
                    flex_shrink: 0.0,
                    ..default()
                },
            ));
            row.spawn(Node {
                flex_grow: 1.0,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(6.0),
                ..default()
            })
            .with_children(widget);
        });
}

/// Shape of a one-line input: the prompt's own text line, and the modal fields
/// that still are real text boxes.
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

/// The prompt itself: the line being typed with the caret drawn at the cursor,
/// and the suggestion rows under it.
///
/// The same widget wherever it stands — in a property's row, on the line where
/// an arm would open, or alone in a panel that is about no node. It carries no
/// marker of its own, for the reason `spawn_prompt_rows` carries none: whatever
/// holds it is already tagged, and `despawn` takes its children with it.
fn spawn_prompt_body(
    parent: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    text: &str,
    cursor: usize,
    candidates: &[Suggestion],
    selected: Option<usize>,
) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            flex_grow: 1.0,
            min_width: Val::Px(0.0),
            row_gap: Val::Px(2.0),
            ..default()
        })
        .with_children(|column| {
            column
                .spawn((
                    editor_text_input_node(),
                    BackgroundColor(Color::srgba(0.06, 0.06, 0.12, 0.95)),
                    BorderColor::all(Color::srgb(0.133, 0.827, 0.933)),
                ))
                .with_children(|row| {
                    let (before, after) = text.split_at(cursor);
                    row.spawn((
                        Text::new(format!("{}|{}", before, after)),
                        text_font(font, 14.0),
                        // A prefix nothing answers to is shown on the text
                        // itself. An empty box below would read as a broken
                        // widget instead of as a refusal.
                        TextColor(if candidates.is_empty() {
                            Color::srgb(0.95, 0.30, 0.30)
                        } else {
                            Color::srgb(0.91, 0.89, 0.87)
                        }),
                    ));
                });
            if candidates.is_empty() {
                return;
            }
            column
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        padding: UiRect::all(Val::Px(2.0)),
                        border_radius: BorderRadius::all(Val::Px(4.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.08, 0.08, 0.14, 0.98)),
                ))
                .with_children(|options| {
                    spawn_prompt_rows(options, font, candidates, selected);
                });
        });
}

/// Begin a run: the checks a press has to pass, and whichever phase comes of
/// them — the values modal where there are Sources to answer for, otherwise the
/// run itself, at step 0.
///
/// A free function because two buttons start a run now. `▶` takes a step at a
/// time from here on, `▶▌` goes to the end, but what it takes to *start* is the
/// same question asked once.
fn begin_evaluation(
    eval: &mut EvalState,
    state: &GraphState,
    diagnostics: &Diagnostics,
    open: &mut DiagnosticsOpen,
) {
    if is_evaluating(eval) {
        // Already showing a modal or running — ignore.
        return;
    }
    // What used to be one question asked here — is anything wired to the sink —
    // is now the whole of what `lint` finds, and it is asked of the graph
    // standing still rather than discovered a step into the run.
    //
    // No modal. A modal names one thing, has to be dismissed before the graph
    // can be looked at, and would take the list with it; the panel is already
    // on screen, says all of them at once, and each row goes to the cell it is
    // about. Unfolding it is the whole of the answer — the press is not
    // ignored, it is answered somewhere the answer can be acted on.
    if diagnostics.blocking() {
        open.0 = true;
        return;
    }
    // Flatten pattern sub-scenes in so eval and var-decl collection see every
    // node, not just the program-level ones.
    let graph = state.root_graph().flattened_graph();
    let sources = infer::collect_sources(&graph);
    if !sources.is_empty() {
        eval.phase = EvalPhase::SourcePrompt {
            inputs: sources
                .into_iter()
                .map(|(id, _name)| (id, String::new()))
                .collect(),
        };
        return;
    }
    // A program with no Sources still opens on step 0. There is nothing waiting
    // at its edges to look at, which is what a program given nothing looks like
    // — and the alternative is starting such a run one step in while every other
    // run starts at zero, so that the number on screen would mean two things.
    eval.phase = EvalPhase::Running {
        states: vec![eval::State::nothing_yet()],
        current: 0,
        user_source_values: std::collections::HashMap::new(),
    };
}

/// Nothing but the press. What the button *looks* like is
/// `update_step_button_visuals`' business, along with the rest of the row —
/// see the note there about why the tint used to live in each handler and no
/// longer does.
fn handle_evaluate_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<EvaluateButton>)>,
    mut eval: ResMut<EvalState>,
    state: Res<GraphState>,
    diagnostics: Res<Diagnostics>,
    mut open: ResMut<DiagnosticsOpen>,
) {
    if interaction_q.iter().any(|i| *i == Interaction::Pressed) {
        begin_evaluation(&mut eval, &state, &diagnostics, &mut open);
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
    ConfirmNew,
    SourcePrompt,
}

fn modal_kind(phase: &EvalPhase) -> ModalKind {
    match phase {
        EvalPhase::ErrorModal(_) => ModalKind::Error,
        EvalPhase::ControlsModal => ModalKind::Controls,
        EvalPhase::ConfirmNew => ModalKind::ConfirmNew,
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
        EvalPhase::ConfirmNew => {
            spawn_confirm_new_modal(&mut commands, &ui_font.0);
        }
        EvalPhase::SourcePrompt { inputs } => {
            let root = state.root_graph();
            let graph = root.flattened_graph();
            // Asked for by index as well as by name, and listed in the index's
            // order: the index is the one label every Source is certain to
            // have — a name may be empty and two Sources may carry the same
            // one — so it leads, and the same number stands on the node itself.
            let mut rows: Vec<(Option<usize>, model::node::Id, String)> = inputs
                .iter()
                .map(|(id, _)| {
                    let name = match graph.nodes.get(id) {
                        Some(model::node::ENode::Source { name, .. }) => name.clone(),
                        _ => "?".to_string(),
                    };
                    (root.source_index(id), id.clone(), name)
                })
                .collect();
            // A Source with no index cannot happen — every one of them is laid
            // out in the root scope — but were it to, it sorts last rather than
            // first, where it would renumber what the eye already read.
            rows.sort_by(|(a_index, a_id, _), (b_index, b_id, _)| {
                a_index
                    .unwrap_or(usize::MAX)
                    .cmp(&b_index.unwrap_or(usize::MAX))
                    .then_with(|| a_id.cmp(b_id))
            });
            spawn_source_modal(
                &mut commands,
                &ui_font.0,
                rows.into_iter()
                    .map(|(index, id, name)| {
                        let label = match index {
                            Some(index) if name.is_empty() => format!("[{}]", index),
                            Some(index) => format!("[{}] {}", index, name),
                            None => name,
                        };
                        (id, label)
                    })
                    .collect(),
            );
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
                    .spawn((modal_button(), ModalOkButton, ModalEntity))
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

/// The question a New has to get through. Same frame as the error modal, two
/// buttons instead of one, and the affirmative one named after what it does
/// rather than after agreeing — "Discard" is the fact of the matter, and the
/// graph is not recoverable once it is pressed.
fn spawn_confirm_new_modal(commands: &mut Commands, font: &Handle<Font>) {
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
                    Text::new("New"),
                    text_font(font, 20.0),
                    TextColor(Color::srgb(0.95, 0.95, 1.0)),
                    Node {
                        margin: UiRect::all(Val::Px(12.0)),
                        align_self: AlignSelf::Center,
                        ..default()
                    },
                    ModalEntity,
                ));
                panel.spawn((
                    Text::new("Discard the current graph and start over?"),
                    text_font(font, 16.0),
                    TextColor(Color::srgb(0.85, 0.85, 0.9)),
                    Node {
                        margin: UiRect::all(Val::Px(12.0)),
                        ..default()
                    },
                    ModalEntity,
                ));
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
                        btns.spawn((modal_button(), ConfirmNewButton, ModalEntity))
                            .with_children(|b| {
                                b.spawn((
                                    Text::new("Discard"),
                                    text_font(font, 14.0),
                                    TextColor(Color::srgb(0.85, 0.85, 0.9)),
                                    ModalEntity,
                                ));
                            });
                        btns.spawn((modal_button(), ModalCancelButton, ModalEntity))
                            .with_children(|b| {
                                b.spawn((
                                    Text::new("Cancel"),
                                    text_font(font, 14.0),
                                    TextColor(Color::srgb(0.85, 0.85, 0.9)),
                                    ModalEntity,
                                ));
                            });
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
        ("i", "INSERT: build here, or edit what stands here"),
        ("Up/Down", "INSERT: walk the suggestion list"),
        ("Left/Right, Home/End", "INSERT: move the text cursor"),
        ("Space", "INSERT: open a cell behind the caret"),
        (
            "Return",
            "INSERT: commit the highlighted row, else open a column (X)",
        ),
        ("Shift + Return", "INSERT: open a row (Y)"),
        ("Escape", "Leave INSERT"),
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
                        btns.spawn((modal_button(), ModalOkButton, ModalEntity))
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
                                    // Wide enough for the index that now leads
                                    // the name: at this size the bracketed
                                    // number costs about four characters, and
                                    // a column that cannot hold it wraps the
                                    // label onto a second line and pulls the
                                    // field beside it out of line.
                                    width: Val::Px(150.0),
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
                        btns.spawn((modal_button(), ModalCancelButton, ModalEntity))
                            .with_children(|b| {
                                b.spawn((
                                    Text::new("Cancel"),
                                    text_font(font, 14.0),
                                    TextColor(Color::srgb(0.85, 0.85, 0.9)),
                                ));
                            });
                        btns.spawn((modal_button(), ModalEvaluateButton, ModalEntity))
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

/// A button inside a modal: its shape, its fill, and how it answers the
/// pointer — a bundle rather than a `Node`, because all six of them are the
/// same button and the three things that made them one were being written out
/// six times. The margin is what separates it from `chrome_button_node`: a
/// modal's buttons stand in a row that sets no gap of its own.
fn modal_button() -> impl Bundle {
    (
        Button,
        Node {
            padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
            margin: UiRect::axes(Val::Px(6.0), Val::Px(0.0)),
            border_radius: BorderRadius::all(Val::Px(6.0)),
            ..default()
        },
        BackgroundColor(MODAL_REST),
        HoverFill {
            rest: MODAL_REST,
            hot: MODAL_HOT,
        },
        HoverInk {
            rest: INK_BRIGHT,
            hot: INK_HOT,
        },
    )
}

/// Ask before resetting. The question is a modal phase like any other, so
/// everything that already steps aside for a modal steps aside for this one
/// too; `handle_confirm_new_button` is what carries it out.
fn handle_new_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<NewButton>)>,
    mut eval: ResMut<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    if interaction_q.iter().any(|i| *i == Interaction::Pressed) {
        eval.phase = EvalPhase::ConfirmNew;
    }
}

fn handle_help_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<HelpButton>)>,
    mut eval: ResMut<EvalState>,
) {
    if is_evaluating(&eval) {
        return;
    }
    if interaction_q.iter().any(|i| *i == Interaction::Pressed) {
        eval.phase = EvalPhase::ControlsModal;
    }
}

/// The top bar's two buttons, and the one thing they could not say: that a run
/// has taken them.
///
/// Both handlers refuse to act while `is_evaluating`, and the bar is *not*
/// hidden for it — `sync_editor_chrome` steps aside for a modal or a
/// screenshot, and a `Running` phase is neither. So they stood there answering
/// the pointer and doing nothing. A `HoverFill` would have kept that promise;
/// this keeps the one `update_step_button_visuals` settled on for the
/// transport row instead: one writer, three states, the switched-off one
/// first, because it outranks wherever the pointer is.
///
/// No `Changed<Interaction>` filter: what turns these grey is the phase, and a
/// phase moves without the pointer moving with it.
fn sync_chrome_buttons(
    eval: Res<EvalState>,
    mut button_q: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        Or<(With<NewButton>, With<HelpButton>)>,
    >,
    mut text_color_q: Query<&mut TextColor>,
) {
    let enabled = !is_evaluating(&eval);
    for (interaction, mut bg, children) in button_q.iter_mut() {
        let (fill, ink) = match (enabled, interaction) {
            (false, _) => (CONTROL_OFF, INK_OFF),
            (true, Interaction::Hovered | Interaction::Pressed) => (CONTROL_HOT, INK_HOT),
            (true, Interaction::None) => (CONTROL_REST, INK_BRIGHT),
        };
        if bg.0 != fill {
            bg.0 = fill;
        }
        let Ok(mut color) = text_color_q.get_mut(children[0]) else {
            continue;
        };
        if color.0 != ink {
            color.0 = ink;
        }
    }
}

/// Carry out the New that was asked about.
///
/// Only the root sub-layout is replaced; the wrapper and its `root_id` stand,
/// which is what keeps the caret's addresses meaning the same thing. The id
/// domains are threaded through rather than reset, so an id minted before the
/// New can never collide with one minted after it.
fn handle_confirm_new_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<ConfirmNewButton>)>,
    mut state: ResMut<GraphState>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut pick: ResMut<PickState>,
    mut eval: ResMut<EvalState>,
) {
    for interaction in interaction_q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
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
        eval.phase = EvalPhase::Idle;
    }
}

/// Sole writer of every `EditorChrome` node's `display`.
///
/// Two reasons and one answer: a modal owns the screen, or the point is that
/// nothing of the editor is on it at all. Both say the same thing about the
/// chrome, so both are decided here rather than at each widget.
fn sync_editor_chrome(
    eval: Res<EvalState>,
    screenshot: Res<ScreenshotMode>,
    mut hideable: Query<&mut Node, With<EditorChrome>>,
    mut last_hidden: Local<Option<bool>>,
) {
    let hidden = modal_is_open(&eval) || screenshot.active();
    if *last_hidden == Some(hidden) {
        return;
    }
    *last_hidden = Some(hidden);
    let d = if hidden { Display::None } else { Display::Flex };
    for mut n in hideable.iter_mut() {
        n.display = d;
    }
}

/// What the button does, and nothing about how it looks: `modal_button` hands
/// every one of them a `HoverFill`, and `paint_hover` is their only painter. A
/// modal button is only on screen while its own modal is, so there is no state
/// here for a look to depend on — which is the whole condition `HoverFill`
/// states for itself.
fn handle_modal_ok_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<ModalOkButton>)>,
    mut eval: ResMut<EvalState>,
) {
    if interaction_q.iter().any(|i| *i == Interaction::Pressed) {
        eval.phase = EvalPhase::Idle;
    }
}

fn handle_modal_cancel_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<ModalCancelButton>)>,
    mut eval: ResMut<EvalState>,
) {
    if interaction_q.iter().any(|i| *i == Interaction::Pressed) {
        eval.phase = EvalPhase::Idle;
    }
}

fn handle_modal_evaluate_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<ModalEvaluateButton>)>,
    mut eval: ResMut<EvalState>,
    state: Res<GraphState>,
    input_q: Query<(&ModalSourceInput, &TextInput)>,
) {
    for interaction in interaction_q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let graph = state.root_graph().flattened_graph();
        let mut user_source_values: std::collections::HashMap<model::node::Id, eval::EValue> =
            std::collections::HashMap::new();
        let mut parse_errors: Vec<String> = Vec::new();
        for (m, input) in input_q.iter() {
            if let Some(model::node::ENode::Source { r#type, name, .. }) =
                graph.nodes.get(&m.node_id)
            {
                // A Source that declares nothing has nothing to parse the
                // answer as. Like a cast with no target, that is a half-built
                // node and so an error of the graph, not a `none` travelling
                // along an edge.
                let Some(r#type) = r#type else {
                    parse_errors.push(format!("{}: no type declared", name));
                    continue;
                };
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
            eval.phase = EvalPhase::Running {
                states: vec![eval::State::nothing_yet()],
                current: 0,
                user_source_values,
            };
        }
    }
}

/// End the screenshot mode at the first sign of life.
///
/// The input is not swallowed: it does whatever it normally does, and the mode
/// ending is a side effect of the editor being used again rather than a command
/// of its own. That is also why no system ordering is needed — nothing here
/// races anything, and a `MessageReader` carries its own cursor, so the camera
/// draining motion for itself does not drain it for us.
///
/// The readers are drained whatever the mode, because a message lives two
/// frames: left standing, the movement that reached for the button would be
/// waiting the moment the mode began, and end it before the grace could.
fn end_screenshot_mode(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    mut screenshot: ResMut<ScreenshotMode>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    // `fold` and `count`, not `any`/`next`: both have to walk the whole queue.
    // A short-circuiting read leaves the rest of it standing for next frame,
    // and next frame it would end a mode that had only just begun.
    let moved = motion
        .read()
        .fold(false, |seen, m| seen || m.delta != Vec2::ZERO);
    let scrolled = wheel.read().count() > 0;
    if !screenshot.listening(time.elapsed_secs()) {
        return;
    }
    let touched = moved
        || scrolled
        || keys.get_just_pressed().next().is_some()
        || mouse.get_just_pressed().next().is_some();
    if touched {
        screenshot.entered_at = None;
        rebuild.0 = true;
    }
}

/// Fill the control row with what the current phase has to offer.
///
/// Idle is `▶ ▶▌` — begin, or begin and go all the way. Running is `◀ ■ ▶ ▶▌`,
/// and `■` stands exactly where `▶` did: the same place answers "begin" and
/// "stop", so the row grows outward from a fixed point rather than shuffling
/// under the pointer. `▶▌` keeps its own place at the end throughout, because
/// it means the same thing in both.
///
/// The shapes are the ones the bundled font has, which is why they are not the
/// media-control block: it carries neither `⏭` (U+23ED) nor `⏩`, so `▶▌` is
/// U+25B6 with U+258C — a left half block, which fills the left half of its
/// cell and so sits against the triangle. Likewise `■` (U+25A0) rather than
/// `⏹`. No pause at all, and none is wanted: a run already sits on its step
/// until told to take another.
///
/// Latched on the phase, so the row is rebuilt only when it changes. The latch
/// is an `Option` rather than a `bool` so the first run fills the idle row too;
/// as a bare `bool` it would start out agreeing with "not running" and put
/// nothing on screen until the first evaluation.
fn sync_player_controls(
    mut commands: Commands,
    eval: Res<EvalState>,
    ui_font: Res<UiFont>,
    row_q: Query<Entity, With<PlayerControls>>,
    content_q: Query<Entity, With<PlayerControlsEntity>>,
    mut was_running: Local<Option<bool>>,
) {
    let running_now = matches!(eval.phase, EvalPhase::Running { .. });
    if *was_running == Some(running_now) {
        return;
    }
    *was_running = Some(running_now);
    for e in content_q.iter() {
        commands.entity(e).despawn();
    }
    let Ok(row) = row_q.single() else {
        return;
    };
    commands.entity(row).with_children(|parent| {
        if !running_now {
            spawn_control_button(parent, &ui_font.0, "\u{25B6}", EvaluateButton);
            spawn_control_button(parent, &ui_font.0, "\u{25B6}\u{258C}", FullRunButton);
            return;
        }
        // Ahead of the buttons, so it reads to the *left* of them. The row is
        // right-aligned, and a counter on the right would shove every button
        // sideways the moment "Step 9" became "Step 10". Here it grows into the
        // empty half of the box and nothing else moves — which is the whole
        // point of the row being pinned to that edge.
        //
        // Spawned empty: `update_step_button_visuals` writes it every frame, and
        // it is the only thing that knows which step the row has moved to since.
        parent.spawn((
            Text::new(""),
            text_font(&ui_font.0, 14.0),
            TextColor(Color::srgb(0.6, 0.6, 0.7)),
            StepCounterText,
            PlayerControlsEntity,
        ));
        // Prev before Exit before Next: the two arrows sit either side of the
        // step they move, and what stops the run is between them, where the
        // thing that started it stood.
        spawn_control_button(parent, &ui_font.0, "\u{25C0}", PrevStepButton);
        spawn_control_button(parent, &ui_font.0, "\u{25A0}", ExitEvaluationButton);
        spawn_control_button(parent, &ui_font.0, "\u{25B6}", NextStepButton);
        spawn_control_button(parent, &ui_font.0, "\u{25B6}\u{258C}", FullRunButton);
    });
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

/// Say that the run should reach its end, and start one if none is going.
///
/// Only ever says it — `apply_run_to_end` is what does it. That split is what
/// carries the wish across the values modal: pressed on an idle graph with
/// Sources, this opens the modal and the run begins a few presses later, with
/// nothing in between that could have remembered what was asked for.
///
/// No hover tint, like `Prev` and `Next` beside it: these three say whether they
/// can be pressed, not whether they are being looked at.
fn handle_full_run_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<FullRunButton>)>,
    mut eval: ResMut<EvalState>,
    state: Res<GraphState>,
    diagnostics: Res<Diagnostics>,
    mut open: ResMut<DiagnosticsOpen>,
) {
    if !interaction_q.iter().any(|i| *i == Interaction::Pressed) {
        return;
    }
    if matches!(eval.phase, EvalPhase::Idle) {
        begin_evaluation(&mut eval, &state, &diagnostics, &mut open);
        // A graph that will not run must not be left with a standing wish to
        // run to its end: the wish would be granted the moment the last hole
        // was filled, by a press nobody made.
        if diagnostics.blocking() {
            return;
        }
    }
    eval.run_to_end = true;
}

/// Take the run to its end, one step at a time, in a single frame.
///
/// Exactly what pressing `Next` until it goes quiet would do, and for the same
/// reason: every snapshot is kept, so `Step N` lands on the last number and
/// `Prev` walks back through the run rather than jumping over it. The button
/// spares presses, not the history.
///
/// Three ways to stop, and all three are needed:
///
/// - the sink has a value, which is what being evaluated *is*;
/// - the step failed, which ends the run the way a failed `Next` does;
/// - the step resolved nothing new. This is the one that matters: a node whose
///   inputs never all arrive leaves `eval_next_step` returning the state
///   unchanged, for ever. `Next` already tests for it — a press that resolves
///   nothing does not grow the history — and here it is the difference between
///   stopping and hanging.
///
/// The cap behind them is a real bound rather than a guess: a productive round
/// resolves at least one node, so there cannot be more rounds than there are
/// nodes. It is there in case a fourth way to stand still is ever invented.
///
/// The graph is flattened once. A step never changes it, and flattening deep
/// clones every sub-layout — doing it per round would cost more than the steps.
fn apply_run_to_end(mut eval: ResMut<EvalState>, state: Res<GraphState>) {
    if !eval.run_to_end {
        return;
    }
    // Still on its way to a run — the values modal is open, and the wish waits
    // for it rather than being spent on a phase that is not one.
    if matches!(eval.phase, EvalPhase::SourcePrompt { .. }) {
        return;
    }
    eval.run_to_end = false;
    if !matches!(eval.phase, EvalPhase::Running { .. }) {
        // Cancelled, or refused before it began. Nothing to run.
        return;
    }
    let graph = state.root_graph().flattened_graph();
    let Some(sink_node) = graph.nodes.get(&graph.sink_node_id).cloned() else {
        return;
    };
    let cap = graph.nodes.len() + 1;
    let mut rounds = 0usize;
    loop {
        if rounds >= cap {
            warn!("run to end: gave up after {} rounds without finishing", cap);
            break;
        }
        rounds += 1;
        // Computed in a block of its own so the read of `eval.phase` is over
        // before the write below — the same two-step the `Next` press takes,
        // and for the same reason.
        let stepped = {
            let EvalPhase::Running {
                states,
                current,
                user_source_values,
            } = &eval.phase
            else {
                break;
            };
            if states[*current].is_evaluated(&state.root_graph().graph) {
                break;
            }
            states[*current].eval_next_step(
                &graph,
                user_source_values,
                (graph.sink_node_id.clone(), sink_node.clone()),
                &state.function_declarations,
            )
        };
        let next_state = match stepped {
            Ok(next_state) => next_state,
            Err(errors) => {
                eval.phase = EvalPhase::ErrorModal(errors.join("\n"));
                break;
            }
        };
        let EvalPhase::Running {
            states, current, ..
        } = &mut eval.phase
        else {
            break;
        };
        // Nothing new resolved: the run is standing still and no further press
        // would move it either.
        if next_state.node_ids_to_values.len() <= states[*current].node_ids_to_values.len() {
            break;
        }
        states.truncate(*current + 1);
        states.push(next_state);
        *current += 1;
    }
}

fn update_step_button_visuals(
    eval: Res<EvalState>,
    state: Res<GraphState>,
    diagnostics: Res<Diagnostics>,
    // One query over the whole row rather than one per button with `Without`
    // filters of each other, which grow as the square of them. Which button an
    // entity is, it says itself; the two that are simply pressable whenever they
    // are on screen say nothing and fall through.
    mut row_q: Query<
        (
            &Interaction,
            &mut BackgroundColor,
            &Children,
            Has<PrevStepButton>,
            Has<NextStepButton>,
            Has<FullRunButton>,
            Has<EvaluateButton>,
        ),
        Or<(
            With<EvaluateButton>,
            With<PrevStepButton>,
            With<NextStepButton>,
            With<FullRunButton>,
            With<ExitEvaluationButton>,
        )>,
    >,
    mut text_color_q: Query<&mut TextColor>,
    mut counter_q: Query<&mut Text, With<StepCounterText>>,
) {
    // Running to the end is the one of the three that can be asked for with no
    // run going: it starts one. The other two only move within a run.
    // An error standing in the list above these buttons is the one reason a run
    // cannot start, and now that the list is directly over them it is worth
    // saying in the buttons too. `begin_evaluation` refuses either way — this
    // only makes the refusal visible before it is met.
    //
    // It bears only on *starting*: inside a run the graph cannot change, so a
    // diagnostic cannot appear mid-run to grey out the step that is under way.
    let startable = !diagnostics.blocking();
    let (prev_enabled, next_enabled, full_enabled, start_enabled) = match &eval.phase {
        EvalPhase::Running {
            states, current, ..
        } => {
            let next_possible = !states[*current].is_evaluated(&state.root_graph().graph);
            (*current > 0, next_possible, next_possible, startable)
        }
        EvalPhase::Idle => (false, false, startable, startable),
        // A modal owns the screen and the row is hidden behind it anyway.
        _ => (false, false, false, false),
    };
    // The index itself: snapshot 0 is a run at rest
    // (`eval::State::nothing_yet`), with the prompt's answers waiting at the
    // Sources and nothing read yet, so step 0 is where a run honestly starts.
    //
    // A number and no total. The history grows one press at a time and its
    // length is how far the run has been stepped, not how long it is — a
    // denominator here would move as the numerator did and mean nothing.
    let counted = match &eval.phase {
        EvalPhase::Running { current, .. } => format!("Step {}", current),
        _ => String::new(),
    };
    for mut text in counter_q.iter_mut() {
        if text.0 != counted {
            text.0 = counted.clone();
        }
    }
    // Three states and one writer for all of them. It used to be two — the
    // greying here, the hover tint in each button's own handler — which left
    // `Evaluate` resting a shade darker than the `Prev` beside it, for no
    // reason either of them stated.
    let apply = |enabled: bool,
                 interaction: &Interaction,
                 bg: &mut BackgroundColor,
                 text_color: &mut TextColor| {
        let (fill, ink) = match (enabled, interaction) {
            (false, _) => (CONTROL_OFF, INK_OFF),
            (true, Interaction::Hovered | Interaction::Pressed) => (CONTROL_HOT, INK_BRIGHT),
            (true, Interaction::None) => (CONTROL_REST, INK_DIM),
        };
        bg.0 = fill;
        text_color.0 = ink;
    };
    for (interaction, mut bg, children, is_prev, is_next, is_full, is_start) in row_q.iter_mut() {
        // `Exit` falls through: it is only ever on screen during a run, and a
        // run can always be left.
        let enabled = if is_prev {
            prev_enabled
        } else if is_next {
            next_enabled
        } else if is_full {
            full_enabled
        } else if is_start {
            start_enabled
        } else {
            true
        };
        if let Ok(mut c) = text_color_q.get_mut(children[0]) {
            apply(enabled, interaction, &mut *bg, &mut *c);
        }
    }
}

/// The yellow a value a run has produced is written in. Named because the label
/// is spawned in one place and regraded in another, and the two have to agree
/// on what colour they are fading.
const VALUE_LABEL_COLOR: Color = Color::srgb(1.0, 0.95, 0.3);

fn sync_value_labels(
    mut commands: Commands,
    eval: Res<EvalState>,
    state: Res<GraphState>,
    pick: Res<PickState>,
    clipping: Res<lod::Clipping>,
    ui_font: Res<UiFont>,
    mut existing_q: Query<(Entity, &ValueLabel, &mut Text, &mut TextColor)>,
) {
    let Some((states, current)) = (match &eval.phase {
        EvalPhase::Running {
            states, current, ..
        } => Some((states, *current)),
        _ => None,
    }) else {
        for (entity, _, _, _) in existing_q.iter() {
            commands.entity(entity).despawn();
        }
        return;
    };

    // These labels outlive a rebuild — they are reconciled by hand rather than
    // cleared with the scene — so the grading has to be asked here too, and
    // asked every frame: the caret moves, the picture around it regrades, and
    // nothing else would tell a standing label about it.
    let grading = lod::Lod::new(
        state.scope_of_caret(&pick).map(|scope| scope.path),
        clipping.0,
    );
    let opacity_of = |id: &model::node::Id| match state.root_graph().context_of_node(id) {
        Some(context) if grading.hidden(&context) => None,
        Some(context) => Some(grading.content(&context)),
        // A node no scope claims is not the grading's business.
        None => Some(1.0),
    };

    // What *this* step resolved, rather than everything the run knows by now.
    //
    // A snapshot is cumulative — `eval_next_step` folds the state it started
    // from into the one it hands back (`eval::State::merged_with`) — so the
    // whole history stands in every entry, and a label per entry says nothing
    // about the press that produced it. The difference to the snapshot behind
    // does: it is exactly the frontier that press resolved. Snapshot 0 has
    // nothing behind it, and there the difference is the whole of it — which
    // is nothing at all, step 0 having produced nothing. What it shows instead
    // stands on the Sources' own front cells (`infer::Known::offered_at`), and
    // that is the right place for it: those values were not produced here, they
    // were handed in.
    //
    // The values a step is no longer the news of are not lost with their
    // label: they are written on the anchors themselves, as the literals the
    // run narrowed them to.
    let resolved_now: std::collections::HashMap<&model::node::Id, &eval::EValue> = match current {
        0 => states[0].node_ids_to_values.iter().collect(),
        n => {
            let before = &states[n - 1].node_ids_to_values;
            states[n]
                .node_ids_to_values
                .iter()
                .filter(|(node_id, _)| !before.contains_key(*node_id))
                .collect()
        }
    };

    let mut kept: std::collections::HashSet<model::node::Id> = std::collections::HashSet::new();
    for (entity, label, mut text, mut color) in existing_q.iter_mut() {
        let Some(opacity) = opacity_of(&label.node_id) else {
            // Inside a volume the grading has closed over or taken to nothing.
            // A label is screen chrome and would otherwise hang in front of a
            // box that is supposed to be shut.
            commands.entity(entity).despawn();
            continue;
        };
        if let Some(value) = resolved_now.get(&label.node_id) {
            let rendered = value.to_string();
            if text.0 != rendered {
                text.0 = rendered;
            }
            let wanted = lod::faded_color(VALUE_LABEL_COLOR, opacity);
            if color.0 != wanted {
                color.0 = wanted;
            }
            kept.insert(label.node_id.clone());
        } else {
            commands.entity(entity).despawn();
        }
    }

    // Where the new labels go, asked of every scope rather than of the root's
    // own map.
    //
    // The step driver runs on `flattened_graph()` (see the `Next` handler),
    // which holds the nodes of every branch alongside the root's: a
    // `BranchSource`, a `Tunnel`, a branch's own Sink, anything built inside
    // an arm. Those live one `LayoutGraph` further in, under
    // `sub_layouts[pattern_id]`, and asking `root_graph().layout_nodes` for
    // them found nothing and said nothing about it. So a step that resolved a
    // node inside a branch grew the snapshot, advanced the history and drew
    // exactly nothing — several presses of `Next` that read as a dead button,
    // and then the answer appearing at the Sink as if from nowhere.
    //
    // `walk_all` is the walk `spawn_graph_nodes` already takes, and
    // `extra_offset` is what carries a branch-local position out into the
    // world — composed exactly as `render::layoutnode_to_rendernode` composes
    // it, so a value stands over the node it belongs to whichever scope that
    // node lives in.
    //
    // Taken only when there is something new to place. This runs every frame
    // and the walk allocates; with every label already standing there is
    // nothing for it to answer.
    let missing: Vec<(&model::node::Id, &eval::EValue)> = resolved_now
        .into_iter()
        .filter(|(id, _)| !kept.contains(*id))
        .collect();
    if missing.is_empty() {
        return;
    }
    // A node's own cell, except where that is not where it answers.
    //
    // A Match hands its value out behind its arms — `match_output_z` is how far
    // behind, the same call the renderer places the anchor with — and its own
    // cell is the entry in front of them. Written there, the answer to a large
    // Match appears at the far end of the run from the branch that produced it,
    // as if the value had jumped back over everything. So it is written where
    // it is handed out, which is also the next thing anything downstream reads.
    //
    // Every other kind keeps its cell. Their outputs sit a cell or two along a
    // body the eye crosses in one go, and moving those labels would buy nothing
    // but a second rule.
    let positions: std::collections::HashMap<model::node::Id, Vec3> = state
        .layout_graph
        .walk_all()
        .into_iter()
        .map(|walked| {
            let answers_at = match walked
                .layout_graph
                .graph
                .nodes
                .get(&walked.layout_node.node_id)
            {
                Some(model::node::ENode::Match { patterns, .. }) => Vec3::new(
                    0.0,
                    0.0,
                    walked.layout_graph.match_output_z(patterns) as f32,
                ),
                _ => Vec3::ZERO,
            };
            (
                walked.layout_node.node_id.clone(),
                render::cell_center_world(
                    walked.layout_node.pos + walked.extra_offset + answers_at,
                ),
            )
        })
        .collect();

    for (id, value) in missing {
        let Some(&world_pos) = positions.get(id) else {
            continue;
        };
        let Some(opacity) = opacity_of(id) else {
            continue;
        };
        commands.spawn((
            Text::new(value.to_string()),
            text_font(&ui_font.0, 28.0),
            TextColor(lod::faded_color(VALUE_LABEL_COLOR, opacity)),
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
    state: Res<GraphState>,
    eval: Res<EvalState>,
    ui_font: Res<UiFont>,
    pick: Res<PickState>,
    editor_mode: Res<EditorMode>,
    screenshot: Res<ScreenshotMode>,
    clipping: Res<lod::Clipping>,
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
            state,
            eval,
            ui_font,
            pick,
            editor_mode,
            screenshot,
            clipping,
        );
        rebuild.0 = false;
    }
}

/// Redraw when the run moves, because a step changes what the anchors say.
///
/// Watching rather than flagging. `Running` is entered from the Evaluate button
/// and from the source modal, left by `Exit Evaluation` and by any error a step
/// raises, and `current` moves under both `Prev` and `Next` — six places that
/// would each have to remember, against one that cannot forget. The pair it
/// compares is the whole of what the drawing reads out of `EvalState`.
fn sync_eval_rebuild(
    eval: Res<EvalState>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut last: Local<Option<(bool, usize)>>,
) {
    let now = match &eval.phase {
        EvalPhase::Running { current, .. } => (true, *current),
        _ => (false, 0),
    };
    if *last != Some(now) {
        *last = Some(now);
        rebuild.0 = true;
    }
}

#[derive(Component)]
pub struct WorldLabel {
    pub world_pos: Vec3,
    pub offset: Vec2, // screen-space pixel offset
}

/// What a scope's floor carries beyond the plain surface: the component that
/// makes it the mouse's pick target, the border rect marking the caret's own
/// scope, and the footprint rects that flatten multi-cell nodes.
///
/// One struct rather than four more parameters on the spawner, which is the
/// whole of why it exists.
struct InteractiveFloor {
    scope: ScopeGridEntity,
    border_min: Vec2,
    border_max: Vec2,
    footprints: [Vec4; grid::MAX_FOOTPRINTS],
    footprint_count: u32,
}

/// Spawn the four grid surfaces that frame one volume: the floor it stands on,
/// the back wall on its lesser-X side, and the two Z faces closing it front and
/// back.
///
/// Four for every volume there is — the program's scope and every branch of
/// every Match — so that a volume is recognisable as one wherever it sits. What
/// tells them apart is `fade`, which says how much of this volume survives at
/// the distance it stands from the one the caret is in.
///
/// `shell` is the other direction: the alpha of a closed grey body drawn around
/// the whole volume, filling in the deeper inside the caret's own the volume
/// sits. At `1.0` it is all there is — `spawn_graph_nodes` builds nothing
/// inside a sealed volume, so the sub-graph reads as the one node-sized box it
/// has become. `0.0` spawns none.
///
/// A volume *is* a scope, and that is why this takes an `InteractiveFloor`
/// rather than an optional one. A Match used to be drawn as a volume too,
/// without one, which put two nested rooms on screen for something that is one
/// node: it owns no `LayoutGraph`, nothing addresses a cell in it, and its arms
/// are the volumes. It gets what a TypeCast gets now — its anchors, its links,
/// and no room.
///
/// `min`/`max` are the volume's inclusive cell bounds in coordinates `offset`
/// carries to global, always `grid_bounds()`. Cells are corner-anchored, so
/// every far edge is `max + 1`.
///
/// `fog_origin` goes to all four surfaces unchanged: it is the caret's, not the
/// volume's, and a fade that re-centred per volume would say nothing about
/// where the work is.
#[allow(clippy::too_many_arguments)]
fn spawn_volume_surfaces(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials_grid: &mut Assets<grid::GridMaterial>,
    materials: &mut Assets<StandardMaterial>,
    min: IVec3,
    max: IVec3,
    offset: Vec3,
    fade: f32,
    shell: f32,
    fog_origin: Vec3,
    floor: InteractiveFloor,
) {
    let size_x = (max.x - min.x + 1) as f32 * render::LAYOUT_SCALE.x.abs();
    let size_y = (max.y - min.y + 1) as f32 * render::LAYOUT_SCALE.y.abs();
    let size_z = (max.z - min.z + 1) as f32 * render::LAYOUT_SCALE.z.abs();
    // The inclusive range spans [min, max+1], so its centre is (min+max+1)/2 on
    // every axis. Where a surface pins one of the three, it takes an edge
    // instead — and which edge is the whole of what that surface says.
    let centre = |x: f32, y: f32, z: f32| render::layout_to_world(Vec3::new(x, y, z) + offset);
    let mid_x = (min.x + max.x + 1) as f32 * 0.5;
    let mid_y = (min.y + max.y + 1) as f32 * 0.5;
    let mid_z = (min.z + max.z + 1) as f32 * 0.5;

    // ── The shell. Closed where the four surfaces below are open, because it is
    // saying the opposite thing: they frame a volume to be looked into, and this
    // one stands in for a volume that is not to be. Back faces are culled, so it
    // reads as a body rather than as a room seen from inside.
    //
    // Drawn first so what follows can simply leave: by the step where the shell
    // is whole the volume itself has nothing left, and the four surfaces would
    // be four blended planes inside a closed box.
    if shell > 0.0 {
        let shell_center = centre(mid_x, mid_y, mid_z);
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(size_x, size_y, size_z).mesh().build())),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: lod::SHELL_COLOR.with_alpha(shell).into(),
                unlit: true,
                // Opaque once it is whole: a sealed volume has to occlude, and a
                // blended fragment at alpha 1 still spends an OIT slot to say
                // the same thing.
                alpha_mode: if shell >= 1.0 {
                    AlphaMode::Opaque
                } else {
                    AlphaMode::Blend
                },
                ..default()
            })),
            Transform::from_translation(shell_center),
            SceneEntity,
        ));
    }

    // Nothing of the volume itself survives the grading. With no floor there is
    // also no way to click into it, which is the point: what is not drawn is not
    // edited either.
    if fade <= 0.0 {
        return;
    }

    // ── The floor: the lower bounding edge of the volume's last row, so what
    // stands in it stands *on* it rather than hanging under it.
    let floor_center = centre(mid_x, (max.y + 1) as f32, mid_z);
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(size_x, size_z).build())),
        MeshMaterial3d(materials_grid.add(grid::GridMaterial {
            border_min: floor.border_min,
            border_max: floor.border_max,
            footprint_count: floor.footprint_count,
            footprints: floor.footprints,
            ..grid::GridMaterial::scope_surface(Vec3::X, Vec3::Z, fade, fog_origin)
        })),
        Transform::from_translation(floor_center),
        floor.scope,
        SceneEntity,
    ));

    // ── The back wall, on the lesser-X side. The bound camera's depth axis is
    // world X, so this is the surface the volume is seen *against* — the one
    // that says where it ends behind everything in it.
    let wall_center = centre(min.x as f32, mid_y, mid_z);
    commands.spawn((
        Mesh3d(
            meshes.add(
                // `Plane3d::new` takes half sizes and rotates a Y-up plane onto
                // the normal, so which world axis each one lands on depends on
                // that normal: for +X the first goes to Y and the second to Z,
                // where the Z faces' first goes to X and second to Y.
                Plane3d::new(Vec3::X, Vec2::new(size_y * 0.5, size_z * 0.5))
                    .mesh()
                    .build(),
            ),
        ),
        // `u` runs the way the volume does, `v` is its rows — the same reading
        // the Z faces' `(X, Y)` gives.
        MeshMaterial3d(materials_grid.add(grid::GridMaterial::scope_surface(
            Vec3::Z,
            Vec3::Y,
            fade,
            fog_origin,
        ))),
        Transform::from_xyz(wall_center.x, wall_center.y, wall_center.z),
        SceneEntity,
    ));

    // ── The two Z faces. The front one is where the volume opens — the source
    // row's face, looking toward the origin — and the back one the far side of
    // its last cell. Both span its X and Y, so the four surfaces together frame
    // the volume without closing it off.
    let face_center = centre(mid_x, mid_y, 0.0);
    for z_cell in [min.z, max.z + 1] {
        let face_z = centre(0.0, 0.0, z_cell as f32).z;
        commands.spawn((
            Mesh3d(
                meshes.add(
                    Plane3d::new(Vec3::Z, Vec2::new(size_x * 0.5, size_y * 0.5))
                        .mesh()
                        .build(),
                ),
            ),
            MeshMaterial3d(materials_grid.add(grid::GridMaterial::scope_surface(
                Vec3::X,
                Vec3::Y,
                fade,
                fog_origin,
            ))),
            Transform::from_xyz(face_center.x, face_center.y, face_z),
            SceneEntity,
        ));
    }
}

/// Spawn one structural link: a ribbon from `from` to `to` whose two ends may
/// wear different shapes — and, since a strand may be a conversion rather
/// than a carriage, different colours.
///
/// Two leaves and not one. Almost every link has the same type at both ends
/// and passes the same leaf twice; a cast is where they part, because what
/// leaves an input row is not what arrives at its target. Both are looked up
/// through `render::strand_color`, which is also what paints the flat anchor
/// segments the ribbon meets, so each seam falls on an exact match.
///
///
/// No `Edge` component and no parent entity. `Edge` holds an `Entity` for each
/// end, and a Pattern's band is no anchor — there is nothing to point at. A
/// link is not something the user wired, so nothing needs to pick it, follow it
/// or take it apart.
#[allow(clippy::too_many_arguments)]
fn spawn_link_ribbon(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials_edge: &mut Assets<edge::EdgeMaterial>,
    start_leaf: &infer::EType,
    end_leaf: &infer::EType,
    from: Vec3,
    to: Vec3,
    start: edge::RibbonEnd,
    end: edge::RibbonEnd,
    opacity: f32,
) {
    // A leaf that claims no row of its own — a sum type, or `Pending` — has no
    // strand to draw, and a strand has to know what it is at *both* ends
    // before it can be coloured at either.
    if edge::leaf_kind_of(start_leaf).is_none() || edge::leaf_kind_of(end_leaf).is_none() {
        return;
    }
    let curve = edge::EdgeCurve::from_endpoints(from, to);
    let (mesh, arc_total) = edge::build_tapered_ribbon_mesh(&curve, &start, &end);
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials_edge.add(edge::EdgeMaterial {
            band_color_start: render::strand_color(start_leaf).to_linear(),
            band_color_end: render::strand_color(end_leaf).to_linear(),
            time: 0.0,
            line_mode_start: start.line_mode,
            line_half_thickness: edge::RIBBON_LINE_HALF_THICKNESS_UV,
            line_mode_end: end.line_mode,
            height_start: start.height(),
            height_end: end.height(),
            arc_total,
            dash_period: 0.0,
            dash_duty: 0.0,
            opacity,
        })),
        SceneEntity,
    ));
}

/// Spawn one pending ribbon: the connection is there, the type that would
/// colour it is not.
///
/// The twin of `spawn_link_ribbon`, and deliberately without its `leaf` — there
/// is none to ask. What it fixes instead is the one appearance every undecided
/// strand in the picture wears: the neutral grey of a pending anchor body, cut
/// across by gaps. One function for all of them, so an edge and a link cannot
/// end up saying "not decided yet" in two different ways.
///
/// `parent` is what a graph edge hangs its strands off — its `Edge` root. A
/// link passes `None`, for the reason `spawn_link_ribbon` gives just above.
#[allow(clippy::too_many_arguments)]
fn spawn_pending_ribbon(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials_edge: &mut Assets<edge::EdgeMaterial>,
    from: Vec3,
    to: Vec3,
    start: edge::RibbonEnd,
    end: edge::RibbonEnd,
    parent: Option<Entity>,
    opacity: f32,
) {
    let curve = edge::EdgeCurve::from_endpoints(from, to);
    let (mesh, arc_total) = edge::build_tapered_ribbon_mesh(&curve, &start, &end);
    // Through the same call the coloured strands go through, of the same type
    // the grey anchor bodies are drawn from, so a band and the two cells it
    // joins cannot come apart. Bound once and used at both ends: an undecided
    // strand is undecided along its whole length, and two greys read from two
    // places could drift.
    let grey = render::strand_color(&infer::EType::Pending).to_linear();
    let mut spawned = commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials_edge.add(edge::EdgeMaterial {
            band_color_start: grey,
            band_color_end: grey,
            time: 0.0,
            line_mode_start: start.line_mode,
            line_half_thickness: edge::RIBBON_LINE_HALF_THICKNESS_UV,
            line_mode_end: end.line_mode,
            height_start: start.height(),
            height_end: end.height(),
            arc_total,
            dash_period: edge::RIBBON_DASH_PERIOD,
            dash_duty: edge::RIBBON_DASH_DUTY,
            opacity,
        })),
        SceneEntity,
    ));
    if let Some(parent) = parent {
        spawned.insert(ChildOf(parent));
    }
}

/// A ribbon end that takes a whole leaf row as a band: what a strand wears
/// where nothing narrower has been said about it.
///
/// The row an undecided anchor offers is its own — `plain_anchor_body` draws
/// exactly one, a full `STRAND_BAND_HEIGHT` tall — so a pending band meets it
/// by claiming all of it.
fn whole_row_end(row_center_y: f32, as_line: bool) -> edge::RibbonEnd {
    edge::ribbon_end(row_center_y, &infer::RowSpan::FULL, as_line)
}

/// Fan a Match input's rows out onto one arm's declared-type cell: every row
/// that arm can describe gets a strand, claiming the share of the row it takes.
///
/// The Match's alone. A TypeCast used to borrow it, on the grounds that an arm
/// and a cast target make the same statement with their cell — they do not. An
/// arm **selects**, and a selection has one destination per row: the arm, or
/// nothing. A cast **converts**, and a conversion has two, the target and
/// `none`, which between them are a partition of the row. One destination
/// cannot describe two, so the cast has its own pass (`spawn_cast_links`) and
/// this one is free to say the simple thing again.
///
/// The two ways a row can end up with no coloured strand are not the same thing
/// and are not drawn the same way.
///
/// A row the arm **can never describe** gets nothing. Note the deliberate
/// divergence from the edge pass, which aims an unmatched leaf at row 0:
/// docking a `Bool` arm onto an `Integer` band would draw the lie that it
/// consumes it. Here the gap *is* the statement — the Match reads as
/// non-exhaustive.
///
/// A cell that has **not said yet** what it describes — `declared` is `None`,
/// an arm the user has not typed — makes no such statement, and neither does a
/// row that is not there because what arrives is still `Pending`. Those get a
/// pending band: the cell is wired to the anchor either way, and only what
/// travels between them is open.
#[allow(clippy::too_many_arguments)]
fn spawn_declared_cell_links(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials_edge: &mut Assets<edge::EdgeMaterial>,
    in_pos: Vec3,
    in_rows: &[(usize, infer::EType)],
    in_value: Option<&str>,
    band_pos: Vec3,
    declared: Option<&model::r#type::EType>,
    opacity: f32,
) {
    let declared_leaf = declared.map(infer::graph_type_to_eval_type);
    let declared_value = declared.and_then(layout::value_of_etype);
    // Leave by the band's far face. The cell centre is the outward face, where
    // the *incoming* edge already ends.
    let from = render::anchor_body_face_world(in_pos, true);
    // The cell fills its own row whatever it declares — or fails to declare —
    // so the taper happens entirely at the anchor end.
    let cell_is_line = declared_leaf
        .as_ref()
        .is_some_and(|leaf| render::leaf_is_drawn_as_line(leaf, declared_value.as_deref()));

    // Every row the cell can take something from, not just one: an `Integer`
    // arm against a `1|2` anchor consumes both rows, and has to be seen doing
    // it or the match would read as missing an arm.
    //
    // The rows carry their own indices (`render::drawn_rows`), because a run
    // may have dropped the ones it did not take and the survivors keep the
    // places the layout gave them.
    //
    // An empty list still draws one row: an anchor whose type is undecided has
    // no leaf rows, but it still has the one its grey `plain_anchor_body` is
    // drawn on, and that is where its band leaves from.
    let rows: Vec<(usize, Option<&infer::EType>)> = if in_rows.is_empty() {
        vec![(0, None)]
    } else {
        in_rows
            .iter()
            .map(|(row, leaf)| (*row, Some(leaf)))
            .collect()
    };
    for (row, anchor_leaf) in rows {
        let row_y = in_pos.y + render::leaf_row_offset(row);
        let row_is_line =
            anchor_leaf.is_some_and(|leaf| render::leaf_is_drawn_as_line(leaf, in_value));

        // Both sides have to have spoken before the strand can carry a type:
        // what travels here is what the cell consumes, and it takes an arriving
        // type *and* a declared one to say how much of it that is.
        let (Some(anchor_leaf), Some(declared_leaf)) = (anchor_leaf, declared_leaf.as_ref()) else {
            spawn_pending_ribbon(
                commands,
                meshes,
                materials_edge,
                from,
                // A declared-type cell is drawn input-side, so its near face is
                // the cell centre — which is the face this meets.
                band_pos,
                whole_row_end(row_y, row_is_line),
                whole_row_end(band_pos.y, cell_is_line),
                None,
                opacity,
            );
            continue;
        };
        let Some(span) = infer::claimed_span(anchor_leaf, declared_leaf) else {
            continue;
        };
        spawn_link_ribbon(
            commands,
            meshes,
            materials_edge,
            anchor_leaf,
            anchor_leaf,
            from,
            band_pos,
            edge::ribbon_end(row_y, &span, row_is_line),
            whole_row_end(band_pos.y, cell_is_line),
            opacity,
        );
    }
}

/// Draw the connections a Match is made *of* rather than wired from: what
/// arrives at its input reaching each arm's band, and what each branch produces
/// reaching its output.
///
/// These are not edges. No `Edge` ever crosses a branch's volume boundary — a
/// branch reads the matched value from its own `BranchSource` instead — so the
/// edge table says nothing about them and they have to be drawn from the
/// structure itself.
///
/// What they show is a **consumption**. Each arm claims a slice of the band its
/// scrutinee arrives on (`infer::RowSpan`): a whole band for a base type, half
/// of one for `true` or `false`, and nothing at all — a line at the band's top
/// edge — for a literal of a type with infinitely many values. Claimed exactly
/// once over the anchor's whole height, the Match is exhaustive. A gap is a
/// missing arm and an overlap a redundant one, and both are left standing where
/// they can be seen rather than reported: the picture is the report, and the
/// linter that will read these same spans comes after it.
///
/// A plain `fn` and not a system: the two maps it needs are locals of
/// `spawn_graph_nodes`, and making them a resource would buy nothing but an
/// ordering constraint.
#[allow(clippy::too_many_arguments)]
fn spawn_match_links(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials_edge: &mut Assets<edge::EdgeMaterial>,
    state: &GraphState,
    flat_graph: &model::term_graph::TermGraph,
    known: &infer::Known,
    anchor_world_positions: &std::collections::HashMap<model::anchor::Id, Vec3>,
    declared_band_positions: &std::collections::HashMap<model::node::Id, Vec3>,
    grading: &lod::Lod,
) {
    let decls = &state.function_declarations;
    // `walk_all` reaches every scope with its offset already composed, and a
    // Match's Patterns always live in the same LayoutGraph the Match does
    // (`plus_match`), so a nested Match needs nothing path-aware here — it is
    // reached like any other and resolves its arms by lookup.
    for walked in state.layout_graph.walk_all() {
        // Every link a Match owns is graded with the Match, arms included: they
        // are the one node's own making. A leg reaching into an arm whose scope
        // is not drawn ends at an anchor that was never placed, and falls away
        // on the lookup below.
        let scope = scope_of_walk(state, &walked.context);
        if grading.hidden(scope) {
            continue;
        }
        let opacity = grading.content(scope);
        let Some(model::node::ENode::Match {
            patterns,
            input_anchor,
            output_anchor,
        }) = walked
            .layout_graph
            .graph
            .nodes
            .get(&walked.layout_node.node_id)
        else {
            continue;
        };

        // ── What arrives, reaching the arms ──
        //
        // Read through the same calls the Match's own anchor is drawn from, so
        // the rows a strand aims at are the rows that are actually there — the
        // run's narrowing included, or a strand would leave a row that is no
        // longer drawn.
        let in_rows = infer::incoming_anchor_type(flat_graph, input_anchor, decls)
            .map(|t| render::drawn_rows(&t, known.at_input(flat_graph, input_anchor)))
            .unwrap_or_default();
        let in_value = infer::incoming_anchor_literal(flat_graph, input_anchor, known);
        if let Some(&in_pos) = anchor_world_positions.get(input_anchor) {
            for pattern_id in patterns {
                // An arm that declares nothing is still an arm: what reaches it
                // is undecided, not absent, and `spawn_declared_cell_links`
                // draws that as a pending band.
                let Some(model::node::ENode::Pattern {
                    r#type: arm_type, ..
                }) = flat_graph.nodes.get(pattern_id)
                else {
                    continue;
                };
                let Some(&band_pos) = declared_band_positions.get(pattern_id) else {
                    continue;
                };
                spawn_declared_cell_links(
                    commands,
                    meshes,
                    materials_edge,
                    in_pos,
                    &in_rows,
                    in_value.as_deref(),
                    band_pos,
                    arm_type.as_ref(),
                    opacity,
                );
            }
        }

        // ── What each branch produces, reaching the output ──
        //
        // The mirror image: here the branch's Sink is the declared, narrow side
        // and the Match's output is the band. Several sinks landing on one
        // output row is what sum-type normalisation looks like.
        //
        // Empty while the Match output is `Pending`, which one undecided branch
        // is enough to cause (`infer::match_output_type`). That decides which
        // *row* a strand lands on and nothing else: a branch whose own type is
        // settled still draws its strand, aimed at the anchor's own row, rather
        // than being blanked by a sibling it has nothing to do with.
        let out_rows = render::drawn_rows(
            &infer::anchor_type(flat_graph, output_anchor, decls).unwrap_or(infer::EType::Pending),
            known.at_output(flat_graph, output_anchor),
        );
        let Some(&out_pos) = anchor_world_positions.get(output_anchor) else {
            continue;
        };
        for pattern_id in patterns {
            // The whole branch resolves through the flattened graph: a Pattern
            // names its Sink, a Sink names its anchor, and every branch node is
            // in `flat_graph`. No layout traversal is needed for any of it.
            let Some(model::node::ENode::Pattern { sink_node_id, .. }) =
                flat_graph.nodes.get(pattern_id)
            else {
                continue;
            };
            let Some(model::node::ENode::Sink {
                input_anchor: sink_input,
            }) = flat_graph.nodes.get(sink_node_id)
            else {
                continue;
            };
            let Some(&sink_pos) = anchor_world_positions.get(sink_input) else {
                continue;
            };
            let sink_value = infer::incoming_anchor_literal(flat_graph, sink_input, known);
            let sink_rows = infer::incoming_anchor_type(flat_graph, sink_input, decls)
                .map(|t| render::drawn_rows(&t, known.at_input(flat_graph, sink_input)))
                .unwrap_or_default();
            let from = render::anchor_body_face_world(sink_pos, true);
            // Arrive at the output band's near face, in front of it — aiming at
            // the cell centre would run the strand through the whole band
            // before stopping at its back.
            let to = render::anchor_body_face_world(out_pos, false);

            // What travels here is what the branch produces, so the branch's
            // own Sink is what is asked. Nothing wired into it, or wired and
            // itself still `Pending`, and there is a connection to draw with no
            // type to draw it in.
            if sink_rows.is_empty() {
                spawn_pending_ribbon(
                    commands,
                    meshes,
                    materials_edge,
                    from,
                    to,
                    whole_row_end(sink_pos.y, false),
                    whole_row_end(out_pos.y, false),
                    None,
                    opacity,
                );
                continue;
            }

            for (k, leaf) in sink_rows.iter() {
                // The output has not decided what it is yet, so there is no row
                // to pick and the strand aims at the anchor's own — the same
                // fallback the edge pass uses for a leaf its target has no row
                // for. It carries the branch's own colour all the same: what
                // *this* branch produces is settled even while the union of all
                // of them is not.
                if out_rows.is_empty() {
                    let as_line = render::leaf_is_drawn_as_line(leaf, sink_value.as_deref());
                    spawn_link_ribbon(
                        commands,
                        meshes,
                        materials_edge,
                        leaf,
                        leaf,
                        from,
                        to,
                        whole_row_end(sink_pos.y + render::leaf_row_offset(*k), as_line),
                        // The same shape at both ends: there is nothing at the
                        // output end that could ask for a different one.
                        whole_row_end(out_pos.y, as_line),
                        opacity,
                    );
                    continue;
                }
                for (row, out_leaf) in out_rows.iter() {
                    // What this branch produces claims its share of the output
                    // row it lands on. Two branches yielding `true` and `false`
                    // cover the Bool band between them — which is why the
                    // output *is* `Bool` and not `true|false`. Two yielding `1`
                    // and `2` land on their own rows, because the output kept
                    // them apart rather than widening to `Integer`.
                    let Some(span) = infer::claimed_span(out_leaf, leaf) else {
                        continue;
                    };
                    let start = edge::ribbon_end(
                        sink_pos.y + render::leaf_row_offset(*k),
                        &span,
                        render::leaf_is_drawn_as_line(leaf, sink_value.as_deref()),
                    );
                    // The Match output is drawn from its type alone — deriving
                    // an extra literal for it would be constant folding — so it
                    // is a line only where its own type says so. Under a run
                    // the type *does* say so, the run having narrowed it to the
                    // value: proved rather than folded, which is the whole
                    // difference.
                    let end = edge::ribbon_end(
                        out_pos.y + render::leaf_row_offset(*row),
                        &span,
                        render::leaf_is_drawn_as_line(out_leaf, None),
                    );
                    spawn_link_ribbon(
                        commands,
                        meshes,
                        materials_edge,
                        leaf,
                        leaf,
                        from,
                        to,
                        start,
                        end,
                        opacity,
                    );
                }
            }
        }
    }
}

/// Draw the connections a TypeCast is made of: what arrives at its input
/// reaching its target cell, and what cannot get there reaching the `none` its
/// output carries.
///
/// These are not edges. No `Edge` ever runs between a cast's own cells — the
/// node *is* the connection — so the edge table says nothing about them and
/// they have to be drawn from the structure itself, the way a Match's are.
///
/// What they show is a **partition**. One input row leaves in two pieces and
/// the two pieces are the whole of it: the share that converts goes to the
/// target cell, the share that cannot goes past the cell to `none`. Past it,
/// not through it — a cast that fails never reaches its target. Which share is
/// which is `infer::cast_row_split`'s answer and nothing here re-decides it;
/// this places geometry and no more.
///
/// That both pieces come out of one call, in one pass over the row, is the
/// point and not an accident of style. They used to be worked out in two
/// places — the target leg through `spawn_declared_cell_links`, the `none` leg
/// here — and two places agreeing by convention is exactly how a `String`
/// cast to `Integer` came to be drawn twice at full height, once into the
/// target and once into `none`, as though the same values did both.
///
/// The third leg is the reverse one a Match has: target cell → output anchor,
/// the path taken when the cast succeeds. It is drawn when the output has a
/// `none` row **and** at least one input row actually arrived at the cell, and
/// both halves of that condition earn their place.
///
/// A **total** cast does without it because its target cell and its output say
/// the same thing one cell apart, and a strand between them would only spell
/// it twice. The moment a `none` row appears that stops being true: the output
/// is two rows now, no longer one statement repeated, and without this leg the
/// only strand arriving anywhere on it is the one reaching `none` — a cast
/// that could perfectly well succeed, drawn as a cast that always fails.
///
/// And a cast nothing can get through — `none` alone arriving at a cast to
/// `Integer` — has a `none` row all the same, so the first half would draw a
/// leg out of a cell that nothing ever reaches. `reaches_target` is set by the
/// loop that draws the strands rather than worked out a second time, so the
/// condition *is* the drawing and the two cannot drift apart.
///
/// A plain `fn` and not a system, for the reason `spawn_match_links` gives.
#[allow(clippy::too_many_arguments)]
fn spawn_cast_links(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials_edge: &mut Assets<edge::EdgeMaterial>,
    state: &GraphState,
    flat_graph: &model::term_graph::TermGraph,
    known: &infer::Known,
    anchor_world_positions: &std::collections::HashMap<model::anchor::Id, Vec3>,
    declared_band_positions: &std::collections::HashMap<model::node::Id, Vec3>,
    grading: &lod::Lod,
) {
    let decls = &state.function_declarations;
    for walked in state.layout_graph.walk_all() {
        // Graded with the cast, the same way a Match grades its own legs.
        let scope = scope_of_walk(state, &walked.context);
        if grading.hidden(scope) {
            continue;
        }
        let opacity = grading.content(scope);
        let node_id = &walked.layout_node.node_id;
        let Some(model::node::ENode::TypeCast {
            r#type,
            input_anchor,
            output_anchor,
        }) = walked.layout_graph.graph.nodes.get(node_id)
        else {
            continue;
        };
        let Some(&in_pos) = anchor_world_positions.get(input_anchor) else {
            continue;
        };
        let Some(&band_pos) = declared_band_positions.get(node_id) else {
            continue;
        };
        // Read through the same call the cast's own input anchor is drawn from,
        // so the rows a strand aims at are the rows that are actually there.
        let in_rows = infer::incoming_anchor_type(flat_graph, input_anchor, decls)
            .map(|t| render::drawn_rows(&t, known.at_input(flat_graph, input_anchor)))
            .unwrap_or_default();
        let in_value = infer::incoming_anchor_literal(flat_graph, input_anchor, known);

        let target_leaf = r#type.as_ref().map(infer::graph_type_to_eval_type);
        let target_value = r#type.as_ref().and_then(layout::value_of_etype);
        // The cell fills its own row whatever it declares — or fails to
        // declare — so the taper happens entirely at the anchor end.
        let cell_is_line = target_leaf
            .as_ref()
            .is_some_and(|leaf| render::leaf_is_drawn_as_line(leaf, target_value.as_deref()));

        // Held as options rather than behind a `continue`: a total cast has no
        // `none` row and still has strands to draw, and an output that has not
        // been placed must not take the target leg down with it.
        let out_pos = anchor_world_positions.get(output_anchor).copied();
        let out_rows = render::drawn_rows(
            &infer::anchor_type(flat_graph, output_anchor, decls).unwrap_or(infer::EType::Pending),
            known.at_output(flat_graph, output_anchor),
        );
        let out_value = infer::anchor_literal(flat_graph, output_anchor, known);
        // The row index the `none` still owns, not where it sits in the list: a
        // run that took the happy path leaves no `none` row at all, and then
        // the sad leg has nowhere to arrive and is not drawn.
        let none_row = out_rows
            .iter()
            .find(|(_, leaf)| matches!(leaf, infer::EType::None))
            .map(|(row, _)| *row);

        // Leave by the band's far face. The cell centre is the outward face,
        // where the *incoming* edge already ends.
        let from = render::anchor_body_face_world(in_pos, true);
        let mut reaches_target = false;

        // The rows carry their own indices, for the reason
        // `spawn_declared_cell_links` gives: a run drops the rows it did not
        // take and the survivors keep their places.
        //
        // An empty list still draws one row: an anchor whose type is undecided
        // has no leaf rows, but it still has the one its grey
        // `plain_anchor_body` is drawn on, and that is where its band leaves
        // from.
        let rows: Vec<(usize, Option<&infer::EType>)> = if in_rows.is_empty() {
            vec![(0, None)]
        } else {
            in_rows
                .iter()
                .map(|(row, leaf)| (*row, Some(leaf)))
                .collect()
        };
        for (row, anchor_leaf) in rows {
            let row_y = in_pos.y + render::leaf_row_offset(row);
            let row_is_line = anchor_leaf
                .is_some_and(|leaf| render::leaf_is_drawn_as_line(leaf, in_value.as_deref()));

            // Both sides have to have spoken before the row can be divided:
            // nothing arriving, or no target chosen, and there is a connection
            // to draw with nothing yet to say about what travels along it.
            let (Some(anchor_leaf), Some(target_leaf)) = (anchor_leaf, target_leaf.as_ref()) else {
                spawn_pending_ribbon(
                    commands,
                    meshes,
                    materials_edge,
                    from,
                    // A declared-type cell is drawn input-side, so its near
                    // face is the cell centre — which is the face this meets.
                    band_pos,
                    whole_row_end(row_y, row_is_line),
                    whole_row_end(band_pos.y, cell_is_line),
                    None,
                    opacity,
                );
                continue;
            };
            let (to_target, to_none) = infer::cast_row_split(anchor_leaf, target_leaf);

            // ── The share that converts, reaching the target cell ──
            if let Some(span) = to_target {
                reaches_target = true;
                spawn_link_ribbon(
                    commands,
                    meshes,
                    materials_edge,
                    anchor_leaf,
                    target_leaf,
                    from,
                    band_pos,
                    edge::ribbon_end(row_y, &span, row_is_line),
                    whole_row_end(band_pos.y, cell_is_line),
                    opacity,
                );
            }

            // ── The share that cannot, reaching the `none` ──
            //
            // There is no `none` row to reach when the cast is total, and then
            // there is no share to send there either — the two are the same
            // fact, stated once by `cast_row_split` and once by
            // `infer::type_cast_output_type`, which reads the same function.
            let (Some(span), Some(none_row), Some(out_pos)) = (to_none, none_row, out_pos) else {
                continue;
            };
            spawn_link_ribbon(
                commands,
                meshes,
                materials_edge,
                // It leaves as the value it was and arrives as the absence of
                // one. The strand used to be `none` along its whole length,
                // on the grounds that the colour should say where a strand
                // ends up; it can say both now, and what it shows is the
                // conversion rather than the destination claimed from the
                // start.
                anchor_leaf,
                &infer::EType::None,
                from,
                // Arrive at the output band's near face, in front of it.
                render::anchor_body_face_world(out_pos, false),
                edge::ribbon_end(row_y, &span, row_is_line),
                // `none` is always a line, so the strand tapers into one
                // however much of the band it left with.
                edge::ribbon_end(
                    out_pos.y + render::leaf_row_offset(none_row),
                    &infer::RowSpan::FULL,
                    true,
                ),
                opacity,
            );
        }

        // ── What the target does take, reaching the output ──
        if none_row.is_none() || !reaches_target {
            continue;
        }
        let Some(out_pos) = out_pos else {
            continue;
        };
        // `reaches_target` cannot be set without a target, so this holds
        // wherever the two guards above did — asked outright rather than
        // unwrapped, because a picture is not worth a panic.
        let Some(target_leaf) = target_leaf.as_ref() else {
            continue;
        };
        // Leave by the band's far face and arrive at the output's near one,
        // the way every leg between two cells is drawn here.
        let from = render::anchor_body_face_world(band_pos, true);
        let to = render::anchor_body_face_world(out_pos, false);
        for (row, out_leaf) in out_rows.iter() {
            // Every row but the sad one: what the target produces may itself
            // be more than a single leaf, and each of those rows is reached on
            // success.
            if matches!(out_leaf, infer::EType::None) {
                continue;
            }
            spawn_link_ribbon(
                commands,
                meshes,
                materials_edge,
                // The same type at both ends in practice, and passed twice
                // all the same: if the output ever grows a row the target
                // does not name, the leg should show it rather than assert
                // the target's colour over it.
                target_leaf,
                out_leaf,
                from,
                to,
                whole_row_end(band_pos.y, cell_is_line),
                whole_row_end(
                    out_pos.y + render::leaf_row_offset(*row),
                    render::leaf_is_drawn_as_line(out_leaf, out_value.as_deref()),
                ),
                opacity,
            );
        }
    }
}

/// Spawn a UI text label that tracks a world position.

/// Spawn one stack of strands: per leaf, the band and its type letter, or the
/// line and its value when that leaf carries a literal.
///
/// Shared by the per-anchor stacks and the node-level ones a Pattern uses, so a
/// Pattern's band is built exactly like an anchor's.
fn spawn_anchor_strands(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    font: &Handle<Font>,
    strands: Vec<render::RenderStrand>,
    opacity: f32,
) {
    for strand in strands {
        let render::RenderStrand {
            band,
            band_label,
            line,
            line_label,
        } = strand;
        // Exactly one of the two shapes stands per strand, but nothing here
        // needs to know which: each is spawned if it is there.
        for shape in [band, line].into_iter().flatten() {
            commands.spawn((
                Mesh3d(meshes.add(shape.mesh)),
                MeshMaterial3d(materials.add(lod::faded(shape.material, opacity))),
                shape.transform,
                SceneEntity,
            ));
        }
        for text in [band_label, line_label].into_iter().flatten() {
            spawn_world_label(commands, font, text, SceneEntity, opacity);
        }
    }
}
fn spawn_world_label(
    commands: &mut Commands,
    font: &Handle<Font>,
    render_label: render::RenderLabel,
    marker: impl Bundle,
    opacity: f32,
) -> Entity {
    commands
        .spawn((
            Text::new(render_label.text),
            text_font(font, render_label.font_size),
            TextColor(lod::faded_color(render_label.color, opacity)),
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
    screenshot: Res<ScreenshotMode>,
    mut text_q: Query<(&mut Text, &mut Node), With<FpsDisplay>>,
) {
    let Ok((mut text, mut node)) = text_q.single_mut() else {
        return;
    };
    // The one readout that stays lit behind the start menu — a frame rate is
    // true of the program whoever is looking at it. A screenshot is the one
    // case where it is not wanted, so this writes its own `display` rather
    // than wearing `EditorChrome` and answering to every reason at once.
    let desired = if screenshot.active() {
        Display::None
    } else {
        Display::Flex
    };
    if node.display != desired {
        node.display = desired;
    }
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

/// How many diagnostic rows are on screen at once.
///
/// The list is sorted worst-first, so a plain prefix window shows the rows
/// worth reading first — unlike the prompt's window, which has a highlight to
/// keep in view and so has to slide. What does not fit is *stated*, never
/// clipped, for the reason `spawn_prompt_rows` states it: a list that silently
/// ends reads as a list that finished.
const DIAGNOSTIC_ROWS: usize = 6;

/// Everything wrong with the graph as it stands, recomputed whenever it moves.
#[derive(Resource, Default)]
struct Diagnostics(Vec<lint::Diagnostic>);

impl Diagnostics {
    fn count(&self, severity: lint::Severity) -> usize {
        self.0.iter().filter(|d| d.severity == severity).count()
    }

    /// Whether a run may start. The definition of `Error` and nothing more.
    fn blocking(&self) -> bool {
        self.count(lint::Severity::Error) > 0
    }
}

/// Whether the list is unfolded. Not derived from whether there is anything to
/// show: a fold the editor opened and closed behind the user's back is a fold
/// that cannot be trusted. Pressing `▶` on a graph that will not run opens it —
/// that is the one place anything but a click writes it.
#[derive(Resource)]
struct DiagnosticsOpen(bool);

impl Default for DiagnosticsOpen {
    fn default() -> Self {
        Self(true)
    }
}

#[derive(Component)]
struct DiagnosticsPanel;

/// On every row and heading the panel rebuilds, so the sweep that clears them
/// cannot reach the other panels' rows. Same reason `EditorPanelEntity` exists.
#[derive(Component)]
struct DiagnosticsEntity;

#[derive(Component)]
struct DiagnosticsHeader;

/// The cell a row points at. Only rows that have one carry it, so a row about
/// the graph as a whole is not a button that goes nowhere.
#[derive(Component)]
struct DiagnosticTarget(IVec3);

#[derive(Default, PartialEq, Eq, Clone)]
struct DiagnosticsFingerprint {
    rows: Vec<(lint::Diagnostic, Option<IVec3>)>,
    open: bool,
}

/// The cell a node stands on, in the coordinates the caret is addressed in.
///
/// The same three steps the click-to-select path takes in `pick_nodes`: the
/// owner path, the node's position inside that scope, and the scope's own
/// offset. A node the layout does not hold has no cell, which is why this is an
/// `Option` rather than an assertion.
fn cell_of_node(state: &GraphState, id: &model::node::Id) -> Option<IVec3> {
    let root = state.root_graph();
    let context = root.context_of_node(id)?;
    let local = root
        .resolve_context(&context)
        .layout_nodes
        .get(id)?
        .pos
        .round()
        .as_ivec3();
    Some(local + root.scope_offset(&context))
}

/// Ask the graph what is wrong with it, once per change.
///
/// Keyed on the graph and not on the rebuild: what the linter reads is the
/// program, and a caret move rebuilds the scene without changing a thing about
/// it. `lod` is the other way round — it grades by where the caret stands — and
/// that is why the two are computed in different places.
fn recompute_diagnostics(state: Res<GraphState>, mut diagnostics: ResMut<Diagnostics>) {
    if !state.is_changed() {
        return;
    }
    diagnostics.0 = lint::check(state.root_graph(), &state.function_declarations);
}

/// The colour a severity is drawn in.
///
/// The error red is the one the modal already uses. The warning amber is new
/// and deliberately outside the type palette — a row that reads as a type would
/// be a row saying something about the value rather than about the graph.
fn severity_color(severity: lint::Severity) -> Color {
    match severity {
        lint::Severity::Error => Color::srgb(0.95, 0.30, 0.30),
        lint::Severity::Warning => Color::srgb(0.95, 0.70, 0.25),
        lint::Severity::Note => Color::srgb(0.55, 0.55, 0.65),
    }
}

/// The tallies the heading shows, worst first, leaving out what is not there.
///
/// Each carries its own severity so the heading can paint it: the glyph is what
/// says *which* count this is once the list is folded away and the rows that
/// would have explained it are gone. Without it a folded heading reads "2, 1".
fn diagnostics_tallies(diagnostics: &Diagnostics) -> Vec<(lint::Severity, String)> {
    [
        (lint::Severity::Error, "error"),
        (lint::Severity::Warning, "warning"),
        (lint::Severity::Note, "note"),
    ]
    .into_iter()
    .filter_map(|(severity, noun)| {
        let count = diagnostics.count(severity);
        (count > 0).then(|| {
            (
                severity,
                format!(
                    "{} {} {}{}",
                    severity.glyph(),
                    count,
                    noun,
                    if count == 1 { "" } else { "s" }
                ),
            )
        })
    })
    .collect()
}

fn sync_diagnostics_ui(
    mut commands: Commands,
    diagnostics: Res<Diagnostics>,
    open: Res<DiagnosticsOpen>,
    state: Res<GraphState>,
    ui_font: Res<UiFont>,
    panel_q: Query<Entity, With<DiagnosticsPanel>>,
    children_q: Query<Entity, With<DiagnosticsEntity>>,
    mut cache: Local<DiagnosticsFingerprint>,
) {
    let rows: Vec<(lint::Diagnostic, Option<IVec3>)> = diagnostics
        .0
        .iter()
        .map(|d| {
            let cell = d.node.as_ref().and_then(|id| cell_of_node(&state, id));
            (d.clone(), cell)
        })
        .collect();
    let fp = DiagnosticsFingerprint {
        rows: rows.clone(),
        open: open.0,
    };
    if *cache == fp {
        return;
    }
    *cache = fp;

    for e in children_q.iter() {
        commands.entity(e).despawn();
    }
    let Ok(panel_entity) = panel_q.single() else {
        return;
    };

    let font = &ui_font.0;
    let tallies = diagnostics_tallies(&diagnostics);
    let shown = if open.0 {
        rows.len().min(DIAGNOSTIC_ROWS)
    } else {
        0
    };
    commands.entity(panel_entity).with_children(|panel| {
        // The heading stands whether or not there is anything under it. A zone
        // that came and went would take the frame's proportions with it, and
        // "nothing is wrong" is worth saying once the reader has learnt to look
        // here for what is.
        let mut heading = panel.spawn((
            Text::new(if tallies.is_empty() {
                "No problems found".to_string()
            } else if open.0 {
                "▾ ".to_string()
            } else {
                "▸ ".to_string()
            }),
            text_font(font, 13.0),
            TextColor(if tallies.is_empty() {
                Color::srgb(0.45, 0.45, 0.55)
            } else {
                Color::srgb(0.75, 0.75, 0.9)
            }),
            Node {
                // The same 4 px of side padding the rows below carry, so the
                // lit heading and a lit row stand on the same two edges. It
                // had none while it was only text: a highlight is a shape, and
                // a shape that ended at the glyph would read as a misprint.
                padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                ..default()
            },
            BackgroundColor(FILL_NONE),
            DiagnosticsEntity,
        ));
        // Only foldable when there is something to fold. An empty heading that
        // answered to a click would offer to hide nothing — and so it is also
        // the one heading that stays dark under the pointer.
        if !tallies.is_empty() {
            heading.insert((
                Button,
                DiagnosticsHeader,
                HoverFill {
                    rest: FILL_NONE,
                    hot: ROW_HOT,
                },
            ));
        }
        // One span per tally, in its own colour, so the glyph and the count it
        // belongs to are one statement. A span carries its own font: it
        // inherits nothing from the root but its place in the line.
        heading.with_children(|line| {
            for (index, (severity, text)) in tallies.iter().enumerate() {
                if index > 0 {
                    line.spawn((
                        TextSpan::new("  "),
                        text_font(font, 13.0),
                        TextColor(Color::srgb(0.75, 0.75, 0.9)),
                    ));
                }
                line.spawn((
                    TextSpan::new(text.clone()),
                    text_font(font, 13.0),
                    TextColor(severity_color(*severity)),
                ));
            }
        });

        for (diagnostic, cell) in rows.iter().take(shown) {
            let mut row = panel.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(8.0),
                    padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    ..default()
                },
                BackgroundColor(FILL_NONE),
                DiagnosticsEntity,
            ));
            // Only a row with somewhere to go is a button, and only a button
            // lights up: a row that highlighted under the pointer and then did
            // nothing would be promising a jump it has no address for. No
            // `HoverInk` — the glyph carries its severity and the message its
            // own ink, and a painter that levelled the two would be saying
            // less than the row already says.
            if let Some(cell) = cell {
                row.insert((
                    Button,
                    DiagnosticTarget(*cell),
                    HoverFill {
                        rest: FILL_NONE,
                        hot: ROW_HOT,
                    },
                ));
            }
            row.with_children(|row| {
                row.spawn((
                    Text::new(match cell {
                        Some(cell) => format!(
                            "{} ({}, {}, {})",
                            diagnostic.severity.glyph(),
                            cell.x,
                            cell.y,
                            cell.z
                        ),
                        None => diagnostic.severity.glyph().to_string(),
                    }),
                    text_font(font, 13.0),
                    TextColor(severity_color(diagnostic.severity)),
                    Node {
                        width: Val::Px(96.0),
                        flex_shrink: 0.0,
                        ..default()
                    },
                ));
                // No marker on either: `despawn` takes descendants with it, so
                // the sweep only ever names what hangs directly off the panel.
                // Marking these too would have it despawn them a second time.
                row.spawn((
                    Text::new(diagnostic.message.clone()),
                    text_font(font, 13.0),
                    TextColor(Color::srgb(0.85, 0.85, 0.9)),
                ));
            });
        }

        if open.0 && rows.len() > shown {
            panel.spawn((
                Text::new(format!("… {} more", rows.len() - shown)),
                text_font(font, 12.0),
                TextColor(Color::srgb(0.35, 0.35, 0.4)),
                Node {
                    padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                    ..default()
                },
                DiagnosticsEntity,
            ));
        }
    });
}

/// The heading folds the list; a row goes to the cell it is about.
///
/// The camera is not moved here. `trigger_camera_focus_on_selection_change`
/// watches `selected_pos` and tweens on its own, so writing the caret is the
/// whole of what a row has to do — and the rebuild is asked for because the
/// caret is a scene entity like any other.
fn handle_diagnostics_click(
    header_q: Query<&Interaction, (Changed<Interaction>, With<DiagnosticsHeader>)>,
    row_q: Query<(&Interaction, &DiagnosticTarget), Changed<Interaction>>,
    mut open: ResMut<DiagnosticsOpen>,
    mut pick: ResMut<PickState>,
    state: Res<GraphState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    for interaction in header_q.iter() {
        if *interaction == Interaction::Pressed {
            open.0 = !open.0;
        }
    }
    for (interaction, target) in row_q.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let wanted = state.root_graph().clamp_to_volume(target.0);
        if pick.selected_pos != wanted {
            pick.selected_pos = wanted;
            rebuild.0 = true;
        }
    }
}

/// The editing mode, centred on the bottom edge — a control, not a readout.
///
/// A row spanning the window with its content centred, which is how the run
/// controls used to sit here: what stands in it can change width without
/// anything having to be measured. The two halves sit flush against each other
/// and are rounded only on their outer edge, so the pair reads as one widget
/// with two states rather than as two buttons. No rule between them: one half
/// is always on, and the fill it carries draws the seam.
///
/// The third child is what stands in the same place while an evaluation runs,
/// alone and rounded all round.
///
/// `EditorChrome` goes on the row and stays off the three inside it —
/// `sync_mode_toggles` is the only writer of their `display`, and two writers
/// flicker on the transition frame.
fn spawn_mode_display(mut commands: Commands, ui_font: Res<UiFont>) {
    const RADIUS: Val = Val::Px(6.0);
    let segment = |radius: BorderRadius, display: Display| Node {
        padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
        border_radius: radius,
        display,
        ..default()
    };
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                bottom: Val::Px(16.0),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::Center,
                ..default()
            },
            EditorChrome,
        ))
        .with_children(|row| {
            for (mode, label, radius) in [
                (EditorMode::Normal, "NORMAL", BorderRadius::left(RADIUS)),
                (EditorMode::Insert, "INSERT", BorderRadius::right(RADIUS)),
            ] {
                row.spawn((
                    Button,
                    segment(radius, Display::Flex),
                    BackgroundColor(CONTROL_REST),
                    ModeToggle(mode),
                ))
                .with_children(|half| {
                    half.spawn((
                        Text::new(label),
                        text_font(&ui_font.0, 14.0),
                        TextColor(Color::srgb(0.6, 0.6, 0.7)),
                    ));
                });
            }
            // A bare `Button` for the same reason the panels carry one: it is
            // what `pick_nodes`' `over_ui` test sees, so a click that lands on
            // it doesn't reach the cell behind it.
            row.spawn((
                Button,
                segment(BorderRadius::all(RADIUS), Display::None),
                BackgroundColor(CONTROL_REST),
                EvalModeLabel,
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new("EVALUATE"),
                    text_font(&ui_font.0, 14.0),
                    TextColor(Color::srgb(0.85, 0.85, 0.9)),
                ));
            });
        });
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

/// The mode control's only writer: which half is on, which one the mouse is
/// over, and whether either is offered at all.
///
/// Green stays the colour that means INSERT — it only moves from the word into
/// the half the word stands in. The half that is already on does not answer to
/// hover: a click on it does nothing, and a control that lights up under the
/// cursor promises that it would.
///
/// Nothing about the camera here any more. The free camera does suspend the
/// guarantees the bound one gives, and this used to say so — but the view
/// control now names the view outright, and it stands in the same row a hand's
/// width to the left. Two sentences about one fact read as two facts.
fn sync_mode_toggles(
    mode: Res<EditorMode>,
    eval: Res<EvalState>,
    mut toggle_q: Query<
        (
            &ModeToggle,
            &Interaction,
            &mut Node,
            &mut BackgroundColor,
            &Children,
        ),
        Without<EvalModeLabel>,
    >,
    mut eval_label_q: Query<&mut Node, With<EvalModeLabel>>,
    mut text_color_q: Query<&mut TextColor>,
) {
    // An evaluation owns the graph, so it owns the mode with it: neither half
    // is offered, and the word that stands there instead says whose it is.
    let evaluating = is_evaluating(&eval);
    let halves = if evaluating {
        Display::None
    } else {
        Display::Flex
    };
    for (toggle, interaction, mut node, mut bg, children) in toggle_q.iter_mut() {
        if node.display != halves {
            node.display = halves;
        }
        let Ok(mut text_color) = text_color_q.get_mut(children[0]) else {
            continue;
        };
        let (fill, tint) = if toggle.0 == *mode {
            match toggle.0 {
                EditorMode::Normal => (
                    Color::srgba(0.28, 0.28, 0.36, 0.95),
                    Color::srgb(0.85, 0.85, 0.9),
                ),
                // Brighter than NORMAL: INSERT is the state that changes the
                // graph, so it should be the one that catches the eye.
                EditorMode::Insert => (
                    Color::srgba(0.16, 0.34, 0.22, 0.95),
                    Color::srgb(0.35, 0.85, 0.55),
                ),
            }
        } else {
            match *interaction {
                Interaction::Hovered | Interaction::Pressed => (CONTROL_HOT, INK_BRIGHT),
                Interaction::None => (CONTROL_REST, INK_DIM),
            }
        };
        bg.0 = fill;
        *text_color = TextColor(tint);
    }
    let desired = if evaluating {
        Display::Flex
    } else {
        Display::None
    };
    for mut node in eval_label_q.iter_mut() {
        if node.display != desired {
            node.display = desired;
        }
    }
}

/// A grid address as the two lines above the panel write one.
fn cell_text(v: IVec3) -> String {
    format!("({},{},{})", v.x, v.y, v.z)
}

/// The way to the caret, written as the chain of offsets it is.
///
/// Each segment names the frame its numbers are measured in, and the numbers
/// are the step from that frame's origin to the next one — the last segment
/// steps to the caret itself. So the segments sum to the absolute address, and
/// that is the property worth having: the line can be checked by adding it up.
///
/// A nesting level becomes *two* segments, because the way into a branch goes
/// past a node that `scope.path` does not mention. The path holds only the
/// Pattern, so its frame is entered as `Match(…)` — the step from the Match
/// node to the branch's origin — after the enclosing frame has stepped to the
/// Match node itself. `Root(a) > Match(b) > Branch(c)` therefore reads: `a` to
/// the Match, `b` on into the branch, `c` to the caret.
///
/// A step that is not shaped like that — a path element that is no Pattern, a
/// Match with no layout position — is written as one segment carrying the
/// whole offset instead of two. The sum still holds; only the detail is lost.
fn relative_path_text(state: &GraphState, scope: &CaretScope) -> String {
    let root = state.root_graph();
    let mut parts: Vec<String> = Vec::new();
    for i in 0..scope.path.len() {
        // Frames alternate: the outermost is the root scope, every one after
        // it is the branch the previous step led into.
        let frame = if i == 0 { "Root" } else { "Branch" };
        let parent = root.resolve_context(&scope.path[..i]);
        let into_branch =
            root.scope_offset(&scope.path[..=i]) - root.scope_offset(&scope.path[..i]);
        let to_match = match parent.graph.nodes.get(&scope.path[i]) {
            Some(model::node::ENode::Pattern { parent_match, .. }) => parent
                .layout_nodes
                .get(parent_match)
                .map(|ln| ln.pos.round().as_ivec3()),
            _ => None,
        };
        match to_match {
            Some(to_match) => {
                parts.push(format!("{}{}", frame, cell_text(to_match)));
                parts.push(format!("Match{}", cell_text(into_branch - to_match)));
            }
            None => parts.push(format!("{}{}", frame, cell_text(into_branch))),
        }
    }
    let innermost = if scope.path.is_empty() {
        "Root"
    } else {
        "Branch"
    };
    parts.push(format!("{}{}", innermost, cell_text(scope.local)));
    parts.join(" > ")
}

/// Write both lines from one walk of the path.
///
/// Only on a change: a `Text` written every frame is a `Text` changed every
/// frame, and the layout is recomputed for it.
fn update_address_lines(
    pick: Res<PickState>,
    state: Res<GraphState>,
    mut line_q: Query<(&mut Text, &AddressLine)>,
) {
    let scope = state.scope_of_caret(&pick);
    for (mut text, line) in line_q.iter_mut() {
        let next = match line {
            // A caret inside no scope volume at all has no path to write —
            // its absolute address is still a fact, and stays below.
            AddressLine::Relative => match &scope {
                Some(scope) => format!("Relative: {}", relative_path_text(&state, scope)),
                None => "Relative: —".to_string(),
            },
            AddressLine::Absolute => format!("Absolute: {}", cell_text(pick.selected_pos)),
        };
        if text.0 != next {
            text.0 = next;
        }
    }
}

fn pick_nodes(
    camera_q: Query<(&Camera, &GlobalTransform), With<camera::OrbitCameraTag>>,
    windows: Query<&Window>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut pick: ResMut<PickState>,
    node_q: Query<(&NodeEntity, &Transform)>,
    grid_q: Query<(Entity, &ScopeGridEntity)>,
    state: Res<GraphState>,
    eval: Res<EvalState>,
    ui_interactions: Query<&Interaction, With<Button>>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    // A modal owns the screen, and what was hovered or half-pressed under it
    // is not what the user will be looking at when it closes.
    if modal_is_open(&eval) {
        pick.hovered_node = None;
        pick.hovered_grid = None;
        pick.press_cursor = None;
        pick.press_over_ui = false;
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

    // Ray-plane test against each spawned graph grid. Each graph grid is its
    // scope's floor — one row-edge below the scope's last row, at its own Y —
    // and spans a rectangle in local grid coords. Pick the closest rect hit.
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
            // The plane the user can see is the floor, not the address plane
            // the scope's origin sits on, and the ray has to meet the surface
            // that is actually there. Derived from `max` rather than carried on
            // the component, so it cannot drift from where the mesh is spawned.
            let floor_world_y = render::layout_to_world(
                ag.origin_offset + Vec3::new(0.0, (ag.max.y + 1) as f32, 0.0),
            )
            .y;
            let t = (floor_world_y - ray.origin.y) / denom;
            if t <= 0.0 || t >= best_t {
                continue;
            }
            let hit = ray.origin + *ray.direction * t;
            // Cells are corner-anchored — cell N covers [N, N+1) — so the
            // containing cell is the floor, not the nearest address.
            //
            // Read off the scope's origin and not off the plane: the plane's Y
            // says *which* grid was hit, the origin says which cell of it. Only
            // X and Z are taken, so the Y this leaves behind is never read.
            let local = render::world_to_layout(hit - origin_world);
            let local_x = local.x.floor() as i32;
            let local_z = local.z.floor() as i32;
            if local_x < ag.min.x || local_x > ag.max.x {
                continue;
            }
            if local_z < ag.min.z || local_z > ag.max.z {
                continue;
            }
            // The cell a floor belongs to is the one standing on it: the
            // volume's last row, which is the row the plane bounds from below.
            let cell_local = IVec3::new(local_x, ag.max.y, local_z);
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

fn update_grid_material(
    pick: Res<PickState>,
    state: Res<GraphState>,
    screenshot: Res<ScreenshotMode>,
    grid_q: Query<(
        Entity,
        &ScopeGridEntity,
        &MeshMaterial3d<grid::GridMaterial>,
    )>,
    mut materials: ResMut<Assets<grid::GridMaterial>>,
) {
    // Both of these say where something *is* rather than what the program does
    // — the border where the caret stands, the highlight where the pointer
    // does. In screenshot mode neither is on screen to be pointed at, so
    // neither marks anything.
    let marking = !screenshot.active();
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
        if marking && Some(entity) == hit_entity {
            mat.hover_pos = hit_center;
            mat.hover_active = 1.0;
        } else {
            mat.hover_active = 0.0;
        }
        mat.border_active =
            if marking && Some(scope_grid.context.as_slice()) == caret_path.as_deref() {
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
    //
    // Enter counts as well as Escape: both mean "done here", and a field that
    // keeps focus keeps `keyboard_captured` true, which would leave the graph
    // deaf to `hjkl` until something was clicked. In a modal Enter stays
    // inert — there the buttons commit.
    let finished = key_events
        .read()
        .filter(|ev| ev.state == bevy::input::ButtonState::Pressed)
        .fold((false, false), |(esc, enter), ev| {
            (
                esc || matches!(ev.logical_key, bevy::input::keyboard::Key::Escape),
                enter || matches!(ev.logical_key, bevy::input::keyboard::Key::Enter),
            )
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

        let (escaped, entered) = finished;
        if escaped || (entered && modal_tag.is_none()) {
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

/// Keeps the prompt standing on the property the caret addresses: entering
/// INSERT on a cell that already holds something opens on what it holds, so
/// changing a value starts from the value instead of from nothing.
///
/// And where the caret has moved *off* a node that was still owing its one
/// mandatory property, it unmakes that node — the mouse's half of what Escape
/// does for the keyboard. The two jobs are one system because they are one
/// question asked once: what does the caret address now, and what did it
/// address before.
///
/// Re-seeds when the mode or the target changes and at no other time. A
/// keystroke changes neither, so typing is safe from it; a mouse click can move
/// the caret under INSERT, which is exactly the case a one-shot seed at the `i`
/// keypress would miss — and the only way to walk away from an unfinished node,
/// since INSERT takes the arrow keys away.
///
/// It stands where the Source name field's focus projection used to. That field
/// is gone — every property is typed into the one prompt now — so what has to
/// be projected from the mode is no longer a focus but the text itself.
fn sync_prompt_seed(
    mode: Res<EditorMode>,
    mut state: ResMut<GraphState>,
    pick: Res<PickState>,
    mut prompt: ResMut<InsertPrompt>,
    mut pending: ResMut<PendingNode>,
    mut rebuild: ResMut<NeedsRebuild>,
    mut last: Local<Option<(EditorMode, InsertTarget)>>,
) {
    let mut target = insert_target(&state, &pick);
    if last.as_ref() == Some(&(*mode, target.clone())) {
        return;
    }

    // A node is only unfinished while the caret is still standing on it, and
    // leaving it is the same act as cancelling it: an insert that was walked
    // away from leaves no placeholder behind, exactly as an Escape leaves
    // none. Only the caret differs — Escape puts it back where the insert
    // started, while a click is itself a statement about where it should be,
    // so it stays where it was put.
    //
    // A click is the only way to get here, since INSERT takes the arrow keys
    // away. That is why this is not the same rule said twice: Escape is what
    // the keyboard has, this is what the mouse has, and they agree.
    //
    // "Still standing on it" asks after the node and not after its property
    // cell: a pending node may have cells that name nothing — an anchor, a gap
    // — and stepping onto one of those is not walking away from anything.
    let pending_here = match &pending.0 {
        Some(edit) => addressed_cell(&state, &pick).is_some_and(|(id, _)| id == edit.node),
        None => false,
    };
    if !pending_here {
        if let Some(edit) = pending.0.take() {
            if remove_node(&mut state, &edit.node) {
                rebuild.0 = true;
                // Removing a node re-settles the layout, so what the caret
                // addresses has to be asked again before the prompt opens on
                // it.
                target = insert_target(&state, &pick);
            }
        }
    }
    *last = Some((*mode, target.clone()));

    match (*mode, &target) {
        (EditorMode::Insert, InsertTarget::Edit(id, edit)) => {
            // A node that was built a keystroke ago has nothing to open on, and
            // this needs no special case to know it: the property it is being
            // asked for is exactly the one it does not carry, and
            // `property_text` spells an absent property as the empty string.
            prompt.set_text(property_text(&state, id, edit));
            reselect(&mut prompt, &state, &pick);
        }
        // A create prompt has nothing to open on, and NORMAL has no prompt at
        // all — both start from empty.
        _ => prompt.clear(),
    }
}

fn text_input_keyboard(
    mut input_q: Query<(&mut TextInput, &Children), With<TextInputBox>>,
    mut text_q: Query<&mut Text, With<TextInputDisplay>>,
    mut key_events: MessageReader<KeyboardInput>,
) {
    // Read once, outside the loop over the fields. Two reasons, and the first
    // is a bug the inner `read()` used to hide: with nothing focused the
    // reader was never touched, so the batch stood until the next frame — and
    // the key that *gave* a field its focus then got typed into it. The second
    // is that the first focused field used to drain the batch, starving any
    // other.
    let pressed: Vec<&KeyboardInput> = key_events
        .read()
        .filter(|ev| ev.state == bevy::input::ButtonState::Pressed)
        .collect();

    for (mut input, children) in input_q.iter_mut() {
        if input.focused {
            for ev in &pressed {
                let input_cursor = input.cursor;

                match &ev.logical_key {
                    bevy::input::keyboard::Key::Character(s) => {
                        input.value.insert_str(input_cursor, s.as_str());
                        input.cursor += s.len();
                    }
                    bevy::input::keyboard::Key::Space => {
                        input.value.insert(input_cursor, ' ');
                        input.cursor += 1;
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
                        }
                    }
                    bevy::input::keyboard::Key::Delete => {
                        if input.cursor < input.value.len() {
                            input.value.remove(input_cursor);
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
                    _ => {}
                }
            }
        }

        // Outside the focus test on purpose: a field that just lost focus has
        // to lose its cursor bar with it, and only this writes the text.
        // Guarded on inequality, because writing it unconditionally would
        // dirty every text node in the UI every frame.
        if let Ok(mut text) = text_q.get_mut(children[0]) {
            let (before, after) = input.value.split_at(input.cursor);
            let wanted = if input.focused {
                format!("{}|{}", before, after)
            } else {
                input.value.clone()
            };
            if text.0 != wanted {
                text.0 = wanted;
            }
        }
    }
}

/// Detect a change in `PickState::selected_pos` and start a camera
/// auto-focus tween toward the new position.
///
/// The first observation (fresh `Local`) only records where the caret stands.
/// There is nothing to travel to: `setup_scene` has already put the camera on
/// that cell, so startup is a view that is right rather than one that arrives.
fn trigger_camera_focus_on_selection_change(
    pick: Res<PickState>,
    orbit: Res<camera::OrbitCamera>,
    mut tween: ResMut<camera::CameraTween>,
    mut last_selection: Local<Option<IVec3>>,
) {
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

/// True while something other than the graph owns the keyboard: a modal's text
/// field, a modal, or a running evaluation. Mode switching and the INSERT-mode
/// inserts stay out of the way then — otherwise `i` would both type an `i` and
/// change the mode.
///
/// The editor itself never appears here any more: its one text field was the
/// Source's name, and that is typed into the prompt now, which is deliberately
/// not a `TextInput` precisely so it does not capture.
fn keyboard_captured(text_inputs: &Query<&TextInput>, eval: &EvalState) -> bool {
    modal_is_open(eval) || is_evaluating(eval) || text_inputs.iter().any(|input| input.focused)
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
/// then INSERT's jobs — the room-makers `Space`, `Return` and `Shift+Return`,
/// which act on the scope the caret addresses and in that scope's local
/// coordinates, and the prompt, into which every other key writes.
///
/// What the prompt is *for* is a question about the caret's cell, not about
/// this system: on a free cell it names what to create, on a cell that stands
/// for a property it names that property's new value (`insert_target`). The
/// room-makers belong to the first case alone — where the prompt edits
/// something that already stands there, making room is not an answer to
/// anything, and `Space` is simply a character. On a create prompt `Space`
/// belongs to the prompt for the length of an unclosed quote, and only then.
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
/// While something else owns the keyboard the batch is walked but ignored,
/// rather than left standing: a message survives two updates, so keys struck
/// under the guard would otherwise fire once it lifts — and a field's text
/// would land in the prompt. There are no exceptions left: the editor's own
/// text field is gone, so whatever owns the keyboard is a modal or an
/// evaluation, and neither is a mode to leave.
fn handle_editor_keys(
    mut key_events: MessageReader<KeyboardInput>,
    keys: Res<ButtonInput<KeyCode>>,
    text_inputs: Query<&TextInput>,
    eval: Res<EvalState>,
    mut mode: ResMut<EditorMode>,
    mut prompt: ResMut<InsertPrompt>,
    mut pending: ResMut<PendingNode>,
    mut state: ResMut<GraphState>,
    mut pick: ResMut<PickState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    let captured = keyboard_captured(&text_inputs, &eval);
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let target = insert_target(&state, &pick);

    for ev in key_events.read() {
        if ev.state != bevy::input::ButtonState::Pressed {
            continue;
        }
        if captured {
            // Nothing in the editor takes the keyboard any more — every
            // property is typed into the prompt, which deliberately is not a
            // text field — so what is left owning it is the start menu, a modal
            // and a running evaluation, none of which belong to the editor at
            // all. The batch is walked rather than left standing, because a
            // message survives two updates and would otherwise fire once the
            // guard lifts.
            continue;
        }
        match (*mode, &ev.logical_key) {
            (_, bevy::input::keyboard::Key::Escape) => {
                // Escape leaves INSERT outright, which is the same act the
                // control's NORMAL half performs.
                if leave_insert_mode(
                    &mut state,
                    &mut pick,
                    &mut prompt,
                    &mut pending,
                    &mut mode,
                    &mut rebuild,
                ) {
                    // A placeholder went with it: the rest of the batch was
                    // struck against a graph that no longer holds what those
                    // keys were about.
                    break;
                }
            }
            // TAB walks the panel's rows, and the caret walks with it — a row
            // and a cell are one address seen from two sides. Both modes,
            // because the panel stands in both; in INSERT what was typed and
            // not committed falls away with the move, the same as on `Escape`,
            // and `sync_prompt_seed` opens the row it arrives on.
            (_, bevy::input::keyboard::Key::Tab) if !ev.repeat => {
                let Some(subject) = panel_subject(&state, &pick) else {
                    continue;
                };
                let here = caret_row_address(&state, &pick);
                let Some(cell) = tab_step(&subject, &here, shift) else {
                    continue;
                };
                let wanted = state.root_graph().clamp_to_volume(cell);
                if wanted != pick.selected_pos {
                    pick.selected_pos = wanted;
                    // The caret is a scene entity, so where it stands is drawn
                    // rather than moved.
                    rebuild.0 = true;
                    // The rest of the batch was struck against the cell the
                    // caret was standing on.
                    break;
                }
            }
            (EditorMode::Normal, bevy::input::keyboard::Key::Character(s)) if s.as_str() == "i" => {
                prompt.clear();
                *mode = EditorMode::Insert;
                rebuild.0 = true;
            }
            // Deleting is the one edit NORMAL makes itself. In INSERT the
            // same key is the prompt's forward delete, an arm below.
            (EditorMode::Normal, bevy::input::keyboard::Key::Delete) if !ev.repeat => {
                if delete_node_at_caret(&mut state, &pick) {
                    rebuild.0 = true;
                    // The rest of the batch was struck against a graph that no
                    // longer holds what those keys were about.
                    break;
                }
            }
            // NORMAL's own keys belong to `handle_arrow_keys`.
            (EditorMode::Normal, _) => {}
            (EditorMode::Insert, key) => match key {
                bevy::input::keyboard::Key::Character(s) => {
                    // A single press can carry more than one character when a
                    // dead key resolves; control characters are not a name.
                    if !s.is_empty() && !s.chars().any(|c| c.is_control()) {
                        prompt.insert_str(s.as_str());
                        reselect(&mut prompt, &state, &pick);
                    }
                }
                // Deleting back to an empty text drops the highlight with it,
                // which on a create prompt is the other way out of the list:
                // `Return` goes back to opening a column.
                bevy::input::keyboard::Key::Backspace => {
                    if let Some(prev) = prompt.prev_boundary() {
                        prompt.text.remove(prev);
                        prompt.cursor = prev;
                        reselect(&mut prompt, &state, &pick);
                    }
                }
                // Deletes forward, so the cursor does not move.
                bevy::input::keyboard::Key::Delete => {
                    if prompt.next_boundary().is_some() {
                        let at = prompt.cursor;
                        prompt.text.remove(at);
                        reselect(&mut prompt, &state, &pick);
                    }
                }
                // The list claims Up and Down; Left and Right are the text's,
                // which is what makes the prompt usable when it opens on a
                // value that is already there.
                bevy::input::keyboard::Key::ArrowLeft => {
                    if let Some(prev) = prompt.prev_boundary() {
                        prompt.cursor = prev;
                    }
                }
                bevy::input::keyboard::Key::ArrowRight => {
                    if let Some(next) = prompt.next_boundary() {
                        prompt.cursor = next;
                    }
                }
                bevy::input::keyboard::Key::Home => prompt.cursor = 0,
                bevy::input::keyboard::Key::End => prompt.cursor = prompt.text.len(),
                bevy::input::keyboard::Key::ArrowDown | bevy::input::keyboard::Key::ArrowUp => {
                    let delta = if matches!(key, bevy::input::keyboard::Key::ArrowDown) {
                        1
                    } else {
                        -1
                    };
                    let candidates = prompt_candidates(&state, &pick, &prompt.text);
                    prompt.selected = step_selection(&candidates, prompt.selected, delta);
                }
                // On a cell that names a property there is no room to make, so
                // the space bar is simply a character — which is what keeps a
                // Source name with a space in it typeable.
                //
                // Inside an unclosed quote it writes one on a create prompt
                // too: `Key::Space` is its own variant, so the `Character` arm
                // never sees one and a string could otherwise not hold one.
                // That guard needs a non-empty text and the room-maker below an
                // empty one, so the two can never both fire.
                bevy::input::keyboard::Key::Space
                    if target != InsertTarget::Create || in_open_quote(&prompt.text) =>
                {
                    prompt.insert_str(" ");
                    reselect(&mut prompt, &state, &pick);
                }
                // The room-makers belong to the prompt that builds: where the
                // prompt edits a property of a node that already stands there,
                // making room is not an answer to anything. They are also
                // bounded to one insert per press — a held key auto-repeats,
                // and the repeats must not each open a cell.
                bevy::input::keyboard::Key::Space
                    if target == InsertTarget::Create && prompt.text.is_empty() && !ev.repeat =>
                {
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
                // `Return` is the newline only while the list has not answered:
                // a standing highlight — typed to or walked to — takes the key
                // for the commit below, because that is what a highlight means.
                // The empty text is still part of the guard beside it: a
                // half-written name that matches nothing has no highlight
                // either, and must not open a column behind the typist's back.
                bevy::input::keyboard::Key::Enter
                    if target == InsertTarget::Create
                        && prompt.text.is_empty()
                        && prompt.selected.is_none()
                        && !ev.repeat =>
                {
                    let inserted = if shift {
                        apply_room_insert(
                            &mut state,
                            &mut pick,
                            // Inside a Match a row belongs to the arm stack:
                            // it opens between two arms and the scope keeps
                            // its height, because the space is the Match's own
                            // and grows its footprint rather than the volume.
                            // Outside one, the row is the scope's and every
                            // node behind it steps back.
                            |graph, scope| match graph.match_containing(scope.local) {
                                Some(match_id) => graph.plus_arm_row(&match_id, scope.local.y),
                                None => graph.plus_empty_slab(layout::Axis::Y, scope.local.y),
                            },
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
                    // Commit the highlighted row — and with no highlight,
                    // nothing: a prompt that nothing answers has not been
                    // answered yet, and there is no second meaning for the key
                    // to fall back on the way the create prompt has its column.
                    //
                    // What the commit leaves behind — whether INSERT goes on
                    // and whether a node is owed a property — is
                    // `commit_outcome`'s, because the mouse path has to reach
                    // the same conclusion.
                    let candidates = prompt_candidates(&state, &pick, &prompt.text);
                    if let Some(action) = clamped_selection(&candidates, prompt.selected)
                        .and_then(|selected| candidates.get(selected))
                        .filter(|suggestion| suggestion.allowed)
                        .map(|suggestion| suggestion.action.clone())
                    {
                        if let Some(outcome) = apply_prompt_action(&mut state, &mut pick, &action) {
                            commit_outcome(
                                &state,
                                &mut pick,
                                outcome,
                                &mut prompt,
                                &mut pending,
                                &mut mode,
                                &mut rebuild,
                            );
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
    //
    // `keyboard_captured` covers the arrows here, not just the letters. It
    // used to gate only `hjkl`, and the arrows — which come from `ButtonInput`
    // rather than from the message — walked straight past it: typing in a
    // field, Left and Right moved the text cursor *and* the caret, and the
    // travelling caret pulled the panel out from under the typist.
    if is_evaluating(&eval) || *mode == EditorMode::Insert || keyboard_captured(&text_inputs, &eval)
    {
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
    // QWERTY position. Repeats are dropped so a held key steps once, the way
    // `just_pressed` bounds the arrows.
    let letter = key_events
        .read()
        .filter(|ev| ev.state == bevy::input::ButtonState::Pressed && !ev.repeat)
        .find_map(|ev| match &ev.logical_key {
            bevy::input::keyboard::Key::Character(s) => {
                let mut chars = s.chars();
                // Shift reports the uppercase character; the shifted meaning
                // comes from the modifier, as it does for the arrows.
                let c = chars.next()?.to_ascii_lowercase();
                (chars.next().is_none() && matches!(c, 'h' | 'j' | 'k' | 'l')).then_some(c)
            }
            _ => None,
        });

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
        // Ein Ziel in einem fremden Scope schnappt nur ein, wenn es der Input
        // eines Tunnels genau eine Ebene tiefer ist. Hier und nicht erst beim
        // Drop, damit die Preview-Linie gelb bleibt statt eine Verbindung zu
        // versprechen, die drag_end ohnehin verwirft.
        //
        // Vorher in die Richtung gebracht, in der die Kante gespeichert würde:
        // `connection_allowed` spricht von Erzeuger -> Verbraucher, gezogen
        // werden darf aber von beiden Enden aus. Ungedreht würde ein Zug, der
        // am Tunnel-Input beginnt und beim Source endet, als Verbindung aus
        // dem Branch heraus gelesen und abgelehnt.
        let (from, to) = if info.source_is_output {
            (&info.source_anchor_id, &target_id)
        } else {
            (&target_id, &info.source_anchor_id)
        };
        let is_permitted = connection_allowed(state.root_graph(), from, to);
        if !is_self && is_opposite_kind && !is_duplicate && is_permitted {
            info.target_anchor_id = Some(target_id);
            info.current_end = tf.translation();
        }
    }
}

/// True if `a` and `b` are already joined by an edge, in either stored
/// direction.
///
/// Not what keeps a pair from being doubled any more — `LayoutGraph::plus_edge`
/// clears the target input before wiring, so a second edge onto it is not a
/// thing that can exist. What this still answers is whether there is anything
/// to do: re-dragging a connection that already stands would drop it and put
/// the identical one back, and a rebuild for that is a flicker in exchange for
/// nothing.
///
/// The reverse direction is checked too because edges recorded before drag-end
/// started normalising to output → input may still sit the other way around,
/// and `eval::neighbours_of_anchor` treats both orientations as connected.
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

/// Whether an edge may join these two anchors at all.
///
/// A value enters a sub-graph through a Tunnel or it does not enter. Within
/// one scope everything is as it was — that is the ordinary case and this says
/// nothing about it — but the moment the two ends sit in different scopes,
/// exactly one shape is admitted: an output of the immediately enclosing graph
/// reaching the **input** of a Tunnel one level in.
///
/// Everything else that crosses a boundary is refused, and the refusals are
/// the point rather than a side effect:
///
/// - onto an ordinary node of a branch, which is the hole this closes: a value
///   could be wired straight onto whatever wanted it, and nothing in the
///   branch recorded that it came from outside.
/// - out of a branch into the enclosing graph. A branch's result leaves
///   through its Sink and the owning Match's output; an edge doing it too
///   would be a second way out that no arm accounts for.
/// - from inside a branch into that branch's own Tunnel — a value entering
///   from where it already is.
/// - past a level. A Tunnel is reachable from its parent, not from its
///   grandparent: a value that skipped a scope would cross that scope's
///   volume without appearing anywhere in it.
fn connection_allowed(
    root: &layout::LayoutGraph,
    source: &model::anchor::Id,
    target: &model::anchor::Id,
) -> bool {
    let scope_of = |anchor: &model::anchor::Id| {
        let la = root.try_layout_anchor(anchor)?;
        let path = root.context_of_node(&la.node_id)?;
        Some((path, la.node_id))
    };
    let (Some((source_path, _)), Some((target_path, target_node_id))) =
        (scope_of(source), scope_of(target))
    else {
        return false;
    };

    // Asked of the target first, because it is the target that decides which
    // of the two rules applies. A Tunnel's *input* is the only anchor in the
    // program with its own — its output faces the branch like any other
    // producer and is wired to from inside, as normal.
    let target_is_tunnel_input = matches!(
        root.find_node_graph(&target_node_id)
            .and_then(|g| g.graph.nodes.get(&target_node_id)),
        Some(model::node::ENode::Tunnel { input_anchor, .. }) if input_anchor == target
    );

    if target_is_tunnel_input {
        // Fed from exactly one scope out, and from nowhere else — not from
        // further out, which would skip a scope, and not from the branch it
        // opens into, which would be a value entering from where it already
        // is. Note this is the *only* way in, so the same-scope case below
        // must not be allowed to answer for it first.
        return target_path.len() == source_path.len() + 1
            && target_path[..source_path.len()] == source_path[..];
    }

    // Everything else stays where it is. This is the ordinary case and the
    // whole of the old behaviour; what has changed is that it is now the
    // *only* other case.
    source_path == target_path
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
                // `connection_allowed` wird über die gespeicherte Richtung
                // gefragt, nicht über die gezogene: die Regel spricht von
                // Output -> Tunnel-Input, und genau dieses Paar ist `from`,
                // `to`. Nochmal geprüft wie die beiden Invarianten daneben —
                // drag_update lässt so ein Ziel nicht einschnappen, aber
                // zwischen Snap und Drop kann ein Rebuild liegen.
                if from != to
                    && !anchors_already_connected(state.root_graph(), &from, &to)
                    && connection_allowed(state.root_graph(), &from, &to)
                {
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
        .add_plugins(depth_cue::DepthCuePlugin)
        .init_resource::<GraphState>()
        .init_resource::<NeedsRebuild>()
        .init_resource::<PickState>()
        .init_resource::<DragState>()
        .init_resource::<EvalState>()
        .init_resource::<EditorMode>()
        .init_resource::<InsertPrompt>()
        .init_resource::<PendingNode>()
        .init_resource::<ScreenshotMode>()
        .init_resource::<lod::Clipping>()
        .init_resource::<Diagnostics>()
        .init_resource::<DiagnosticsOpen>()
        .init_resource::<ViewMenuOpen>()
        .add_systems(
            Startup,
            (
                load_ui_font,
                setup_scene,
                spawn_graph_nodes,
                spawn_ui,
                spawn_view_bar,
                spawn_editor_column,
                spawn_run_panel,
                spawn_fps_display,
                spawn_mode_display,
                // Last: it reads the window's physical size, and the camera
                // above is what puts the OIT settings there to be read.
                log_oit_budget,
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
                        // First, because the mode it ends is the reason the
                        // caret is missing: leaving has to be seen before
                        // `rebuild_scene` at the end of this chain, or the
                        // caret comes back a frame late.
                        end_screenshot_mode,
                        handle_mode_toggle,
                        handle_insert_prompt_click,
                        handle_new_button,
                        handle_help_button,
                        sync_chrome_buttons,
                        // With the other click handlers, not with the modal
                        // ones: it rewrites the graph, and this is the chain
                        // that ends in `rebuild_scene`, so the fresh graph is
                        // on screen in the frame it was asked for.
                        handle_confirm_new_button,
                        sync_editor_chrome,
                        pick_nodes,
                    )
                        .chain()
                        // A click resolves before the keys of the same frame.
                        // `handle_mode_toggle` sets the mode and `pick_nodes`
                        // moves the caret, and the keyboard chain projects both
                        // onto the prompt's text at its end — ambiguous, that
                        // projection could run on the state the click was about
                        // to change.
                        .before(handle_editor_keys),
                    highlight_hovered,
                    update_grid_material,
                    update_cursor,
                    // Mode and caret before keys before the projection of the
                    // two back onto the prompt's text. The last link has to be
                    // last: it re-seeds the prompt when the addressed property
                    // changes, and running it before the keys would wipe what
                    // was typed in the same frame the caret was moved.
                    (
                        handle_editor_keys,
                        text_input_focus,
                        text_input_keyboard,
                        sync_prompt_seed,
                    )
                        .chain(),
                    handle_arrow_keys,
                    trigger_camera_focus_on_selection_change,
                ),
                (
                    anchor_hover_system,
                    drag_start_system,
                    drag_update_system,
                    drag_end_system,
                    // Between the press and the teardown, so a step is
                    // redrawn in the frame it moved in rather than the one
                    // after. `apply_run_to_end` ends the chain that every way
                    // into and out of `Running` goes through, which is why
                    // naming it alone is enough.
                    sync_eval_rebuild.after(apply_run_to_end),
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
                update_address_lines,
                // Unordered on purpose. `Interaction` is written in
                // `PreUpdate`, so whatever this paints was decided before any
                // of the syncs around it ran, and a control respawned during
                // `Update` takes its colour the next time the focus pass sees
                // it — one frame, and the same one a click already costs.
                paint_hover,
                sync_mode_toggles,
                blink_caret,
                // Chained: the click may fold the list, and the sync that draws
                // it has to see the fold in the same frame it was asked for.
                (
                    recompute_diagnostics,
                    handle_diagnostics_click,
                    sync_diagnostics_ui,
                )
                    .chain(),
            ),
        )
        .add_systems(
            Update,
            (
                handle_evaluate_button,
                // Chained: the click folds or unfolds the list, and the sync
                // that draws it has to see the fold in the same frame.
                (handle_view_menu_click, sync_view_menu).chain(),
                // Chained for the same reason: a handler runs before the sync
                // that reads what it wrote.
                (handle_clipping_checkbox, sync_clipping_checkbox).chain(),
                handle_modal_ok_button,
                handle_modal_cancel_button,
                handle_modal_evaluate_button,
                handle_eval_step_buttons,
                handle_full_run_button,
                apply_run_to_end,
                sync_modal_ui,
                sync_player_controls,
                update_step_button_visuals,
                sync_value_labels,
            )
                .chain(),
        )
        .add_systems(
            Update,
            sync_editor_panel
                // The panel draws what was typed this frame, so it has to run
                // after the keys landed and after the seed that follows them —
                // otherwise the prompt shows the previous frame's text every
                // other frame.
                .after(sync_prompt_seed),
        )
        .run();
}
