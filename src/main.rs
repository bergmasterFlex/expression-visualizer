mod camera;
mod colors;
mod common;
mod depth_cue;
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
/// leaving behind a placeholder nobody chose. Three kinds need this: a
/// `TypeCast`, a `Pattern` and a `Match`, whose property is not part of the
/// name that was typed to create it. A `Constant` carries its literal and a
/// `FunctionCall` its function already, and a `Source` has nothing mandatory —
/// its name may be empty and its type is settled on a second cell.
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

/// Marker on what the prompt respawns on a change — its column, the text row
/// and the suggestion list. Only those carry it: `despawn` takes their children
/// with them, and an entity despawned twice warns.
#[derive(Component, Clone)]
struct InsertPromptEntity;

/// A clickable suggestion row, so the mouse path the "Add" button opens does
/// not dead-end at a keyboard-only list.
#[derive(Component)]
struct InsertPromptOption(PromptAction);

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

/// Which Match owns each Pattern, over every scope in the scene.
///
/// A scope's `context` names only the Patterns descended through, because a
/// Match used to be nothing a walk could stop at. It is a volume of its own now,
/// sitting between a scope and its arms, so the path a boundary count is taken
/// over has to name it too — and this is what turns the one into the other.
fn owning_matches(
    root: &layout::LayoutGraph,
) -> std::collections::HashMap<model::node::Id, model::node::Id> {
    let mut owning = std::collections::HashMap::new();
    for walked in root.walk_all_graphs() {
        for (match_id, node) in &walked.layout_graph.graph.nodes {
            let model::node::ENode::Match { patterns, .. } = node else {
                continue;
            };
            for pattern_id in patterns {
                owning.insert(pattern_id.clone(), match_id.clone());
            }
        }
    }
    owning
}

/// The volume path of the scope `context` names: every Pattern in it preceded by
/// the Match it is an arm of.
///
/// A Pattern with no owner is skipped rather than passed through — a path is
/// only good for counting boundaries against another path, and a rung that
/// silently went missing would make two volumes look closer than they are.
fn scope_volume_path(
    context: &[model::node::Id],
    owning: &std::collections::HashMap<model::node::Id, model::node::Id>,
) -> Vec<model::node::Id> {
    let mut path = Vec::with_capacity(context.len() * 2);
    for pattern_id in context {
        let Some(match_id) = owning.get(pattern_id) else {
            continue;
        };
        path.push(match_id.clone());
        path.push(pattern_id.clone());
    }
    path
}

/// Volume boundaries between two volumes: the walls crossed going from one to
/// the other through their nearest common ancestor.
///
/// Siblings come out two apart, not one — there is a wall out of the first and a
/// wall into the second, and no shortcut between them. That is what makes an arm
/// and its sister arm read as exactly as far from each other as either is from
/// the scope holding their Match.
fn volume_boundaries(a: &[model::node::Id], b: &[model::node::Id]) -> usize {
    let common = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    (a.len() - common) + (b.len() - common)
}

impl GraphState {
    /// The volume the caret stands in, as a path `volume_boundaries` can measure
    /// from.
    ///
    /// One step further than `scope_of_caret`: a Pattern's gap, the cell naming
    /// its type, and the Match's own input and output belong to no branch, so
    /// `scope_at` hands them to the enclosing scope — but they are the Match's
    /// own cells, and standing on one of them is standing in the Match.
    ///
    /// At most one Match of a scope can hold the caret: a nested Match lives in
    /// a branch's own sub-layout, which `scope_at` would have resolved to first.
    ///
    /// `None` where `scope_of_caret` is `None` — the caret outside every volume,
    /// where editing is unavailable. Nothing is faded then: there is no volume
    /// to measure from, and measuring anyway would be a fiction.
    fn volume_of_caret(
        &self,
        pick: &PickState,
        owning: &std::collections::HashMap<model::node::Id, model::node::Id>,
    ) -> Option<Vec<model::node::Id>> {
        let scope = self.scope_of_caret(pick)?;
        let mut path = scope_volume_path(&scope.path, owning);
        let graph = self.root_graph().resolve_context(&scope.path);
        for (id, node) in &graph.graph.nodes {
            if !matches!(node, model::node::ENode::Match { .. }) {
                continue;
            }
            let Some(fp) = graph.match_footprint(id) else {
                continue;
            };
            if scope.local.cmpge(fp.min).all() && scope.local.cmple(fp.max).all() {
                path.push(id.clone());
                break;
            }
        }
        Some(path)
    }

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
/// standing on it and choosing what to change are one act — which is why there
/// is no property panel offering a node's type and its value at the same time.
#[derive(Clone, PartialEq, Eq)]
enum EditTarget {
    /// Printed along the body, so the body is where it is typed.
    SourceName,
    /// Hangs off the output anchor, so that is where it is declared.
    SourceType,
    /// Both of a Constant's cells answer for its literal: the value *is* the
    /// node, so there is no second thing either cell could mean. The type is
    /// not carried separately — it is whatever the literal spells.
    ConstantValue,
    /// The cell between a TypeCast's two anchors: a base type to cast to, or a
    /// literal to produce. Its anchors name nothing.
    CastType,
    /// The type a Pattern's arm matches, on the Pattern's one cell.
    PatternType,
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
    let property = match (node, &role) {
        (model::node::ENode::Source { .. }, layout::CellRole::Body) => EditTarget::SourceName,
        (model::node::ENode::Source { .. }, layout::CellRole::Output { .. }) => {
            EditTarget::SourceType
        }
        // The one kind whose two cells say the same thing, because it only has
        // the one thing to say.
        (
            model::node::ENode::Constant { .. },
            layout::CellRole::Body | layout::CellRole::Output { .. },
        ) => EditTarget::ConstantValue,
        (model::node::ENode::TypeCast { .. }, layout::CellRole::Body) => EditTarget::CastType,
        (model::node::ENode::Pattern { .. }, layout::CellRole::Body) => EditTarget::PatternType,
        // A FunctionCall's body cells fall through with everything else: the
        // function is not changeable, so they name nothing. See `EditTarget`.
        _ => return InsertTarget::Create,
    };
    InsertTarget::Edit(id, property)
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
        (EditTarget::SourceType, model::node::ENode::Source { r#type, .. }) => {
            type_choice_label(type_choice_of(r#type)).to_string()
        }
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

/// Tag on every descendant of the editor panel, which is rebuilt whenever the
/// addressed cell or what is being typed into it changes.
#[derive(Component, Clone)]
struct NodeEditorEntity;

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
        // Reads the depth buffer back in a full-screen pass. OIT is what makes
        // that possible without a depth prepass: it already marks the depth
        // texture as bindable, so the cue samples the one the main pass wrote.
        // Off until F9 says otherwise.
        depth_cue::DepthCue::default(),
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
    state: Res<GraphState>,
    ui_font: Res<UiFont>,
    pick: Res<PickState>,
    editor_mode: Res<EditorMode>,
) {
    let mut node_entites = std::collections::HashMap::<model::node::Id, Entity>::new();
    let mut anchor_entities = std::collections::HashMap::<model::anchor::Id, Entity>::new();
    let mut anchor_world_positions = std::collections::HashMap::<model::anchor::Id, Vec3>::new();
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
            walked.extra_offset,
        );
        // A node drawn only as markers or loose objects has no mesh of its own;
        // the entity is still spawned so picking and selection keep working.
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

        // Meshes a node owns without their being its body or one of its
        // anchors: the line and the point a Constant is drawn as. They carry no
        // `NodeEntity`, so nothing picks them and nothing hovers them — the
        // node's own entity, spawned above, answers for all of that.
        for object in render_node.objects {
            commands.spawn((
                Mesh3d(meshes.add(object.mesh)),
                MeshMaterial3d(materials.add(object.material)),
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
        if source_leaves.is_empty() {
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
        let target_leaves = tgt_type
            .as_ref()
            .map(|t| render::ordered_supported_leaves(t))
            .unwrap_or_default();

        // graph-level literal on the source anchor. When present, the sole
        // rendered leaf swaps to the thin "value line" style — same rule
        // the anchor markers follow, via the same lookup.
        let src_graph_value = infer::anchor_literal(&flat_graph, src_id);

        for (k, leaf) in source_leaves.iter().enumerate() {
            // Same row offsets the marker stack uses, so each ribbon meets
            // its marker exactly.
            let y_src = render::leaf_row_offset(k);
            // Which row at the target will accept this strand: the one that
            // admits it. Asked as subsumption rather than by kind because a
            // row may be a literal now — a `1` strand belongs on the `1`
            // row of a `1|2` anchor, not merely on some Integer row.
            let y_tgt =
                if let Some(idx) = target_leaves.iter().position(|l| infer::subsumes(l, leaf)) {
                    render::leaf_row_offset(idx)
                } else {
                    // No matching leaf at the target: aim at its first row.
                    0.0
                };
            // A leaf that claims no row of its own — a sum type, or `Pending` —
            // has no strand to draw.
            if edge::leaf_kind_of(leaf).is_none() {
                continue;
            }
            // A strand carrying a value is a hairline, a strand carrying a type
            // is a band. Literally the same question the anchor's markers ask,
            // asked through the same function and of the same leaf, so a strand
            // and the row it lands on cannot end up wearing different shapes.
            let (height, line_mode) =
                if render::leaf_is_drawn_as_line(leaf, src_graph_value.as_deref()) {
                    (edge::RIBBON_LINE_HEIGHT, 1.0)
                } else {
                    (edge::RIBBON_HEIGHT, 0.0)
                };
            let mesh =
                edge::build_ribbon_mesh(&curve, from_world.y + y_src, to_world.y + y_tgt, height);
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(materials_edge.add(edge::EdgeMaterial {
                    band_color: render::type_marker_color(leaf).to_linear(),
                    time: 0.0,
                    // Both ends of an ordinary edge wear the same shape, so
                    // there is nothing for the interpolation to do and
                    // `arc_total` only has to be non-zero.
                    line_mode_start: line_mode,
                    line_half_thickness: edge::RIBBON_LINE_HALF_THICKNESS_UV,
                    line_mode_end: line_mode,
                    arc_total: 1.0,
                    dash_period: 0.0,
                    dash_duty: 0.0,
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
        &anchor_world_positions,
        &declared_band_positions,
    );
    spawn_cast_links(
        &mut commands,
        &mut meshes,
        &mut materials_edge,
        &state,
        &flat_graph,
        &anchor_world_positions,
        &declared_band_positions,
    );

    // Which volume the caret is in, and the Pattern→Match table the boundary
    // count needs. Both are fixed for the whole pass: every surface below is
    // faded by how far its own volume sits from this one.
    //
    // Baked in at spawn rather than followed per frame, because a caret move
    // already rebuilds the scene — the caret's own mesh is built from
    // `pick.selected_pos` a few lines down, and would not move otherwise.
    let owning = owning_matches(state.root_graph());
    let caret_volume = state.volume_of_caret(&pick, &owning);
    // Everything at full strength while the caret is outside every volume:
    // there is nothing to measure a distance from, and measuring anyway would
    // be a fiction.
    let fade_of = |path: &[model::node::Id]| match &caret_volume {
        Some(caret) => grid::volume_fade(volume_boundaries(path, caret)),
        None => 1.0,
    };

    for walked_graph in state.root_graph().walk_all_graphs() {
        let Some(bounds) = walked_graph.layout_graph.grid_bounds() else {
            continue;
        };
        let offset = walked_graph.extra_offset;
        let scope_path = scope_volume_path(&walked_graph.context, &owning);
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

        spawn_volume_surfaces(
            &mut commands,
            &mut meshes,
            &mut materials_grid,
            bounds.min,
            bounds.max,
            offset,
            fade_of(&scope_path),
            Some(InteractiveFloor {
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
            }),
        );

        // Every Match in this scope is a volume in its own right: its own cells
        // and every arm hanging off them, which is what `match_footprint`
        // measures. It owns no `LayoutGraph`, so no walk reaches it — it has to
        // be asked for here, from the scope that holds it.
        for (id, node) in &walked_graph.layout_graph.graph.nodes {
            if !matches!(node, model::node::ENode::Match { .. }) {
                continue;
            }
            let Some(fp) = walked_graph.layout_graph.match_footprint(id) else {
                continue;
            };
            let mut match_path = scope_path.clone();
            match_path.push(id.clone());
            spawn_volume_surfaces(
                &mut commands,
                &mut meshes,
                &mut materials_grid,
                fp.min,
                fp.max,
                offset,
                fade_of(&match_path),
                // No `InteractiveFloor`: a Match is not a scope. Nothing
                // addresses a cell *in* it, and its footprint is already
                // flattened on the floor of the scope it stands in.
                None,
            );
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

/// A labelled checkbox pinned to a screen corner — the last one in the editor,
/// and a setting rather than a property of the graph, which is why it is not a
/// prompt. It stands still instead of being respawned, so it carries its own
/// label, makes the whole row clickable, and leaves the swatch to a sync
/// system.
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
        if remove_node_at_caret(&mut state, &pick, &selected_node_id) {
            rebuild.0 = true;
        }
    }
}

/// Take a node out of the scope the caret stands in. Returns whether the graph
/// changed, so the caller knows whether to flag a rebuild.
///
/// Caret-relative rather than by a global search, because both callers are:
/// the delete button acts on what the caret selects, and the rollback of an
/// unfinished node acts on the node the caret was put on to finish it.
fn remove_node_at_caret(
    state: &mut GraphState,
    pick: &PickState,
    node_id: &model::node::Id,
) -> bool {
    let Some((caret_graph, _)) = state.caret_graph(pick) else {
        return false;
    };
    let updated = caret_graph.minus_node(node_id);
    let Some((caret_graph_mut, _)) = state.caret_graph_mut(pick) else {
        return false;
    };
    *caret_graph_mut = updated;
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
        _ => graph.node_at(local).is_none() && local.z != 0,
    }
}

/// Create a node of this kind at the caret. `None` when nothing was built.
///
/// For most kinds the caret does not follow: it keeps addressing the cell,
/// which now holds the new node. The three kinds that come into the world
/// unfinished are the exception — there the caret moves onto the cell that
/// names the missing property, because that is where it has to be answered, and
/// the `PendingEdit` that comes back is what an Escape undoes.
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
    // the node was built on. All three sit in the scope it was built in, so no
    // descent into a branch is needed.
    //
    // Two of the steps are `2 * Z` for the same reason: both a TypeCast and a
    // Match hold a gap open after their input anchor, so what names the type is
    // the second cell behind it, not the first. A new Pattern goes one row
    // below the gap the caret was standing on, and one cell further back again
    // to reach its own type.
    let step = match kind {
        AddKind::TypeCast | AddKind::Match => IVec3::Z * 2,
        AddKind::Pattern => IVec3::Y + IVec3::Z,
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
    // Two shapes of the same field: a Constant and a Source always carry a
    // type, a TypeCast and a Pattern carry one only once it has been chosen.
    // Committing a row is exactly the act that makes the second into the first.
    match node {
        Some(model::node::ENode::Constant { r#type, .. })
        | Some(model::node::ENode::Source { r#type, .. }) => {
            *r#type = make_etype(choice, value);
            // Anchor heights follow declared types, so a type change reshapes
            // the node and its neighbours.
            state.resettle();
            true
        }
        Some(model::node::ENode::TypeCast { r#type, .. })
        | Some(model::node::ENode::Pattern { r#type, .. }) => {
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
                | EditTarget::PatternType,
            ),
        ) => set_node_type(state, &id, *choice, value.clone()),
        _ => false,
    };
    changed.then_some(Inserted::Done)
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
                outcome,
                &option.0,
                &mut prompt,
                &mut pending,
                &mut mode,
                &mut rebuild,
            );
        }
    }
}

/// What follows a committed row, whichever key or click committed it.
///
/// INSERT stays on only while there is something left to say at the caret. A
/// commit does not move the caret, so what it lands on is the node that was
/// just built — and for most kinds that node is already finished by the row
/// that built it.
fn commit_outcome(
    outcome: Inserted,
    action: &PromptAction,
    prompt: &mut InsertPrompt,
    pending: &mut PendingNode,
    mode: &mut EditorMode,
    rebuild: &mut NeedsRebuild,
) {
    prompt.clear();
    rebuild.0 = true;
    let stay = match (&outcome, action) {
        // Its one mandatory property is still missing, and the caret has been
        // put on the cell that names it.
        (Inserted::Pending(_), _) => true,
        // A Source arrives nameless, and the caret is left standing on the body
        // its name is written along — so the keystrokes after it are the name.
        (Inserted::Done, PromptAction::Create(AddKind::Source)) => true,
        // Everything else is complete the moment it appears. A Constant *is*
        // its literal and a call *is* its function, and both were typed to
        // reach the row that built them — so the cell the caret is left on
        // holds the answer that was just given, and staying would do nothing
        // but offer to give it again. Changing a property is finished for the
        // same reason.
        _ => false,
    };
    // Answering a property is what finishes a node, so any commit clears the
    // mark — and a new one only ever comes from a `Create`.
    pending.0 = match outcome {
        Inserted::Pending(edit) => Some(edit),
        Inserted::Done => None,
    };
    if !stay {
        *mode = EditorMode::Normal;
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
        // one. No literal: what a Source declares is a type, and the value
        // arrives from outside.
        InsertTarget::Edit(_, EditTarget::SourceType) => type_rows(text),
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

#[derive(Default, PartialEq, Eq, Clone)]
struct InsertPromptFingerprint {
    visible: bool,
    text: String,
    cursor: usize,
    selected: Option<usize>,
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
        // Only where INSERT means "build something". Changing a property that
        // already stands there happens in the node editor's panel, where the
        // property is named, and an empty create prompt standing beside it
        // would claim a choice that is not on offer.
        && insert_target(&state, &pick) == InsertTarget::Create
        && !start_menu.showing
        && !modal_is_open(&eval)
        && !is_evaluating(&eval);
    let candidates = if visible {
        prompt_candidates(&state, &pick, &prompt.text)
    } else {
        Vec::new()
    };
    let selected = clamped_selection(&candidates, prompt.selected);

    let fp = InsertPromptFingerprint {
        visible,
        text: prompt.text.clone(),
        cursor: prompt.cursor,
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
    commands.entity(panel_entity).with_children(|panel| {
        spawn_prompt_body(
            panel,
            font,
            &prompt.text,
            prompt.cursor,
            &candidates,
            selected,
            InsertPromptEntity,
        );
    });
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
            BackgroundColor(if highlighted {
                Color::srgba(0.2, 0.2, 0.3, 0.95)
            } else {
                Color::srgba(0.0, 0.0, 0.0, 0.0)
            }),
        ));
        // A greyed row is not clickable at all, for the same reason the "… N
        // more" tally is not: what cannot be committed must not be committable
        // by another route. `Enter` filters on `allowed`, but the click path
        // commits whatever the row carries — and only the *create* actions are
        // checked a second time, inside `insert_node_kind`.
        if suggestion.allowed {
            entity.insert((Button, InsertPromptOption(suggestion.action.clone())));
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
    /// The node and the one property the addressed cell names. Moving between
    /// two cells of the *same* node changes the property, which is why this is
    /// not keyed on the node alone.
    target: Option<(model::node::Id, EditTarget)>,
    /// What that property says right now. The panel echoes it outside INSERT,
    /// and a commit can change it under a standing panel.
    value: String,
    /// The prompt's text, cursor and highlight, but only while the panel is the
    /// one drawing it. That is also the bit that catches the NORMAL→INSERT
    /// switch, where nothing else in here moves.
    prompt: Option<(String, usize, Option<usize>)>,
    candidates: Vec<Suggestion>,
    visible: bool,
}

/// What the panel calls the node, and what it calls the one property the
/// addressed cell names. A total function of the target, so the panel needs no
/// second look at the node to write its heading.
fn target_labels(target: &EditTarget) -> (&'static str, &'static str) {
    match target {
        EditTarget::SourceName => ("Source", "Name"),
        EditTarget::SourceType => ("Source", "Type"),
        EditTarget::ConstantValue => ("Constant", "Value"),
        EditTarget::CastType => ("TypeCast", "Type"),
        EditTarget::PatternType => ("Pattern", "Type"),
    }
}

/// The panel beside the caret: the kind of node the caret stands on, and the
/// one property its cell names — as a prompt while INSERT is typing into it,
/// and as plain text otherwise.
///
/// One row, never two. A cell names one property, so there is never a second
/// one to show; the panel that used to offer a node's type and its value at the
/// same time is what the cell layout exists to replace.
fn sync_node_editor_ui(
    mut commands: Commands,
    state: Res<GraphState>,
    pick: Res<PickState>,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    mode: Res<EditorMode>,
    prompt: Res<InsertPrompt>,
    ui_font: Res<UiFont>,
    mut panel_q: Query<(Entity, &mut Node), With<NodeEditorPanel>>,
    editor_children_q: Query<Entity, With<NodeEditorEntity>>,
    mut cache: Local<NodeEditorFingerprint>,
) {
    let target = match insert_target(&state, &pick) {
        InsertTarget::Edit(id, edit) => Some((id, edit)),
        // A cell that names no property has no panel. What INSERT does there is
        // build, and the create prompt draws itself in its own place.
        InsertTarget::Create => None,
    };
    let visible = target.is_some() && !start_menu.showing && !is_evaluating(&eval);
    let editing = visible && *mode == EditorMode::Insert;
    let candidates = if editing {
        prompt_candidates(&state, &pick, &prompt.text)
    } else {
        Vec::new()
    };
    let selected = clamped_selection(&candidates, prompt.selected);
    let value = target
        .as_ref()
        .map(|(id, edit)| property_text(&state, id, edit))
        .unwrap_or_default();

    let fp = NodeEditorFingerprint {
        target: target.clone(),
        value: value.clone(),
        prompt: editing.then(|| (prompt.text.clone(), prompt.cursor, selected)),
        candidates: candidates.clone(),
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
    let Some((_, edit)) = target else {
        return;
    };
    if !visible {
        return;
    }

    let (kind_label, property_label) = target_labels(&edit);
    let font = &ui_font.0;
    commands.entity(panel_entity).with_children(|panel| {
        spawn_editor_label(panel, font, kind_label);
        spawn_labeled_row(panel, font, property_label, |slot| {
            if editing {
                spawn_prompt_body(
                    slot,
                    font,
                    &prompt.text,
                    prompt.cursor,
                    &candidates,
                    selected,
                    NodeEditorEntity,
                );
            } else {
                slot.spawn((
                    Text::new(value.clone()),
                    text_font(font, 14.0),
                    TextColor(Color::srgb(0.91, 0.89, 0.87)),
                    NodeEditorEntity,
                ));
            }
        });
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
/// Shared by the prompt that creates nodes and the one that changes a property
/// — they are the same widget, and only the container and its marker differ,
/// because the two are despawned by two different syncs and a row tagged for
/// the other one would be swept away by a rebuild its sync never hears about.
fn spawn_prompt_body(
    parent: &mut ChildSpawnerCommands,
    font: &Handle<Font>,
    text: &str,
    cursor: usize,
    candidates: &[Suggestion],
    selected: Option<usize>,
    marker: impl Bundle + Clone,
) {
    parent
        .spawn((
            Node {
                flex_direction: FlexDirection::Column,
                flex_grow: 1.0,
                min_width: Val::Px(0.0),
                row_gap: Val::Px(2.0),
                ..default()
            },
            marker.clone(),
        ))
        .with_children(|column| {
            column
                .spawn((
                    editor_text_input_node(),
                    BackgroundColor(Color::srgba(0.06, 0.06, 0.12, 0.95)),
                    BorderColor::all(Color::srgb(0.133, 0.827, 0.933)),
                    marker.clone(),
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
                    marker,
                ))
                .with_children(|options| {
                    spawn_prompt_rows(options, font, candidates, selected);
                });
        });
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

/// What a scope's floor carries beyond the plain surface: the component that
/// makes it the mouse's pick target, the border rect marking the caret's own
/// scope, and the footprint rects that flatten multi-cell nodes.
///
/// A Match's floor carries none of it, which is why this is an `Option` at the
/// spawner rather than four more parameters. A Match is not a scope — nothing
/// addresses a cell *in* one, and its own footprint is already flattened on the
/// floor of the scope it stands in.
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
/// Four for every volume there is — the program's scope, every Match, every
/// branch of every Match — so that a volume is recognisable as one whatever kind
/// it happens to be. What tells them apart is `fade`, which says how many volume
/// walls stand between this one and the one the caret is in.
///
/// `min`/`max` are the volume's inclusive cell bounds in coordinates `offset`
/// carries to global: `grid_bounds()` for a scope, `match_footprint()` for a
/// Match. Cells are corner-anchored, so every far edge is `max + 1`.
#[allow(clippy::too_many_arguments)]
fn spawn_volume_surfaces(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials_grid: &mut Assets<grid::GridMaterial>,
    min: IVec3,
    max: IVec3,
    offset: Vec3,
    fade: f32,
    floor: Option<InteractiveFloor>,
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

    // ── The floor: the lower bounding edge of the volume's last row, so what
    // stands in it stands *on* it rather than hanging under it.
    let floor_center = centre(mid_x, (max.y + 1) as f32, mid_z);
    let mut floor_entity = commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(size_x, size_z).build())),
        Transform::from_translation(floor_center),
        SceneEntity,
    ));
    match floor {
        Some(f) => {
            floor_entity.insert((
                MeshMaterial3d(materials_grid.add(grid::GridMaterial {
                    border_min: f.border_min,
                    border_max: f.border_max,
                    footprint_count: f.footprint_count,
                    footprints: f.footprints,
                    ..grid::GridMaterial::scope_surface(Vec3::X, Vec3::Z, fade)
                })),
                f.scope,
            ));
        }
        None => {
            floor_entity.insert(MeshMaterial3d(
                materials_grid.add(grid::GridMaterial::scope_surface(Vec3::X, Vec3::Z, fade)),
            ));
        }
    }

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
            ))),
            Transform::from_xyz(face_center.x, face_center.y, face_z),
            SceneEntity,
        ));
    }
}

/// Spawn one structural link: a ribbon from `from` to `to` whose two ends may
/// wear different shapes.
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
    leaf: &infer::EType,
    from: Vec3,
    to: Vec3,
    start: edge::RibbonEnd,
    end: edge::RibbonEnd,
) {
    // A leaf that claims no row of its own — a sum type, or `Pending` — has no
    // strand to draw.
    if edge::leaf_kind_of(leaf).is_none() {
        return;
    }
    let curve = edge::EdgeCurve::from_endpoints(from, to);
    let (mesh, arc_total) = edge::build_tapered_ribbon_mesh(&curve, &start, &end);
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials_edge.add(edge::EdgeMaterial {
            band_color: render::type_marker_color(leaf).to_linear(),
            time: 0.0,
            line_mode_start: start.line_mode,
            line_half_thickness: edge::RIBBON_LINE_HALF_THICKNESS_UV,
            line_mode_end: end.line_mode,
            arc_total,
            dash_period: 0.0,
            dash_duty: 0.0,
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
) {
    let curve = edge::EdgeCurve::from_endpoints(from, to);
    let (mesh, arc_total) = edge::build_tapered_ribbon_mesh(&curve, &start, &end);
    let mut spawned = commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials_edge.add(edge::EdgeMaterial {
            // Through the same call the coloured strands go through, of the
            // same type the grey anchor bodies are drawn from, so a band and
            // the two cells it joins cannot come apart.
            band_color: render::type_marker_color(&infer::EType::Pending).to_linear(),
            time: 0.0,
            line_mode_start: start.line_mode,
            line_half_thickness: edge::RIBBON_LINE_HALF_THICKNESS_UV,
            line_mode_end: end.line_mode,
            arc_total,
            dash_period: edge::RIBBON_DASH_PERIOD,
            dash_duty: edge::RIBBON_DASH_DUTY,
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
/// exactly one, a full `TYPE_MARKER_Y_STEP` tall — so a pending band meets it
/// by claiming all of it.
fn whole_row_end(row_center_y: f32, as_line: bool) -> edge::RibbonEnd {
    edge::ribbon_end(row_center_y, &infer::RowSpan::FULL, as_line)
}

/// Fan an anchor's rows out onto one declared-type cell: every row that cell
/// can describe gets a strand, claiming the share of the row it takes.
///
/// Two kinds own such a cell — a Match's arm and a TypeCast's target — and they
/// make the same statement with it, so they are drawn by the same code. The
/// caller supplies the anchor side once and calls this per cell.
///
/// The two ways a row can end up with no coloured strand are not the same thing
/// and are not drawn the same way.
///
/// A row the cell **can never describe** gets nothing. Note the deliberate
/// divergence from the edge pass, which aims an unmatched leaf at row 0:
/// docking a `Bool` arm onto an `Integer` band would draw the lie that it
/// consumes it. Here the gap *is* the statement — a Match reads as
/// non-exhaustive, a cast as able to fail.
///
/// A cell that has **not said yet** what it describes — `declared` is `None`,
/// an arm or a cast target the user has not typed — makes no such statement,
/// and neither does a row that is not there because what arrives is still
/// `Pending`. Those get a pending band: the cell is wired to the anchor either
/// way, and only what travels between them is open.
#[allow(clippy::too_many_arguments)]
fn spawn_declared_cell_links(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials_edge: &mut Assets<edge::EdgeMaterial>,
    in_pos: Vec3,
    in_leaves: &[infer::EType],
    in_value: Option<&str>,
    band_pos: Vec3,
    declared: Option<&model::r#type::EType>,
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
    // `max(1)` rather than the leaf count alone: an anchor whose type is
    // undecided has no leaf rows, but it still has the one row its grey
    // `plain_anchor_body` is drawn on, and that is where its band leaves from.
    for row in 0..in_leaves.len().max(1) {
        let anchor_leaf = in_leaves.get(row);
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
            from,
            band_pos,
            edge::ribbon_end(row_y, &span, row_is_line),
            whole_row_end(band_pos.y, cell_is_line),
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
fn spawn_match_links(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials_edge: &mut Assets<edge::EdgeMaterial>,
    state: &GraphState,
    flat_graph: &model::term_graph::TermGraph,
    anchor_world_positions: &std::collections::HashMap<model::anchor::Id, Vec3>,
    declared_band_positions: &std::collections::HashMap<model::node::Id, Vec3>,
) {
    let decls = &state.function_declarations;
    // `walk_all` reaches every scope with its offset already composed, and a
    // Match's Patterns always live in the same LayoutGraph the Match does
    // (`plus_match`), so a nested Match needs nothing path-aware here — it is
    // reached like any other and resolves its arms by lookup.
    for walked in state.layout_graph.walk_all() {
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
        // the rows a strand aims at are the rows that are actually there.
        let in_leaves = infer::incoming_anchor_type(flat_graph, input_anchor, decls)
            .map(|t| render::ordered_supported_leaves(&t))
            .unwrap_or_default();
        let in_value = infer::incoming_anchor_literal(flat_graph, input_anchor);
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
                    &in_leaves,
                    in_value.as_deref(),
                    band_pos,
                    arm_type.as_ref(),
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
        let out_leaves = render::ordered_supported_leaves(
            &infer::anchor_type(flat_graph, output_anchor, decls).unwrap_or(infer::EType::Pending),
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
            let sink_value = infer::incoming_anchor_literal(flat_graph, sink_input);
            let sink_leaves = infer::incoming_anchor_type(flat_graph, sink_input, decls)
                .map(|t| render::ordered_supported_leaves(&t))
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
            if sink_leaves.is_empty() {
                spawn_pending_ribbon(
                    commands,
                    meshes,
                    materials_edge,
                    from,
                    to,
                    whole_row_end(sink_pos.y, false),
                    whole_row_end(out_pos.y, false),
                    None,
                );
                continue;
            }

            for (k, leaf) in sink_leaves.iter().enumerate() {
                // The output has not decided what it is yet, so there is no row
                // to pick and the strand aims at the anchor's own — the same
                // fallback the edge pass uses for a leaf its target has no row
                // for. It carries the branch's own colour all the same: what
                // *this* branch produces is settled even while the union of all
                // of them is not.
                if out_leaves.is_empty() {
                    let as_line = render::leaf_is_drawn_as_line(leaf, sink_value.as_deref());
                    spawn_link_ribbon(
                        commands,
                        meshes,
                        materials_edge,
                        leaf,
                        from,
                        to,
                        whole_row_end(sink_pos.y + render::leaf_row_offset(k), as_line),
                        // The same shape at both ends: there is nothing at the
                        // output end that could ask for a different one.
                        whole_row_end(out_pos.y, as_line),
                    );
                    continue;
                }
                for (row, out_leaf) in out_leaves.iter().enumerate() {
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
                        sink_pos.y + render::leaf_row_offset(k),
                        &span,
                        render::leaf_is_drawn_as_line(leaf, sink_value.as_deref()),
                    );
                    // The Match output is drawn from its type alone — deriving
                    // an extra literal for it would be constant folding — so it
                    // is a line only where its own type says so.
                    let end = edge::ribbon_end(
                        out_pos.y + render::leaf_row_offset(row),
                        &span,
                        render::leaf_is_drawn_as_line(out_leaf, None),
                    );
                    spawn_link_ribbon(commands, meshes, materials_edge, leaf, from, to, start, end);
                }
            }
        }
    }
}

/// Draw the connections a TypeCast is made of: what arrives at its input
/// reaching its target cell, and what the target cannot take reaching the
/// `none` its output carries.
///
/// A cast borrows the Match's rhythm with one arm, so the first half is
/// literally the Match's — `spawn_declared_cell_links`, called once.
///
/// The second half is the cast's own, and it is the thing a cast is *about*.
/// `infer::type_cast_output_type` calls a cast total exactly when its target
/// subsumes everything arriving, and hangs a `none` off it otherwise. Until now
/// that `none` appeared at the output with nothing to say where it came from.
/// It comes from whatever the target does not claim: `RowSpan::complement` of
/// each row's claimed share, drawn as a strand that runs past the target cell
/// straight to the `none` row. Past it, not through it — a cast that fails
/// never reaches its target.
///
/// The two halves agree by construction: a total cast claims every row whole,
/// every complement is empty, and no strand is drawn — which is also exactly
/// when the output has no `none` row to draw one to.
///
/// The reverse leg a Match has, target cell → output anchor, is deliberately
/// absent. On success the target cell and the output say the same thing, one
/// cell apart; a strand between them would only spell it twice.
fn spawn_cast_links(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials_edge: &mut Assets<edge::EdgeMaterial>,
    state: &GraphState,
    flat_graph: &model::term_graph::TermGraph,
    anchor_world_positions: &std::collections::HashMap<model::anchor::Id, Vec3>,
    declared_band_positions: &std::collections::HashMap<model::node::Id, Vec3>,
) {
    let decls = &state.function_declarations;
    for walked in state.layout_graph.walk_all() {
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
        let in_leaves = infer::incoming_anchor_type(flat_graph, input_anchor, decls)
            .map(|t| render::ordered_supported_leaves(&t))
            .unwrap_or_default();
        let in_value = infer::incoming_anchor_literal(flat_graph, input_anchor);

        spawn_declared_cell_links(
            commands,
            meshes,
            materials_edge,
            in_pos,
            &in_leaves,
            in_value.as_deref(),
            band_pos,
            r#type.as_ref(),
        );

        // ── What the target cannot take, reaching the `none` ──
        //
        // A cast with no target chosen declares nothing, so nothing can fail
        // against it and its output is `Pending` either way. The leg above is
        // the whole of what such a cast is drawn as, and it is drawn pending.
        let Some(target) = r#type else {
            continue;
        };
        let out_leaves = render::ordered_supported_leaves(
            &infer::anchor_type(flat_graph, output_anchor, decls).unwrap_or(infer::EType::Pending),
        );
        let (Some(&out_pos), Some(none_row)) = (
            anchor_world_positions.get(output_anchor),
            out_leaves
                .iter()
                .position(|leaf| matches!(leaf, infer::EType::None)),
        ) else {
            continue;
        };
        let target_leaf = infer::graph_type_to_eval_type(target);
        for (row, anchor_leaf) in in_leaves.iter().enumerate() {
            // No claim at all means the whole row fails; a partial claim leaves
            // its complement. `None` from `complement` means nothing is left,
            // so this row always succeeds and needs no strand.
            let failing = match infer::claimed_span(anchor_leaf, &target_leaf) {
                Option::None => Some(infer::RowSpan::FULL),
                Some(claimed) => claimed.complement(),
            };
            let Some(failing) = failing else {
                continue;
            };
            let start = edge::ribbon_end(
                in_pos.y + render::leaf_row_offset(row),
                &failing,
                render::leaf_is_drawn_as_line(anchor_leaf, in_value.as_deref()),
            );
            // `none` is always a line, so the strand tapers into one however
            // much of the band it left with.
            let end = edge::ribbon_end(
                out_pos.y + render::leaf_row_offset(none_row),
                &infer::RowSpan::FULL,
                true,
            );
            spawn_link_ribbon(
                commands,
                meshes,
                materials_edge,
                // Coloured as `none` rather than as the row it leaves: what
                // travels here is not a value of that type, it is the absence
                // of one, and the colour should say where the strand ends up.
                &infer::EType::None,
                render::anchor_body_face_world(in_pos, true),
                // Arrive at the output band's near face, in front of it.
                render::anchor_body_face_world(out_pos, false),
                start,
                end,
            );
        }
    }
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
/// Re-seeds when the mode or the target changes and at no other time. A
/// keystroke changes neither, so typing is safe from it; a mouse click can move
/// the caret under INSERT, which is exactly the case a one-shot seed at the `i`
/// keypress would miss.
///
/// It stands where the Source name field's focus projection used to. That field
/// is gone — every property is typed into the one prompt now — so what has to
/// be projected from the mode is no longer a focus but the text itself.
fn sync_prompt_seed(
    mode: Res<EditorMode>,
    state: Res<GraphState>,
    pick: Res<PickState>,
    mut prompt: ResMut<InsertPrompt>,
    mut pending: ResMut<PendingNode>,
    mut last: Local<Option<(EditorMode, InsertTarget)>>,
) {
    let target = insert_target(&state, &pick);
    let now = (*mode, target.clone());
    if last.as_ref() == Some(&now) {
        return;
    }
    *last = Some(now);

    // A node is only unfinished while the caret is still standing on it.
    // Walking away is not a cancel — the node keeps what it has — but it does
    // mean the next Escape is about something else, so the mark is dropped
    // rather than carried around waiting to unmake a node nobody is looking at.
    let pending_here = match (&pending.0, &target) {
        (Some(edit), InsertTarget::Edit(id, _)) => edit.node == *id,
        _ => false,
    };
    if !pending_here {
        pending.0 = None;
    }

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

/// True while something other than the graph owns the keyboard: a modal's text
/// field, a modal, the start menu, or a running evaluation. Mode switching and
/// the INSERT-mode inserts stay out of the way then — otherwise `i` would both
/// type an `i` and change the mode.
///
/// The editor itself never appears here any more: its one text field was the
/// Source's name, and that is typed into the prompt now, which is deliberately
/// not a `TextInput` precisely so it does not capture.
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
/// text field is gone, so whatever owns the keyboard is the start menu, a modal
/// or an evaluation, and none of them are a mode to leave.
fn handle_editor_keys(
    mut key_events: MessageReader<KeyboardInput>,
    keys: Res<ButtonInput<KeyCode>>,
    text_inputs: Query<&TextInput>,
    start_menu: Res<StartMenu>,
    eval: Res<EvalState>,
    mut mode: ResMut<EditorMode>,
    mut prompt: ResMut<InsertPrompt>,
    mut pending: ResMut<PendingNode>,
    mut state: ResMut<GraphState>,
    mut pick: ResMut<PickState>,
    mut rebuild: ResMut<NeedsRebuild>,
) {
    let captured = keyboard_captured(&text_inputs, &start_menu, &eval);
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
                // Escape leaves INSERT outright and drops what was typed —
                // there is no half-written name worth keeping.
                //
                // And where what stands under the caret was built a moment ago
                // and is still waiting for its one mandatory property, dropping
                // what was typed means dropping the node: a cancelled insert
                // leaves no placeholder behind, and the caret goes back to the
                // cell it was standing on.
                if let Some(edit) = pending.0.take() {
                    if remove_node_at_caret(&mut state, &pick, &edit.node) {
                        pick.selected_pos = state.root_graph().clamp_to_volume(edit.caret_before);
                        prompt.clear();
                        *mode = EditorMode::Normal;
                        rebuild.0 = true;
                        // The rest of the batch was struck against a graph that
                        // no longer holds what those keys were about.
                        break;
                    }
                }
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
                                outcome,
                                &action,
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
    //
    // `keyboard_captured` covers the arrows here, not just the letters. It
    // used to gate only `hjkl`, and the arrows — which come from `ButtonInput`
    // rather than from the message — walked straight past it: typing in a
    // field, Left and Right moved the text cursor *and* the caret, and the
    // travelling caret pulled the panel out from under the typist.
    if is_evaluating(&eval)
        || *mode == EditorMode::Insert
        || keyboard_captured(&text_inputs, &start_menu, &eval)
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
        .add_plugins(depth_cue::DepthCuePlugin)
        .init_resource::<GraphState>()
        .init_resource::<NeedsRebuild>()
        .init_resource::<PickState>()
        .init_resource::<DragState>()
        .init_resource::<EvalState>()
        .init_resource::<StartMenu>()
        .init_resource::<EditorMode>()
        .init_resource::<InsertPrompt>()
        .init_resource::<PendingNode>()
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
                        .chain()
                        // A click resolves before the keys of the same frame.
                        // `handle_add_button` sets the mode and `pick_nodes`
                        // moves the caret, and the keyboard chain projects both
                        // onto the prompt's text at its end — ambiguous, that
                        // projection could run on the state the click was about
                        // to change.
                        .before(handle_editor_keys),
                    highlight_hovered,
                    update_selection_display,
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
            (sync_node_editor_ui, sync_insert_prompt_ui)
                .chain()
                // Both panels draw what was typed this frame, so they have to
                // run after the keys landed and after the seed that follows
                // them — otherwise a prompt shows the previous frame's text
                // every other frame.
                .after(sync_prompt_seed),
        )
        .run();
}
