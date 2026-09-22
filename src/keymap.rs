//! What the keys mean, written down once.
//!
//! The editor is modal, but its ambiguity runs deeper than a mode: `Return`
//! opens a column, or commits the highlighted row, or draws an edge, and which
//! of the three it is depends on the cell the caret stands on, on whether
//! anything has been typed yet, and on whether a row of the suggestion list is
//! lit. `Space` is a room-maker, or a character, or a character only inside a
//! string that has not been closed. Nothing on screen used to say which.
//!
//! So the bindings live here as data, with the condition under which each one
//! is alive written beside it, and both the bar along the bottom edge and —
//! from `resolve` on — the dispatch itself read this one table. A key with no
//! row here is a key that cannot act, which is the only arrangement in which
//! the help and the behaviour cannot drift apart. They already had: `F9` has
//! toggled the depth cue since `depth_cue.rs` was written and has never
//! appeared in any list of the controls.
//!
//! **The order of the rows is semantic.** The three `Space` rows and the two
//! `Return` rows are told apart by guards that used to sit in `match`-arm
//! order inside `handle_editor_keys`, and `match` is first-wins. So this table
//! is first-wins too: `live` lists in source order and `resolve` will ask in
//! source order. Rows that overlap must therefore be written most-specific
//! first, and a row added in the middle changes the meaning of the rows below
//! it.
//!
//! Nothing in `main.rs` had to become `pub` for this. A private item at the
//! crate root is visible in the root *and every descendant of it*, so this
//! module can name `crate::EditorMode`, `crate::EditTarget`, `crate::GraphState`
//! and call `crate::insert_target` as they stand. And because `mod keymap` is
//! itself private, `Context` and `crate::EditorMode` have the same effective
//! visibility, so nothing here leaks a private type out of a public one.

// ── Where the keyboard is standing ──────────────────────────────────────────

/// Everything a `when` predicate is allowed to ask about the world.
///
/// Built once per frame from the same resources the dispatch reads, so that
/// the row the bar shows and the arm the key takes are answers to one question
/// and not two.
///
/// `shift` is deliberately *not* in here. Rows that differ by Shift are rows of
/// their own (`Shift+Return`, `Shift+j k`), the way the old hand-written list
/// already spelled them. Shift in the context would rebuild the bar on every
/// Shift press and make it flicker while the key is held down to move along Y,
/// which is exactly when the hand is least able to afford the distraction.
/// Shift enters as a parameter to `resolve` and nowhere else.
#[derive(Clone, PartialEq, Eq)]
pub struct Context {
    pub mode: crate::EditorMode,
    /// What kind of place the caret addresses.
    pub here: Where,
    /// `Alt` is down, so the six navigation keys read the wiring rather than
    /// walk the grid.
    pub aiming: bool,
    /// Nothing has been typed into the prompt yet, which is what leaves
    /// `Space` and `Return` free to still mean room.
    pub prompt_empty: bool,
    /// A row of the suggestion list is standing lit, so `Return` has something
    /// to commit.
    pub highlighted: bool,
    /// The prompt's cursor is inside a string that has not been closed.
    pub in_quote: bool,
    /// The pointer is dragging an edge out of an anchor.
    pub drafting: bool,
}

/// `InsertTarget` with the identities taken out.
///
/// No predicate has ever wanted to know *which* anchor or *which* node, only
/// what kind of place this is — and an id in here would drag a clone of the
/// graph's naming into a struct that is rebuilt every frame.
#[derive(Clone, PartialEq, Eq)]
pub enum Where {
    /// A cell that names nothing: the prompt offers node kinds to build.
    Create,
    /// An anchor, where the one thing to do is draw an edge.
    ///
    /// `output` is carried because it decides what `Alt` and `j`/`k` mean
    /// together: an output may feed many consumers and so there is a choice to
    /// make between them, while an input is fed by at most one and there is
    /// not.
    Connect { output: bool },
    /// A cell that stands for a property of the node already there.
    Edit(crate::EditTarget),
}

/// Ask the editor where it is standing.
///
/// `aiming` and `drafting` come from the caller rather than from a resource of
/// their own, because the two systems that own them read them from
/// `ButtonInput` and from `DraftState` respectively, and asking twice is how
/// the two answers start to differ.
pub fn context(
    mode: crate::EditorMode,
    state: &crate::GraphState,
    pick: &crate::PickState,
    prompt: &crate::InsertPrompt,
    draft: &crate::DraftState,
    aiming: bool,
) -> Context {
    let here = match crate::insert_target(state, pick) {
        crate::InsertTarget::Create => Where::Create,
        crate::InsertTarget::Edit(_, property) => Where::Edit(property),
        // The cheap question `open_caret_draft` already asks, and deliberately
        // not `flattened_graph`, which `handle_edge_hop_keys` builds for its
        // own reach: that walks every scope, and this runs every frame.
        crate::InsertTarget::Connect(anchor) => Where::Connect {
            output: state
                .root_graph()
                .try_layout_anchor(&anchor)
                .is_some_and(|layout_anchor| {
                    matches!(layout_anchor.anchor, crate::model::anchor::EAnchor::Output)
                }),
        },
    };
    Context {
        mode,
        here,
        aiming,
        prompt_empty: prompt.text.is_empty(),
        highlighted: prompt.selected.is_some(),
        in_quote: crate::in_open_quote(&prompt.text),
        drafting: draft.pointer_active(),
    }
}

// ── The table ───────────────────────────────────────────────────────────────

/// Which kind of thing a binding does, and with it the order the bar groups
/// them in. Groups are drawn in declaration order and separated by a rule, so
/// this enum is also the layout.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Group {
    Move,
    Wire,
    Build,
    Text,
    Mode,
    View,
    Pointer,
}

/// One key, what it does, and where it is alive.
///
/// A binding whose condition cannot be written down is a binding nobody can
/// find, so `when` is not optional and there is no row that is always shown
/// "for reference".
pub struct Binding {
    /// What is struck, spelled the way a keyboard spells it.
    pub keys: &'static str,
    /// One line, saying what the key *does* — not what it is.
    ///
    /// `&'static str` and not a format string, which rules out naming the
    /// property an `Edit` row acts on ("the source's name", "the cast"). That
    /// is deliberate: `place` names it in the heading, two words to the left,
    /// and a second place to say it is a second place to keep it true.
    pub says: &'static str,
    pub group: Group,
    pub when: fn(&Context) -> bool,
}

// The predicates, named rather than written as closures at each row. A table
// of thirty closures is a table nobody proof-reads.

fn normal(c: &Context) -> bool {
    c.mode == crate::EditorMode::Normal && !c.aiming
}

/// `Alt` is down and the caret is reading the wiring.
fn walking(c: &Context) -> bool {
    c.mode == crate::EditorMode::Normal && c.aiming
}

fn on_anchor(c: &Context) -> bool {
    matches!(c.here, Where::Connect { .. })
}

fn walking_input(c: &Context) -> bool {
    walking(c) && matches!(c.here, Where::Connect { output: false })
}

fn walking_output(c: &Context) -> bool {
    walking(c) && matches!(c.here, Where::Connect { output: true })
}

fn walking_anchor(c: &Context) -> bool {
    walking(c) && on_anchor(c)
}

fn walking_open(c: &Context) -> bool {
    walking(c) && !on_anchor(c)
}

fn insert(c: &Context) -> bool {
    c.mode == crate::EditorMode::Insert
}

/// INSERT on an anchor: the prompt offers nothing and the keys aim an edge.
fn insert_wire(c: &Context) -> bool {
    insert(c) && on_anchor(c)
}

/// INSERT where something can be built: an empty cell, or a hole in a node.
fn insert_build(c: &Context) -> bool {
    insert(c) && matches!(c.here, Where::Create)
}

/// INSERT on a cell that stands for a property already there.
fn insert_edit(c: &Context) -> bool {
    insert(c) && matches!(c.here, Where::Edit(_))
}

/// INSERT wherever the prompt is taking text — which is everywhere but an
/// anchor.
fn insert_text(c: &Context) -> bool {
    insert(c) && !on_anchor(c)
}

/// The room-makers: only on a create prompt, and only while nothing is typed.
fn making_room(c: &Context) -> bool {
    insert_build(c) && c.prompt_empty
}

fn committable(c: &Context) -> bool {
    insert_text(c) && c.highlighted
}

fn quoting(c: &Context) -> bool {
    insert_text(c) && !c.prompt_empty && c.in_quote
}

fn always(_: &Context) -> bool {
    true
}

/// The pointer's own rows, and only where they are not noise: while something
/// is being typed the hand is not on the mouse, and a row about dragging would
/// be taking the place of one about the key being struck.
fn pointing(c: &Context) -> bool {
    !c.drafting && c.mode == crate::EditorMode::Normal
}

fn dragging(c: &Context) -> bool {
    c.drafting
}

/// Everything the keyboard and the pointer mean, in the order the question is
/// asked. See the module doc: the order is load-bearing.
pub static BINDINGS: &[Binding] = &[
    // ── Walking the grid ──
    Binding {
        keys: "h l  \u{2190}\u{2192}",
        says: "along the evaluation",
        group: Group::Move,
        when: normal,
    },
    Binding {
        keys: "j k  \u{2191}\u{2193}",
        says: "to the next column",
        group: Group::Move,
        when: normal,
    },
    Binding {
        keys: "Shift+j k",
        says: "to the next row",
        group: Group::Move,
        when: normal,
    },
    Binding {
        keys: "Ctrl+h j k l",
        says: "shove the node",
        group: Group::Move,
        when: normal,
    },
    Binding {
        keys: "Tab",
        says: "walk the panel's rows",
        group: Group::Move,
        when: always,
    },
    // ── Walking the wiring ──
    Binding {
        keys: "Alt",
        says: "light the next hop",
        group: Group::Wire,
        when: normal,
    },
    Binding {
        keys: "Alt+h l",
        says: "across the edge, or through",
        group: Group::Wire,
        when: walking_anchor,
    },
    Binding {
        keys: "Alt+Shift+j k",
        says: "choose which edge",
        group: Group::Wire,
        when: walking_output,
    },
    Binding {
        keys: "Alt+j k",
        says: "between this call's inputs",
        group: Group::Wire,
        when: walking_input,
    },
    Binding {
        keys: "Alt+h j k l",
        says: "reach for the nearest anchor",
        group: Group::Wire,
        when: walking_open,
    },
    // ── Aiming an edge ──
    Binding {
        keys: "h j k l  \u{2190}\u{2192}\u{2191}\u{2193}",
        says: "aim the edge",
        group: Group::Wire,
        when: insert_wire,
    },
    Binding {
        keys: "Shift+j k",
        says: "aim it up and down",
        group: Group::Wire,
        when: insert_wire,
    },
    Binding {
        keys: "Return",
        says: "draw the edge",
        group: Group::Wire,
        when: insert_wire,
    },
    // ── Building ──
    Binding {
        keys: "i",
        says: "build, wire or edit here",
        group: Group::Mode,
        when: normal,
    },
    Binding {
        keys: "Delete",
        says: "take this node out",
        group: Group::Build,
        when: normal,
    },
    Binding {
        keys: "type",
        says: "name what to build",
        group: Group::Build,
        when: insert_build,
    },
    Binding {
        keys: "type",
        says: "write this cell's value",
        group: Group::Build,
        when: insert_edit,
    },
    // The room-makers stand above the commit row on purpose: while nothing is
    // typed there is nothing to commit, and `Return` is the editor's newline.
    Binding {
        keys: "Space",
        says: "open a cell behind",
        group: Group::Build,
        when: making_room,
    },
    Binding {
        keys: "Return",
        says: "open a column",
        group: Group::Build,
        when: making_room,
    },
    Binding {
        keys: "Shift+Return",
        says: "open a row",
        group: Group::Build,
        when: making_room,
    },
    Binding {
        keys: "Return",
        says: "build the lit row",
        group: Group::Build,
        when: committable,
    },
    // ── The prompt ──
    Binding {
        keys: "\u{2191}\u{2193}",
        says: "walk the list",
        group: Group::Text,
        when: insert_text,
    },
    // Above the plain space, because a quote is the narrower case: inside one,
    // a space is text even on a create prompt that would otherwise make room.
    Binding {
        keys: "Space",
        says: "a space, inside the string",
        group: Group::Text,
        when: quoting,
    },
    Binding {
        keys: "Space",
        says: "a space, nothing more",
        group: Group::Text,
        when: insert_edit,
    },
    Binding {
        keys: "\u{2190}\u{2192} Home End",
        says: "move the cursor",
        group: Group::Text,
        when: insert_text,
    },
    Binding {
        keys: "Backspace",
        says: "rub out \u{2014} Delete ahead",
        group: Group::Text,
        when: insert_text,
    },
    // ── Leaving ──
    Binding {
        keys: "Escape",
        says: "let the edge go, and leave INSERT",
        group: Group::Mode,
        when: insert_wire,
    },
    Binding {
        keys: "Escape",
        says: "leave INSERT",
        group: Group::Mode,
        when: insert_text,
    },
    // ── The view ──
    Binding {
        keys: "F9",
        says: "cycle the depth cue",
        group: Group::View,
        when: always,
    },
    // ── The pointer ──
    //
    // In the same table as everything else, and with conditions of its own, so
    // that retiring the Controls modal cost none of what it used to say. What
    // it did lose is the listing of all seven at once — which is the point: a
    // drag that is under way has two answers, and the other five are noise
    // while it is.
    Binding {
        keys: "Release",
        says: "draw it where it points",
        group: Group::Pointer,
        when: dragging,
    },
    Binding {
        keys: "Right click",
        says: "cancel \u{2014} as does a release at the start",
        group: Group::Pointer,
        when: dragging,
    },
    Binding {
        keys: "Click",
        says: "put the caret there, wheel picks deeper",
        group: Group::Pointer,
        when: pointing,
    },
    Binding {
        keys: "Drag an anchor",
        says: "draw an edge",
        group: Group::Pointer,
        when: pointing,
    },
    Binding {
        keys: "Ctrl+drag",
        says: "orbit \u{2014} wheel zooms, right pans",
        group: Group::Pointer,
        when: pointing,
    },
];

// ── Reading the table ───────────────────────────────────────────────────────

/// The bindings alive where the keyboard is standing, in the order the
/// question is asked.
pub fn live(context: &Context) -> impl Iterator<Item = &'static Binding> + '_ {
    BINDINGS
        .iter()
        .filter(move |binding| (binding.when)(context))
}

/// What to call this place, given that the mode is already named beside it.
///
/// Only the qualifier, never the mode itself: the pill to the left of this
/// says NORMAL or INSERT, and a heading that said it a second time would be
/// the editor talking over itself. Blank where the mode alone is the whole
/// answer.
pub fn place(context: &Context) -> &'static str {
    match (context.mode, context.aiming, &context.here) {
        (crate::EditorMode::Normal, true, _) => "walking the wiring",
        (crate::EditorMode::Normal, false, _) => "",
        (crate::EditorMode::Insert, _, Where::Create) => "build",
        (crate::EditorMode::Insert, _, Where::Connect { .. }) => "wire",
        (crate::EditorMode::Insert, _, Where::Edit(property)) => match property {
            crate::EditTarget::SourceName => "name the source",
            crate::EditTarget::SourceType => "declare the type",
            crate::EditTarget::ConstantValue => "spell the value",
            crate::EditTarget::CastType => "choose what to cast to",
            crate::EditTarget::PatternType => "choose what the arm matches",
            crate::EditTarget::TunnelType => "choose what the tunnel lets through",
        },
    }
}
