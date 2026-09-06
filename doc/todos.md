# TODOs

- Tunnel
- new positional constraints of the nodes
- evaluation step wise -> narrowing to literal types? -> also the match branches? fade away/remove/collapse?
- correct relative consumption of band height (0% literals and 50% literals)





- make ordering of patterns determine the evaluation order -> which one is the default?
- evaluation values rendered at correct positions for match/pattern/patternsink
- typecast shall have none output type (sum) if type is not fully castable
- match shall have none in its output type (sum) if the patterns do not cover the incoming type



---

# Spec alignment — found while writing thesis chapter 4 (2026-09-06)

Every item below is a **divergence between the implementation and the design
fixed in the thesis** (`/srv/work/git/bsc-thesis`, chapter 3 + appendix), not a
feature request. The thesis is the specification; where the two disagree, the
code moves.

Order is by urgency: (D) and (E) must be done before any screenshot is taken,
because both are visible in the UI. (A), the complete function set, (B), the
vocabulary rename, and (C), the payload-free None, are done.

## D. Camera: bring it in line with 3.3.2

Today: orbit camera, **perspective** (`Camera3d::default()`), free-floating
target, **auto-rotating** (`auto_speed: 0.15`), tweened on selection change
(`CAMERA_TWEEN_DURATION = 0.4`).

The design (3.3.2 *Camera and viewpoint (D2)*) requires **two modes**:

**1. Bound mode (the default).**
- The camera is bound to the **caret**: it frames the caret and the cells around
  it and follows when the caret moves. It follows even when the caret was
  relocated by a mouse click — the caret jumps, the camera never does.
- **Default orientation** (`render.rs` `LAYOUT_SCALE` already encodes it):
  oblique, from above and from one side — evaluation order (Z) runs **left to
  right across the screen**, conjunction (X) is the **depth** axis with greater
  X **nearer the viewer**, disjunction (Y) **descends**.
- **Projection must be near-orthographic**, not perspective. Distant cells may
  be drawn smaller, but not so much smaller or so distorted that the plane the
  caret is standing on stops being readable. That legibility of the caret's own
  plane is the contract; the exact projection is free.
- **Scale is a user setting, not a consequence of the window.** Resizing the
  window changes *how much* is visible and never *how large* things are drawn —
  same as a font size in a text editor. This is what keeps the legibility
  guarantee from depending on the frame size. `fit_canvas_to_parent: true` is
  fine; what must not happen is the drawn cell size changing with it.
- **Nothing changes abruptly.** Not only position: framing, orientation, zoom,
  and also fading something out, bringing it back, narrowing or widening what is
  shown — all are transitions, never cuts, so the viewer can see what left and
  what arrived (object constancy). Duration: a few hundred ms, under ~1 s. The
  existing 0.4 s tween is the right value; reuse it everywhere.

**2. Free mode.**
- Camera can be moved, turned and placed anywhere; **any projection allowed**,
  perspective included. The guarantees above are deliberately suspended.
- **Leaving the bound mode has to be an explicit act** — a mode switch, *not*
  something that happens by dragging. Today dragging silently detaches the view;
  that is the one behaviour that must change first.

**Open, decide before implementing: auto-rotation.** `auto_speed: 0.15` is not
in the design at all. Two defensible outcomes — (a) drop it, or (b) keep it as
the parallax/motion cue the background chapter names as the actual mechanism
that makes a 3D graph readable, in which case 3.3.2 has to gain a sentence.
**Do not just leave it in silently**; ask before deciding.

## E. Editing must happen at the caret, not in a property panel

3.3.3 states flatly: *"there is no property panel, no dialogue and no second
place to look — what can be changed is determined by where the caret stands, so
going there and selecting what to change are one act rather than two."*

The prototype has a `NodeEditorPanel` with dropdowns (`DropdownKind`,
`TypeChoice`, `spawn_type_dropdown`, `spawn_function_dropdown`,
`spawn_value_widget`, checkboxes). That is exactly the construction the design
rules out, and it is prominent in every screenshot.

**Target behaviour:**

- **One cell carries at most one editable value.** `CellRole` already models
  this; the panel is what bypasses it. Editing acts on `role_at(caret)`.
- **What is editable where** (3.3.3, *What is editable where*):

  | node kind | cell | what is edited there |
  |---|---|---|
  | `Constant` | body cell | its value (value = its own type, so one entry settles both) |
  | `TypeCast` | the band cell between its two anchors | the target type |
  | `Match` | the cell of each pattern's band, one per level | that pattern |
  | `FunctionCall` | **every** cell of the body | which function is called — the name is written along the body and its length *is* the body's extent, so editing it lengthens/shortens the node |
  | `Source` | first of its two Z cells | its declared type |
  | `Source` | second cell (the output anchor) | its name |
  | `Source` | — | its **index** is not typed at all: the index is the lateral order, so it is changed by **moving** the node |
  | `Tunnel` | same two cells, one volume down | declared type, then name |
  | `Sink`, `BranchSource` | — | nothing: they have no static parameter |

- **Values are entered as text and cannot be left invalid.** The editor refuses
  text that is not a value of the kind the cell holds; what stands there after
  an edit is always a well-formed static parameter. Note the two cases differ:
  the **function set is closed** (a list will do), while the **set of types is
  not finite** (sum types) — so what is typed there is an expression in a tiny
  type grammar, and what is guaranteed is only that a well-formed type results,
  not that it was picked from a list.
- **Creating is the same act as changing.** At an empty cell, in the editing
  mode, what is typed next decides what appears — and there are exactly three
  cases, the three a text document has:
  - **space** -> an empty cell, caret advances along the row;
  - **return** -> next row along the conjunction axis; **with the disjunction
    modifier**, it begins a *level* instead;
  - **a letter** -> it can be neither of the other two, so what is being written
    is a node kind's name; the edit closes on a valid one or does not close.
  The caret stays in the editing mode and advances by itself, so building a row
  *is* typing. (`handle_insert_keys` already implements the space/return/
  shift-return half — what is missing is the letter case.)
- **Two sets of keys, one model**: every movement available both on the arrow
  keys and on the modal-editor keys, with a modifier for the third axis. Nothing
  reachable by one set that is not reachable by the other. Proposed and not yet
  fixed: `h/j/k/l` stay **geometric** (X lateral, Y vertical — `j` down matches
  the descending Y), and **Z gets no fourth arrow pair**: it is primarily
  *structural* movement (follow an edge to producer/consumer), closer in
  character to `w`/`b` or `Ctrl-i`/`Ctrl-o`.
- **The mouse stays**, but only as a shortcut to a position: click places the
  caret, hover reports what is under the pointer without moving it. Everything
  must be reachable without it.

## F. Not a bug — keep as is, it is a finding

**`Pattern` as a node** (and the `Program`/root wrapper) are implementation
objects that the abstract syntax does not have: there, a pattern is a static
parameter of the `Match`. **This is correct and stays.** The reason is written
up in thesis ch. 4: a pattern is drawn as a labelled band, occupies cells and
must be addressable by the caret — and *what has to be addressable needs an
entry in the layout*. The implementation therefore needs more objects than the
language has node kinds. Do not "fix" this by folding patterns back into the
`Match`.

**The cycle guard in `infer.rs`** (`visiting` set -> `Pending`) also stays. The
design forbids cycles as a structural invariant the editor refuses to break, so
a cycle should never reach the checker — the guard is a cheap safety net against
*implementation* bugs that would otherwise freeze the program, not a hedge
against the design. Keep the comment saying so.
