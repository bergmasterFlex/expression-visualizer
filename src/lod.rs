//! Level of detail by opacity, anchored at the scope the caret stands in.
//!
//! A program nested more than two or three deep shows everything at once and
//! at full strength, which is exactly when it stops saying where the work is
//! happening. This grades it: outward, a volume's contents fade off in steps
//! until they are gone; inward, a volume's walls fill in until the sub-graph is
//! a grey box that cannot be looked into.
//!
//! Two numbers describe where a scope stands relative to the caret's, and the
//! whole module is built on them. `out + into` is the wall
//! count [`volume_boundaries`] has always measured; `into` alone is how deep
//! inside the caret's line of sight the scope sits.
//!
//! Everything here is read at rebuild time and baked into the materials. A
//! caret move already rebuilds the scene, so there is nothing to follow per
//! frame.

use bevy::prelude::*;

/// Whether the grading is in force. Off means every volume is drawn whole, with
/// no fade of any kind — which is what the `Clipping` checkbox turns back on.
#[derive(Resource)]
pub struct Clipping(pub bool);

impl Default for Clipping {
    fn default() -> Self {
        Self(true)
    }
}

/// What survives of a volume's contents, by the number of walls between it and
/// the volume the caret is in.
///
/// The first two steps are both whole: the caret's own volume and the one just
/// outside it are where the work is, and grading the immediate neighbourhood
/// would only make the thing being edited harder to read. Past the end of the
/// table nothing is left — and nothing left means nothing spawned, labels
/// included.
pub const CONTENT_RAMP: [f32; 5] = [1.0, 1.0, 0.75, 0.5, 0.25];

/// How solid a volume's shell stands, by how many walls deep inside the caret's
/// own it sits.
///
/// The mirror of `CONTENT_RAMP` and deliberately the same shape: on a straight
/// descent the two meet, so the step where the contents reach nothing is the
/// step where the box closes over them. One replaces the other rather than both
/// vanishing at once.
pub const SHELL_RAMP: [f32; 6] = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0];

/// The depth at which a volume is nothing but its box. Past it there is no
/// point building anything at all, box included — it would stand inside a
/// closed one.
pub const SOLID_DEPTH: usize = SHELL_RAMP.len() - 1;

/// The colour a sealed volume reads as: a node-sized grey body, the same
/// substance an unresolved anchor is drawn in, saying only that something is
/// there.
pub const SHELL_COLOR: Srgba = Srgba::new(0.30, 0.30, 0.34, 1.0);

/// Volume boundaries between two volumes: the walls crossed going from one to
/// the other through their nearest common ancestor.
///
/// Both paths are scope `context`s, which is all a volume path is — a Match used
/// to be a rung between a scope and its arms, and is not any more: it is drawn
/// as the node it is, so there is no wall there to cross.
///
/// Siblings come out two apart, not one — there is a wall out of the first and a
/// wall into the second, and no shortcut between them. Going out to the scope
/// that holds their Match is one, because that is one wall and there is nothing
/// standing behind it.
pub fn volume_boundaries(a: &[crate::model::node::Id], b: &[crate::model::node::Id]) -> usize {
    let common = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    (a.len() - common) + (b.len() - common)
}

/// What a rebuild pass reads to grade one scope against another. Fixed for the
/// whole pass: the caret cannot move inside one.
pub struct Lod {
    /// Owner path of the scope the caret stands in. `None` when the caret is
    /// outside every volume — there is then nothing to measure a distance from,
    /// and measuring anyway would be a fiction.
    caret: Option<Vec<crate::model::node::Id>>,
    enabled: bool,
}

impl Lod {
    /// `caret` is `GraphState::scope_of_caret`'s path; `enabled` the checkbox.
    pub fn new(caret: Option<Vec<crate::model::node::Id>>, enabled: bool) -> Self {
        Self { caret, enabled }
    }

    /// `(out, into)` for a scope: the walls climbed out of the caret's volume to
    /// reach the nearest common ancestor, and the walls descended from there.
    /// `None` when there is nothing to measure against.
    fn split(&self, context: &[crate::model::node::Id]) -> Option<(usize, usize)> {
        if !self.enabled {
            return None;
        }
        let caret = self.caret.as_ref()?;
        let common = context
            .iter()
            .zip(caret)
            .take_while(|(x, y)| x == y)
            .count();
        Some((caret.len() - common, context.len() - common))
    }

    /// The factor every mesh, ribbon and label belonging to this scope is
    /// painted with.
    pub fn content(&self, context: &[crate::model::node::Id]) -> f32 {
        match self.split(context) {
            Some((out, into)) => CONTENT_RAMP.get(out + into).copied().unwrap_or(0.0),
            None => 1.0,
        }
    }

    /// The alpha of the box drawn around this scope. `0.0` for the caret's own
    /// volume and the one just inside it, `1.0` where the box is all that is
    /// left of it.
    pub fn shell(&self, context: &[crate::model::node::Id]) -> f32 {
        match self.split(context) {
            Some((_, into)) => SHELL_RAMP.get(into).copied().unwrap_or(1.0),
            None => 0.0,
        }
    }

    /// True when the box around this scope is closed. Nothing inside it is
    /// built — not because it would be invisible, but because a closed box is
    /// the statement, and a label floating over it would contradict it.
    pub fn sealed(&self, context: &[crate::model::node::Id]) -> bool {
        self.shell(context) >= 1.0
    }

    /// True when the scope is not reached at all: neither its contents nor its
    /// own box are built, and neither is anything below it.
    ///
    /// Two ways to get there. Inside a box that is already closed there is
    /// nothing to see and nothing to draw — that is the bound on how deep the
    /// walk goes at all. And a branch the ramp has taken to nothing with no box
    /// standing in for it is simply gone; where a box *does* stand, the scope
    /// stays reached, because the box is the whole point of it having gone.
    ///
    /// The caret's own ancestors are exempt — `into == 0` is the line the caret
    /// hangs from, and dropping one would drop the caret with it. Their contents
    /// still go by `content`, which may well be nothing.
    pub fn dropped(&self, context: &[crate::model::node::Id]) -> bool {
        match self.split(context) {
            Some((_, into)) if into > SOLID_DEPTH => true,
            Some((out, into)) => {
                into > 0
                    && CONTENT_RAMP.get(out + into).is_none()
                    && SHELL_RAMP.get(into).copied().unwrap_or(1.0) <= 0.0
            }
            None => false,
        }
    }

    /// True when nothing of this scope's own contents is drawn — dropped,
    /// sealed, or faded to nothing.
    pub fn hidden(&self, context: &[crate::model::node::Id]) -> bool {
        self.dropped(context) || self.sealed(context) || self.content(context) <= 0.0
    }
}

/// The same material seen through `opacity` more of the picture.
///
/// Opaque up to the moment it is not: a body at full strength keeps the alpha
/// mode it was built with, so it still writes depth and the depth cue still
/// sees it. Only a graded one is pushed into the blended path, and only those
/// spend an OIT slot.
pub fn faded(material: StandardMaterial, opacity: f32) -> StandardMaterial {
    if opacity >= 1.0 {
        return material;
    }
    let mut material = material;
    let alpha = material.base_color.alpha() * opacity;
    material.base_color = material.base_color.with_alpha(alpha);
    material.alpha_mode = AlphaMode::Blend;
    material
}

/// The same colour at `opacity` of its own alpha — what a label's text colour
/// is faded through.
pub fn faded_color(color: Color, opacity: f32) -> Color {
    color.with_alpha(color.alpha() * opacity)
}
