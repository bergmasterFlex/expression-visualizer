/// Evaluation-order 3D layout for AST nodes.
///
/// Y axis = evaluation order (bottom → first evaluated, top → produces result)
/// X axis = binary left/right splits
/// Z axis = ternary then/else branches (orthogonal to binary splits)
///
/// For arithmetic `a + b`: operands sit below, operator above.
/// For ternary `cond ? then : else`:
///   - condition below (evaluated first)
///   - ?: node in the middle
///   - then/else branches above (produce result), split along Z

use bevy::prelude::*;
use crate::ast::AstNode;

/// Direction an edge flows: parent → child downward, or parent → child upward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeDir {
    /// Child is below parent (e.g. operands of +, condition of ?:)
    Down,
    /// Child is above parent (e.g. then/else branches of ?:)
    Up,
}

/// A positioned node in the 3D layout.
#[derive(Debug, Clone)]
pub struct LayoutNode {
    pub id: usize,
    pub ast: AstNode,
    pub pos: Vec3,         // final world position (after centering)
    pub eval_step: f32,    // Y value (evaluation order)
}

/// An edge between two layout nodes.
#[derive(Debug, Clone)]
pub struct LayoutEdge {
    pub from_id: usize,
    pub to_id: usize,
    pub from_pos: Vec3,
    pub to_pos: Vec3,
    pub label: &'static str,
    pub dir: EdgeDir,
}

/// Intermediate layout tree used during position computation.
#[derive(Debug)]
struct LNode {
    id: usize,
    ast: AstNode,
    width: f32,       // horizontal span in X
    x: f32,           // assigned X position
    y: f32,           // assigned Y position (eval order)
    z: f32,           // Z position (ternary branching)
    min_y: f32,
    max_y: f32,
    children: Vec<LNode>,
    child_labels: Vec<&'static str>,
    edge_dirs: Vec<EdgeDir>,
}

const Z_SPREAD: f32 = 3.5;

/// Build the intermediate layout tree with evaluation-order Y.
fn build_layout(node: &AstNode, z_base: f32, counter: &mut usize) -> LNode {
    let id = *counter;
    *counter += 1;

    match node {
        AstNode::NumLiteral(_) | AstNode::BoolLiteral(_) => LNode {
            id,
            ast: node.clone(),
            width: 1.0,
            x: 0.0,
            y: 0.0,
            z: z_base,
            min_y: 0.0,
            max_y: 0.0,
            children: vec![],
            child_labels: vec![],
            edge_dirs: vec![],
        },

        AstNode::BinaryExpr { left, right, .. }
        | AstNode::ComparisonExpr { left, right, .. } => {
            let mut l = build_layout(left, z_base, counter);
            let mut r = build_layout(right, z_base, counter);
            let width = l.width + r.width;

            let child_height = f32::max(l.max_y - l.min_y, r.max_y - r.min_y);
            let parent_y = child_height + 1.0;

            // Shift children so their tops align at y=0..child_height
            let lminy = l.min_y;
            let rminy = r.min_y;
            shift_y(&mut l, -lminy);
            shift_y(&mut r, -rminy);

            LNode {
                id,
                ast: node.clone(),
                width,
                x: 0.0,
                y: parent_y,
                z: z_base,
                min_y: 0.0,
                max_y: parent_y,
                children: vec![l, r],
                child_labels: vec!["L", "R"],
                edge_dirs: vec![EdgeDir::Down, EdgeDir::Down],
            }
        }

        AstNode::TernaryExpr { condition, consequent, alternate } => {
            let mut cond = build_layout(condition, z_base, counter);
            let mut then_ = build_layout(consequent, z_base + Z_SPREAD, counter);
            let mut else_ = build_layout(alternate, z_base - Z_SPREAD, counter);

            let cond_height = cond.max_y - cond.min_y;
            let ternary_y = cond_height + 1.0;

            // Condition below ternary node
            let condminy = cond.min_y;
            shift_y(&mut cond, -condminy);

            // Branches above ternary node
            let branch_base = ternary_y + 1.0;
            let thenminy = then_.min_y;
            let elseminy = else_.min_y;
            shift_y(&mut then_, branch_base - thenminy);
            shift_y(&mut else_, branch_base - elseminy);

            let branch_max = f32::max(then_.max_y, else_.max_y);
            let width = f32::max(cond.width, f32::max(then_.width, else_.width)).max(2.0);

            LNode {
                id,
                ast: node.clone(),
                width,
                x: 0.0,
                y: ternary_y,
                z: z_base,
                min_y: 0.0,
                max_y: branch_max,
                children: vec![cond, then_, else_],
                child_labels: vec!["cond", "then", "else"],
                edge_dirs: vec![EdgeDir::Down, EdgeDir::Up, EdgeDir::Up],
            }
        }
    }
}

fn shift_y(node: &mut LNode, dy: f32) {
    node.y += dy;
    node.min_y += dy;
    node.max_y += dy;
    for child in &mut node.children {
        shift_y(child, dy);
    }
}

/// Assign X positions (horizontal spread).
fn assign_x(node: &mut LNode, x_start: f32) {
    if node.children.is_empty() {
        node.x = x_start + node.width / 2.0;
        return;
    }

    if matches!(node.ast, AstNode::TernaryExpr { .. }) {
        // Center each child within parent's span
        for child in &mut node.children {
            let child_start = x_start + (node.width - child.width) / 2.0;
            assign_x(child, child_start);
        }
        node.x = x_start + node.width / 2.0;
    } else {
        // Binary: spread children side by side
        let mut offset = x_start;
        for child in &mut node.children {
            assign_x(child, offset);
            offset += child.width;
        }
        let sum_x: f32 = node.children.iter().map(|c| c.x).sum();
        node.x = sum_x / node.children.len() as f32;
    }
}

/// Flatten the tree into nodes and edges.
fn collect(
    node: &LNode,
    center_x: f32,
    center_y: f32,
    sx: f32,
    sy: f32,
    nodes: &mut Vec<LayoutNode>,
    edges: &mut Vec<LayoutEdge>,
) {
    let pos = Vec3::new(
        (node.x - center_x) * sx,
        (node.y - center_y) * sy,
        node.z,
    );

    nodes.push(LayoutNode {
        id: node.id,
        ast: node.ast.clone(),
        pos,
        eval_step: node.y,
    });

    for (i, child) in node.children.iter().enumerate() {
        let child_pos = Vec3::new(
            (child.x - center_x) * sx,
            (child.y - center_y) * sy,
            child.z,
        );
        edges.push(LayoutEdge {
            from_id: node.id,
            to_id: child.id,
            from_pos: pos,
            to_pos: child_pos,
            label: node.child_labels[i],
            dir: node.edge_dirs[i],
        });
        collect(child, center_x, center_y, sx, sy, nodes, edges);
    }
}

// ── Public API ──────────────────────────────────────────────

/// Spacing constants for the 3D layout.
const SPACING_X: f32 = 2.0;
const SPACING_Y: f32 = 2.0;

/// Compute the full 3D layout for an AST.
pub fn compute_layout(ast: &AstNode) -> (Vec<LayoutNode>, Vec<LayoutEdge>) {
    let mut counter = 0;
    let mut root = build_layout(ast, 0.0, &mut counter);
    assign_x(&mut root, 0.0);

    // Compute center for visual centering
    let (mut all_x, mut all_y) = (vec![], vec![]);
    fn gather(n: &LNode, xs: &mut Vec<f32>, ys: &mut Vec<f32>) {
        xs.push(n.x);
        ys.push(n.y);
        for c in &n.children {
            gather(c, xs, ys);
        }
    }
    gather(&root, &mut all_x, &mut all_y);

    let center_x = all_x.iter().sum::<f32>() / all_x.len() as f32;
    let min_y = all_y.iter().copied().fold(f32::INFINITY, f32::min);
    let max_y = all_y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let center_y = (min_y + max_y) / 2.0;

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    collect(&root, center_x, center_y, SPACING_X, SPACING_Y, &mut nodes, &mut edges);

    (nodes, edges)
}
