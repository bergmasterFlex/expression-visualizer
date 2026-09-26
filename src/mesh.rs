/// A list of cut positions turned into the boundaries of a strip cover of
/// `0..=1`: clamped, sorted, deduplicated, and with both ends added if they
/// are missing.
///
/// Done here rather than at the call site because the caller's two grids say
/// nothing about each other. A function call's near face is `input_rows` tall
/// and its far face `output_rows`, and neither count knows the other — so the
/// caller hands over both grids as they stand and this reconciles them.
///
/// The epsilon is not a guess. Two distinct fractions `i/a` and `j/b` differ by
/// at least `1/(a*b)`, and the column and row counts a node can have are single
/// digits, so the narrowest gap that can really occur is around `0.04` — four
/// hundred times this. Anything closer came of float division on a value that
/// was meant to be the same cut twice.
fn normalised_cuts(cuts: &[f32]) -> Vec<f32> {
    let mut out: Vec<f32> = cuts
        .iter()
        .copied()
        .chain([0.0, 1.0])
        .map(|c| c.clamp(0.0, 1.0))
        .collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
    out
}

/// How far inside its own span a quad reads the colour field.
///
/// At the near face the field steps — one input's colour beside the next's —
/// and a corner sitting exactly on that step has no single answer, while the
/// two quads meeting there need opposite ones. Reading a thousandth of a span
/// in hands each quad its own side of it. Everywhere the field is continuous,
/// which is everywhere else, the shift is orders of magnitude under anything
/// that can be seen.
///
/// Public because it is a property of *reading* the field and not of this
/// mesh: whatever else samples the same field over a patch of the same body —
/// the name printed on a body's roof does — has to step in at the same places
/// for the same reason, or its edge will not meet the body's.
pub const FIELD_SAMPLE_INSET: f32 = 1e-3;

/// Build the outer shell of a frustum, cut into a grid and painted from a
/// colour field.
///
/// Both quads must be supplied in the same rotational order, CCW viewed from
/// outside the base (the side opposite the top), and the corners are read as
/// `[0] = (u=0, v=1)`, `[1] = (1, 1)`, `[2] = (1, 0)`, `[3] = (0, 0)` — the
/// order `render`'s `quad()` writes, bottom edge first. A point of the solid is
/// `lerp(bilinear(base, u, v), bilinear(top, u, v), w)`.
///
/// Unlike a centred base/top pair this lets the two faces be aligned
/// independently — a function call's near face is flush with its input anchor
/// block and its far face with the output anchor, and those stacks are top
/// aligned rather than centred on each other.
///
/// The three cut lists are a *sampling rate*, not a set of edges. The field
/// this paints from is sharp only where it starts — at the near face one
/// input's colour stands beside the next's — and melts on the way across, so
/// there is no grid of flat cells to trace. A vertex colour ramps straight to
/// the next vertex, so a passage narrower than the spacing comes out as a fold
/// instead of a ramp: the cuts have to be closer together than the narrowest
/// passage that is still meant to read as one. Where the field *is* flat they
/// cost only triangles.
///
/// Every face is cut on every list that reaches it — caps on `u` and `v`, each
/// side on its own cross axis and on `w` — and the shared lists are what keeps
/// a cap's edge and the side beside it in step. Cutting the two differently
/// would leave T-junctions, and a T-junction is a hairline crack with the
/// background behind it.
///
/// `color_at(u, v, w)` is asked for **linear** RGBA — vertex colours reach the
/// shader unconverted — at the corners of each quad, stepped in by
/// `FIELD_SAMPLE_INSET` so that a quad meeting a step in the field gets its own
/// side of it.
///
/// Only the shell is built — no faces between neighbouring cells. `lod::faded`
/// pushes a graded body into `AlphaMode::Blend`, and an interior face would
/// then blend through and darken the body; it would also spend an OIT layer,
/// which is the scarce thing here, on something that can never be seen.
///
/// Flat shading via per-face normals, as everything else in this module. Since
/// each sub-quad carries its own, a ruled side comes out flat-shaded per patch
/// rather than under one normal for its whole length.
pub fn graded_frustum_shell_mesh(
    base: [[f32; 3]; 4],
    top: [[f32; 3]; 4],
    u_cuts: &[f32],
    v_cuts: &[f32],
    w_cuts: &[f32],
    color_at: impl Fn(f32, f32, f32) -> [f32; 4],
) -> bevy::mesh::Mesh {
    let us = normalised_cuts(u_cuts);
    let vs = normalised_cuts(v_cuts);
    let ws = normalised_cuts(w_cuts);
    let nu = us.len() - 1;
    let nv = vs.len() - 1;

    // Bilinear position on one face, in the corner order stated above: the
    // `v = 0` edge runs `[3] -> [2]`, the `v = 1` edge `[0] -> [1]`.
    let face_point = |q: &[[f32; 3]; 4], u: f32, v: f32| {
        let p = |i: usize| bevy::math::Vec3::from(q[i]);
        p(3).lerp(p(2), u).lerp(p(0).lerp(p(1), u), v)
    };
    let at = |u: f32, v: f32, w: f32| face_point(&base, u, v).lerp(face_point(&top, u, v), w);
    // The two ends of a span, each moved toward the other. See
    // `FIELD_SAMPLE_INSET`.
    let stepped_in = |a: f32, b: f32| {
        (
            a + (b - a) * FIELD_SAMPLE_INSET,
            b - (b - a) * FIELD_SAMPLE_INSET,
        )
    };

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();

    // One quad as two triangles, corners given in winding order. Each triangle
    // gets its own normal off its own corners, so a degenerate one cannot drag
    // its neighbour down with it.
    let mut push_quad = |p: [bevy::math::Vec3; 4], uv: [[f32; 2]; 4], c: [[f32; 4]; 4]| {
        for tri in [[0usize, 1, 2], [0, 2, 3]] {
            let a = p[tri[0]];
            let b = p[tri[1]];
            let d = p[tri[2]];
            // A side that did not taper is degenerate and would normalise to
            // NaN; fall back to the axis normal.
            let n = (b - a)
                .cross(d - a)
                .try_normalize()
                .unwrap_or(bevy::math::Vec3::Z)
                .to_array();
            for i in tri {
                positions.push(p[i].to_array());
                normals.push(n);
                uvs.push(uv[i]);
                colors.push(c[i]);
            }
        }
    };

    // The two caps. Corners are named in the same order the quads were handed
    // over in, so the near cap winds exactly as an unsubdivided base would and
    // the far cap is that ring reversed.
    for j in 0..nv {
        for i in 0..nu {
            let (u0, u1) = (us[i], us[i + 1]);
            let (v0, v1) = (vs[j], vs[j + 1]);
            let (su0, su1) = stepped_in(u0, u1);
            let (sv0, sv1) = stepped_in(v0, v1);
            let corners = [(u0, v1), (u1, v1), (u1, v0), (u0, v0)];
            let samples = [(su0, sv1), (su1, sv1), (su1, sv0), (su0, sv0)];
            for (w, reversed) in [(0.0, false), (1.0, true)] {
                let mut points = corners.map(|(u, v)| at(u, v, w));
                let mut uv = corners.map(|(u, v)| [u, v]);
                let mut corner_colors = samples.map(|(u, v)| color_at(u, v, w));
                if reversed {
                    points.reverse();
                    uv.reverse();
                    corner_colors.reverse();
                }
                push_quad(points, uv, corner_colors);
            }
        }
    }

    // The four sides, walked as one ring `[0] -> [1] -> [2] -> [3] -> [0]` so
    // that the outward winding of the unsubdivided frustum carries over segment
    // by segment. Each segment is then cut again along `w`, which is what lets
    // a side show the field melting along the taper rather than only its two
    // ends: a side spanning the whole length in one quad can draw nothing
    // between them but a straight ramp.
    let mut ring = Vec::new();
    for i in 0..nu {
        ring.push(((us[i], 1.0), (us[i + 1], 1.0)));
    }
    for j in (0..nv).rev() {
        ring.push(((1.0, vs[j + 1]), (1.0, vs[j])));
    }
    for i in (0..nu).rev() {
        ring.push(((us[i + 1], 0.0), (us[i], 0.0)));
    }
    for j in 0..nv {
        ring.push(((0.0, vs[j]), (0.0, vs[j + 1])));
    }
    for ((ua, va), (ub, vb)) in ring {
        let (sample_ua, sample_ub) = stepped_in(ua, ub);
        let (sample_va, sample_vb) = stepped_in(va, vb);
        for k in 0..ws.len() - 1 {
            let (w0, w1) = (ws[k], ws[k + 1]);
            push_quad(
                [
                    at(ua, va, w0),
                    at(ua, va, w1),
                    at(ub, vb, w1),
                    at(ub, vb, w0),
                ],
                [[ua, va], [ua, va], [ub, vb], [ub, vb]],
                [
                    color_at(sample_ua, sample_va, w0),
                    color_at(sample_ua, sample_va, w1),
                    color_at(sample_ub, sample_vb, w1),
                    color_at(sample_ub, sample_vb, w0),
                ],
            );
        }
    }

    let indices: Vec<u32> = (0..positions.len() as u32).collect();

    let mut mesh = bevy::mesh::Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(bevy::mesh::Indices::U32(indices));
    mesh
}
