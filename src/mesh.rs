/// Build an octahedron mesh (6 verts, 8 faces).
pub fn octahedron_mesh(size: f32) -> bevy::render::mesh::Mesh {
    let v = [
        [0.0, size, 0.0],  // 0 top
        [0.0, -size, 0.0], // 1 bottom
        [size, 0.0, 0.0],  // 2 +X
        [-size, 0.0, 0.0], // 3 -X
        [0.0, 0.0, size],  // 4 +Z
        [0.0, 0.0, -size], // 5 -Z
    ];
    let indices: Vec<u32> = vec![
        0, 4, 2, 0, 2, 5, 0, 5, 3, 0, 3, 4, 1, 2, 4, 1, 5, 2, 1, 3, 5, 1, 4, 3,
    ];
    // Compute flat normals per face
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for tri in indices.chunks(3) {
        let a = bevy::math::Vec3::from(v[tri[0] as usize]);
        let b = bevy::math::Vec3::from(v[tri[1] as usize]);
        let c = bevy::math::Vec3::from(v[tri[2] as usize]);
        let normal = (b - a).cross(c - a).normalize();
        for &idx in tri {
            positions.push(v[idx as usize]);
            normals.push(normal.to_array());
        }
    }
    let flat_indices: Vec<u32> = (0..positions.len() as u32).collect();

    bevy::render::mesh::Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        bevy::render::render_asset::RenderAssetUsages::default(),
    )
    .with_inserted_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(bevy::render::mesh::Indices::U32(flat_indices))
}

/// Creates a cone mesh pointing up (+Y) with the base centered at the origin.
///
/// - `radius`: base radius
/// - `height`: total height
/// - `segments`: number of subdivisions around the base (higher = smoother)
pub fn create_cone_mesh(radius: f32, height: f32, segments: u32) -> bevy::render::mesh::Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let tip = [0.0, height, 0.0];
    let slope = radius / height; // for normal calculation

    // --- Side faces (each segment is a separate triangle with its own normals) ---
    for i in 0..segments {
        let angle0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let angle1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;

        let (sin0, cos0) = angle0.sin_cos();
        let (sin1, cos1) = angle1.sin_cos();

        let base0 = [radius * cos0, 0.0, radius * sin0];
        let base1 = [radius * cos1, 0.0, radius * sin1];

        // Outward-facing normals (smoothed by averaging the two edge normals)
        let mid_angle = (angle0 + angle1) / 2.0;
        let ny = slope;
        let len = (1.0 + ny * ny).sqrt();
        let n0 = [cos0 / len, ny / len, sin0 / len];
        let n1 = [cos1 / len, ny / len, sin1 / len];
        let n_mid_angle = mid_angle;
        let n_tip = [n_mid_angle.cos() / len, ny / len, n_mid_angle.sin() / len];

        let idx = positions.len() as u32;

        positions.push(base0);
        positions.push(base1);
        positions.push(tip);

        normals.push(n0);
        normals.push(n1);
        normals.push(n_tip);

        uvs.push([i as f32 / segments as f32, 0.0]);
        uvs.push([(i + 1) as f32 / segments as f32, 0.0]);
        uvs.push([(i as f32 + 0.5) / segments as f32, 1.0]);

        indices.push(idx);
        indices.push(idx + 1);
        indices.push(idx + 2);
    }

    // --- Base cap (fan from center) ---
    let center_idx = positions.len() as u32;
    positions.push([0.0, 0.0, 0.0]);
    normals.push([0.0, -1.0, 0.0]);
    uvs.push([0.5, 0.5]);

    for i in 0..segments {
        let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let (sin_a, cos_a) = angle.sin_cos();

        let idx = positions.len() as u32;
        positions.push([radius * cos_a, 0.0, radius * sin_a]);
        normals.push([0.0, -1.0, 0.0]);
        uvs.push([0.5 + 0.5 * cos_a, 0.5 + 0.5 * sin_a]);

        let next = if i + 1 < segments {
            idx + 1
        } else {
            center_idx + 1
        };

        // Wind clockwise when viewed from below (normal is -Y)
        indices.push(center_idx);
        indices.push(next);
        indices.push(idx);
    }

    let mut mesh = bevy::render::mesh::Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        bevy::utils::default(),
    );
    mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}

/// Creates a "bool" type mesh: two cones sharing a base at the origin,
/// one pointing up (+Y) and one pointing down (-Y).
pub fn create_bool_mesh(radius: f32, half_height: f32, segments: u32) -> bevy::render::mesh::Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let slope = radius / half_height;
    let len = (1.0 + slope * slope).sqrt();

    // Helper to add one cone's side faces
    // tip_y: where the apex sits, base_y: where the ring sits
    // flip: whether to reverse winding
    let mut add_cone_sides = |tip_y: f32, base_y: f32, flip: bool| {
        let tip = [0.0, tip_y, 0.0];
        let ny = if tip_y > base_y { slope } else { -slope };

        for i in 0..segments {
            let angle0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let angle1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let mid_angle = (angle0 + angle1) / 2.0;

            let (sin0, cos0) = angle0.sin_cos();
            let (sin1, cos1) = angle1.sin_cos();

            let b0 = [radius * cos0, base_y, radius * sin0];
            let b1 = [radius * cos1, base_y, radius * sin1];

            let n0 = [cos0 / len, ny / len, sin0 / len];
            let n1 = [cos1 / len, ny / len, sin1 / len];
            let n_tip = [mid_angle.cos() / len, ny / len, mid_angle.sin() / len];

            let idx = positions.len() as u32;
            positions.push(b0);
            positions.push(b1);
            positions.push(tip);
            normals.push(n0);
            normals.push(n1);
            normals.push(n_tip);
            uvs.push([i as f32 / segments as f32, 0.0]);
            uvs.push([(i + 1) as f32 / segments as f32, 0.0]);
            uvs.push([(i as f32 + 0.5) / segments as f32, 1.0]);

            if flip {
                indices.push(idx + 1);
                indices.push(idx);
                indices.push(idx + 2);
            } else {
                indices.push(idx);
                indices.push(idx + 1);
                indices.push(idx + 2);
            }
        }
    };

    // Upper cone: base at origin, tip at +half_height
    add_cone_sides(half_height, 0.0, false);
    // Lower cone: base at origin, tip at -half_height
    add_cone_sides(-half_height, 0.0, true);

    let mut mesh = bevy::render::mesh::Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        bevy::utils::default(),
    );
    mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(bevy::render::mesh::Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}
