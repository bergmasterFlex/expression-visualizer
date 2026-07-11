/// Build a square pyramid mesh with the base in the XY plane at z = 0
/// and the apex at (0, 0, -depth). Flat shading via per-face normals.
pub fn square_pyramid_z_mesh(base: f32, depth: f32) -> bevy::mesh::Mesh {
    rect_pyramid_z_mesh(base, base, depth)
}

/// Build a rectangular pyramid mesh with a base_x × base_y rectangle in the
/// XY plane at z = 0 and the apex at (0, 0, -depth). Flat shading via
/// per-face normals.
pub fn rect_pyramid_z_mesh(base_x: f32, base_y: f32, depth: f32) -> bevy::mesh::Mesh {
    let hx = base_x / 2.0;
    let hy = base_y / 2.0;
    let p0 = [-hx, -hy, 0.0];
    let p1 = [hx, -hy, 0.0];
    let p2 = [hx, hy, 0.0];
    let p3 = [-hx, hy, 0.0];
    let tip = [0.0, 0.0, -depth];

    // Side faces wound so outward normals point away from the central axis;
    // base faces wound so the outward normal points +Z (toward the camera).
    let face_tris: [[[f32; 3]; 3]; 6] = [
        [p0, tip, p1],
        [p1, tip, p2],
        [p2, tip, p3],
        [p3, tip, p0],
        [p0, p1, p2],
        [p0, p2, p3],
    ];

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(18);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(18);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(18);
    for tri in face_tris.iter() {
        let a = bevy::math::Vec3::from(tri[0]);
        let b = bevy::math::Vec3::from(tri[1]);
        let c = bevy::math::Vec3::from(tri[2]);
        let n = (b - a).cross(c - a).normalize().to_array();
        positions.push(tri[0]);
        positions.push(tri[1]);
        positions.push(tri[2]);
        normals.push(n);
        normals.push(n);
        normals.push(n);
        uvs.push([0.0, 0.0]);
        uvs.push([1.0, 0.0]);
        uvs.push([0.5, 1.0]);
    }
    let indices: Vec<u32> = (0..positions.len() as u32).collect();

    let mut mesh = bevy::mesh::Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::mesh::Indices::U32(indices));
    mesh
}

/// Build a pyramid mesh from 4 base corners and a tip. Base corners must be
/// supplied in CCW order viewed from outside the base (the side opposite
/// the tip). Side faces wind as (p[i], p[i+1], tip); base winds as
/// (p0, p1, p2) + (p0, p2, p3). Flat shading via per-face normals.
pub fn pyramid_5pt_mesh(base: [[f32; 3]; 4], tip: [f32; 3]) -> bevy::mesh::Mesh {
    let face_tris: [[[f32; 3]; 3]; 6] = [
        [base[0], base[1], tip],
        [base[1], base[2], tip],
        [base[2], base[3], tip],
        [base[3], base[0], tip],
        [base[0], base[1], base[2]],
        [base[0], base[2], base[3]],
    ];

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(18);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(18);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(18);
    for tri in face_tris.iter() {
        let a = bevy::math::Vec3::from(tri[0]);
        let b = bevy::math::Vec3::from(tri[1]);
        let c = bevy::math::Vec3::from(tri[2]);
        let n = (b - a).cross(c - a).normalize().to_array();
        positions.push(tri[0]);
        positions.push(tri[1]);
        positions.push(tri[2]);
        normals.push(n);
        normals.push(n);
        normals.push(n);
        uvs.push([0.0, 0.0]);
        uvs.push([1.0, 0.0]);
        uvs.push([0.5, 1.0]);
    }
    let indices: Vec<u32> = (0..positions.len() as u32).collect();

    let mut mesh = bevy::mesh::Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::mesh::Indices::U32(indices));
    mesh
}
