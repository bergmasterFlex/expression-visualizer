use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::reflect::TypePath;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::render::mesh::MeshVertexBufferLayoutRef;

// --- Configuration ---

#[derive(Resource)]
pub struct GridConfig {
    pub spacing: f32,
    pub half_extent: f32,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            spacing: 3.0,
            half_extent: 200.0,
        }
    }
}

// --- Shader Material ---

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GridMaterial {
    #[uniform(0)]
    pub plane_color: LinearRgba,
    #[uniform(1)]
    pub line_color: LinearRgba,
    #[uniform(2)]
    pub spacing: f32,
    #[uniform(3)]
    pub fade_start: f32,
    #[uniform(4)]
    pub fade_end: f32,
    #[uniform(5)]
    pub line_thickness: f32,
}

impl Material for GridMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/grid.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

// --- Plugin ---

pub struct GridPlugin;

impl Plugin for GridPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GridConfig>()
            .add_plugins(MaterialPlugin::<GridMaterial>::default())
            .add_systems(Startup, spawn_grid);
    }
}

fn spawn_grid(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GridMaterial>>,
    config: Res<GridConfig>,
) {
    let size = config.half_extent * 2.0;
    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(Plane3d::default().mesh().size(size, size).build()),
        material: materials.add(GridMaterial {
            plane_color: LinearRgba::new(0.2, 0.2, 0.3, 0.05),
            line_color: LinearRgba::new(0.2, 0.2, 0.5, 0.15),
            spacing: config.spacing,
            fade_start: 15.0,
            fade_end: 100.0,
            line_thickness: 0.6
        }),
        ..default()
    });
}
