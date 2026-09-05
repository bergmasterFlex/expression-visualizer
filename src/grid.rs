use bevy::asset::{load_internal_asset, uuid_handle};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::*;
use bevy::shader::ShaderRef;

// --- Configuration ---

#[derive(Resource)]
pub struct GridConfig {
    pub spacing: f32,
    pub half_extent: f32,
    /// Distance (in world units) from origin where the grid starts fading out.
    pub fade_start: f32,
    /// Distance where the grid becomes fully transparent. Also the radius
    /// within which grid cells can be hovered/clicked.
    pub fade_end: f32,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            spacing: crate::render::CELL,
            half_extent: 200.0,
            fade_start: crate::render::CELL * 5.0,
            fade_end: crate::render::CELL * 34.0,
        }
    }
}

// --- Shader Material ---

pub const GRID_SHADER_HANDLE: Handle<Shader> = uuid_handle!("47524944-4d41-4000-8000-000000000001");

/// Maximum number of multi-cell node footprints that can suppress interior
/// grid lines in a single grid mesh. Must match `MAX_FOOTPRINTS` in
/// `assets/shaders/grid.wgsl`.
pub const MAX_FOOTPRINTS: usize = 16;

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GridMaterial {
    #[uniform(0)]
    pub plane_color: LinearRgba,
    #[uniform(0)]
    pub line_color: LinearRgba,
    #[uniform(0)]
    pub spacing: f32,
    #[uniform(0)]
    pub fade_start: f32,
    #[uniform(0)]
    pub fade_end: f32,
    #[uniform(0)]
    pub line_thickness: f32,
    /// World-space (x, z) of the hovered cell's center.
    #[uniform(0)]
    pub hover_pos: Vec2,
    /// 1.0 when a grid cell is hovered, 0.0 otherwise.
    #[uniform(0)]
    pub hover_active: f32,
    /// Padding to keep the uniform block 16-byte aligned.
    #[uniform(0)]
    pub _pad: f32,
    /// World-space (x, z) of the outer boundary min corner.
    #[uniform(0)]
    pub border_min: Vec2,
    /// World-space (x, z) of the outer boundary max corner.
    #[uniform(0)]
    pub border_max: Vec2,
    /// Color used when a fragment lies on the outer boundary.
    #[uniform(0)]
    pub border_color: LinearRgba,
    /// 1.0 = draw outer border, 0.0 = off.
    #[uniform(0)]
    pub border_active: f32,
    /// Multiplier over the normal line width for the border.
    #[uniform(0)]
    pub border_thickness: f32,
    /// Number of active entries in `footprints` (0..=MAX_FOOTPRINTS).
    #[uniform(0)]
    pub footprint_count: u32,
    #[uniform(0)]
    pub _pad2: f32,
    /// World-space footprints of multi-cell nodes: `xy = min.xz`, `zw =
    /// max.xz`. Fragments whose neighbours across the nearest grid line all
    /// lie inside the same footprint suppress that line.
    #[uniform(0)]
    pub footprints: [Vec4; MAX_FOOTPRINTS],
}

impl Material for GridMaterial {
    fn fragment_shader() -> ShaderRef {
        GRID_SHADER_HANDLE.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
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
        load_internal_asset!(
            app,
            GRID_SHADER_HANDLE,
            "../assets/shaders/grid.wgsl",
            Shader::from_wgsl
        );
        app.init_resource::<GridConfig>()
            .add_plugins(MaterialPlugin::<GridMaterial>::default())
            .add_systems(Startup, spawn_grid);
    }
}

/// Marker for the base (Y=0) grid so hover/click systems can skip it — the
/// base grid is a passive visual hint, not an interactive surface.
#[derive(Component)]
pub struct BaseGridEntity;

fn spawn_grid(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GridMaterial>>,
    config: Res<GridConfig>,
) {
    let size = config.half_extent * 2.0;
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(size, size).build())),
        MeshMaterial3d(materials.add(GridMaterial {
            plane_color: LinearRgba::new(0.2, 0.2, 0.3, 0.08),
            line_color: LinearRgba::new(0.2, 0.2, 0.5, 0.15),
            spacing: config.spacing,
            fade_start: config.fade_start,
            fade_end: config.fade_end,
            line_thickness: 0.6,
            hover_pos: Vec2::ZERO,
            hover_active: 0.0,
            _pad: 0.0,
            border_min: Vec2::ZERO,
            border_max: Vec2::ZERO,
            border_color: LinearRgba::WHITE,
            border_active: 0.0,
            border_thickness: 1.8,
            footprint_count: 0,
            _pad2: 0.0,
            footprints: [Vec4::ZERO; MAX_FOOTPRINTS],
        })),
        BaseGridEntity,
    ));
}
