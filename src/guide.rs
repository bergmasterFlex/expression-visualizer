//! Dashed guides leaving a caret for the faces of the volume it stands in.
//!
//! A caret says which cell is addressed; six guides say *where that cell is* —
//! how far it stands from each wall of its own scope, on every axis at once.
//! That is the question a single box in a volume several cells deep cannot
//! answer by itself, and the one the pointer's wheel makes worth asking.
//!
//! The dashes are the shader's and not the geometry's. One unit cube, scaled to
//! length along its axis, serves every guide there is: a line that grew or
//! shrank by rebuilding its dashes would mean rebuilding a mesh whenever the
//! pointer crossed a cell.

use bevy::asset::{load_internal_asset, uuid_handle};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::*;
use bevy::shader::ShaderRef;

/// Whether the guides are drawn at all.
///
/// A resource and not a constant, because it is a checkbox — and one that costs
/// nothing to flip. Unlike `lod::Clipping` beside it, which is baked into every
/// material at spawn and so has to ask for a rebuild, the guides are placed
/// afresh each frame and simply stop being placed.
///
/// Off to begin with. Six dashed lines from the caret to the six walls answer
/// a question — *where in the volume is this cell* — that is worth asking
/// while the shape is being learnt and not once it is known, and they are the
/// one piece of chrome drawn in among the nodes rather than over them. The
/// checkbox is where they are asked for.
#[derive(Resource)]
pub struct Guides(pub bool);

impl Default for Guides {
    fn default() -> Self {
        Self(false)
    }
}

pub const GUIDE_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("47554944-4500-4000-8000-000000000001");

/// Where a dash's flank is cut. Must match `COVERAGE_CUTOFF` in
/// `assets/shaders/guide.wgsl`; the material keys its alpha mode with the same
/// number, so the two cannot disagree about where the line ends.
pub const COVERAGE_CUTOFF: f32 = 0.5;

/// Length of one dash-plus-gap, in world units along the guide.
///
/// A divisor of the cell on purpose. The shader counts from the world origin,
/// so a period that divides the cell puts every break on a grid boundary and
/// makes the dashes readable as cells rather than as an arbitrary rhythm.
pub const DASH_PERIOD: f32 = crate::render::CELL * 0.5;

/// Share of a period that is drawn: half on, half off.
pub const DASH_DUTY: f32 = 0.5;

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GuideMaterial {
    #[uniform(0)]
    pub color: LinearRgba,
    /// The world axis the dashes are counted along. `w` is unused padding.
    #[uniform(0)]
    pub axis: Vec4,
    #[uniform(0)]
    pub dash_period: f32,
    #[uniform(0)]
    pub dash_duty: f32,
    /// Padding to keep the uniform block 16-byte aligned.
    #[uniform(0)]
    pub _pad: Vec2,
}

impl GuideMaterial {
    /// A guide counted along `axis` and painted at `level`, which is a linear
    /// brightness and so goes through the tonemapper the same way
    /// `render::DISPLAY_WHITE` and `render::HOVER_GREY` do.
    ///
    /// Both of those are fixed for a guide's whole life — the axis it runs
    /// along and the caret it belongs to never change — so one material serves
    /// the pair of guides leaving a cell in either direction on that axis.
    pub fn along(axis: Vec3, level: f32) -> Self {
        Self {
            color: LinearRgba::new(level, level, level, 1.0),
            axis: axis.extend(0.0),
            dash_period: DASH_PERIOD,
            dash_duty: DASH_DUTY,
            _pad: Vec2::ZERO,
        }
    }
}

impl Material for GuideMaterial {
    fn fragment_shader() -> ShaderRef {
        GUIDE_SHADER_HANDLE.into()
    }

    /// Masked, never blended. A guide belongs in the depth buffer so the scene
    /// occludes it, and a cut fragment costs no OIT slot — which matters, since
    /// twelve guides crossing a nested volume would otherwise spend the layer
    /// budget the volume's own surfaces need.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Mask(COVERAGE_CUTOFF)
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

pub struct GuidePlugin;

impl Plugin for GuidePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            GUIDE_SHADER_HANDLE,
            "../assets/shaders/guide.wgsl",
            Shader::from_wgsl
        );
        app.add_plugins(MaterialPlugin::<GuideMaterial>::default());
    }
}
