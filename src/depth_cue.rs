//! A full-screen pass that reads the scene's depth buffer.
//!
//! The scene is drawn unlit on purpose — colour carries the type, and lighting
//! would make one type read differently depending on which way a face happens
//! to point. The cost of that choice is that the picture holds no shading
//! gradient at all, and a parallel projection takes away the last geometric
//! cue on top: a still frame flattens, and how the nodes stand relative to one
//! another stops being readable. What is left to recover the depth from is the
//! depth buffer itself, in image space, where it touches neither the materials
//! nor the colours.
//!
//! What the shader makes of it is two effects off the one buffer: depth
//! darkening (Luft/Colditz/Deussen, SIGGRAPH 2006), which lays a soft shadow
//! behind the nearer of two overlapping nodes, and a hard silhouette line at
//! depth steps, which is worth more on flat unlit boxes than any amount of soft
//! halo. Both come out as a multiplier on the painted colour, so a type's
//! colour still means exactly what it meant.
//!
//! Two things about the depth it reads are worth keeping in mind:
//!
//! - It is the **main** depth texture, not a prepass copy. Order-independent
//!   transparency already asks for `TEXTURE_BINDING` on it, so no depth prepass
//!   is needed and the geometry is not walked a second time.
//! - Only opaque geometry writes depth, so the cue is about the node bodies.
//!   Edges, type markers, caret faces and the grid blend and stay out of it,
//!   which is the right split: the bodies are what have to be told apart.

use bevy::asset::{load_internal_asset, uuid_handle};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::FullscreenShader;
use bevy::ecs::query::QueryItem;
use bevy::image::BevyDefault;
use bevy::prelude::*;
use bevy::render::extract_component::{
    ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
    UniformComponentPlugin,
};
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{
    sampler, texture_2d, texture_depth_2d, uniform_buffer,
};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::view::{ViewDepthTexture, ViewTarget};
use bevy::render::{RenderApp, RenderStartup};

pub const DEPTH_CUE_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("44455054-4820-4355-8000-000000000001");

/// The key to reading the depth buffer back, carried on the camera.
///
/// It rides on the camera rather than sitting in a resource because that is
/// what puts it in front of the render world: `ExtractComponent` copies it to
/// the view entity, where the node can reach it as a dynamic uniform.
#[derive(Component, Clone, Copy, ExtractComponent, ShaderType)]
pub struct DepthCue {
    /// The bound projection's near plane. Normally negative — the projection
    /// box reaches behind the camera on purpose; see `camera::depth_range`.
    pub near: f32,
    /// The bound projection's far plane.
    pub far: f32,
    /// Reach of the halo, in pixels.
    pub halo_radius: f32,
    /// How dark the halo goes at full separation, `0.0..=1.0`.
    pub halo_strength: f32,
    /// Separation, in cells, at which the halo reaches `halo_strength`.
    pub halo_depth: f32,
    /// Depth step, in cells, that counts as a silhouette.
    pub edge_threshold: f32,
    /// How dark the silhouette line goes, `0.0..=1.0`.
    pub edge_strength: f32,
    /// Which of [`MODE_OFF`], [`MODE_CUE`], [`MODE_DEPTH`] the pass draws.
    pub mode: u32,
}

/// Pass the frame through untouched.
///
/// The pass still runs and still blits in this mode, which is deliberate: every
/// mode travels the same ping-pong path, so anything that changes in the
/// picture when F9 is struck is the cue and never the plumbing.
pub const MODE_OFF: u32 = 0;
/// Depth darkening and the silhouette line — the cue proper.
pub const MODE_CUE: u32 = 1;
/// The linearised depth itself, as one contour band per cell. Not a cue but a
/// way to see what the cue is reading.
pub const MODE_DEPTH: u32 = 2;
/// What F9 walks through, in order, wrapping at the end.
const MODES: [u32; 3] = [MODE_OFF, MODE_CUE, MODE_DEPTH];

// Defaults, gathered here because they are the knobs worth turning and nothing
// else in the module is. Lengths are in the units the shader reads them in:
// `halo_radius` in pixels, because a halo is a thing of the picture and should
// not grow when the camera zooms in; everything else in cells, because that is
// what the scene is built out of.
//
// A cell is `camera::DEFAULT_CELL_PIXELS` wide, so the radius below is a bit
// under half a cell — enough to read as a shadow, short enough that two nodes a
// cell apart still get their own.
const HALO_RADIUS: f32 = 16.0;
const HALO_STRENGTH: f32 = 0.55;
const HALO_DEPTH: f32 = 2.0;
const EDGE_THRESHOLD: f32 = 0.3;
const EDGE_STRENGTH: f32 = 0.45;

impl Default for DepthCue {
    /// Off, and with a depth range that will be overwritten on the first
    /// update. `sync_depth_cue` owns the two planes; standing values here
    /// would only be a second answer to the same question.
    fn default() -> Self {
        Self {
            near: 0.0,
            far: 1.0,
            halo_radius: HALO_RADIUS,
            halo_strength: HALO_STRENGTH,
            halo_depth: HALO_DEPTH,
            edge_threshold: EDGE_THRESHOLD,
            edge_strength: EDGE_STRENGTH,
            mode: MODE_OFF,
        }
    }
}

/// The key that walks through [`MODES`].
///
/// A function key, and read straight from `ButtonInput` rather than from the
/// keyboard message the editor listens to. Both halves of that matter: the
/// letters all belong to NORMAL mode or to the insert prompt, and a function
/// key carries no text, so this cannot end up typed into a field. It is also
/// left outside `keyboard_captured` on purpose — what it changes is how the
/// scene is drawn, not what the editor is doing, so it stays available while a
/// modal or an evaluation owns the keyboard.
const TOGGLE_KEY: KeyCode = KeyCode::F9;

/// Keep the depth range on the camera's component in step with the projection.
///
/// The bound projection rebuilds its planes from the orbit radius every time
/// the camera moves, so the pair has to be re-read rather than captured once.
fn sync_depth_cue(orbit: Res<super::camera::OrbitCamera>, mut cues: Query<&mut DepthCue>) {
    let (near, far) = super::camera::depth_range(orbit.radius);
    for mut cue in &mut cues {
        // Written through `Mut` only when it actually differs, so the change
        // detection that drives extraction stays meaningful.
        if cue.near != near || cue.far != far {
            cue.near = near;
            cue.far = far;
        }
    }
}

fn toggle_depth_cue(keys: Res<ButtonInput<KeyCode>>, mut cues: Query<&mut DepthCue>) {
    if !keys.just_pressed(TOGGLE_KEY) {
        return;
    }
    for mut cue in &mut cues {
        // Walk the list rather than counting modulo the count, so a mode that
        // has fallen out of `MODES` cannot strand the cue on a number the
        // shader no longer knows.
        let next = MODES
            .iter()
            .position(|m| *m == cue.mode)
            .map_or(MODE_OFF, |i| MODES[(i + 1) % MODES.len()]);
        cue.mode = next;
    }
}

pub struct DepthCuePlugin;

impl Plugin for DepthCuePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            DEPTH_CUE_SHADER_HANDLE,
            "../assets/shaders/depth_cue.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins((
            ExtractComponentPlugin::<DepthCue>::default(),
            UniformComponentPlugin::<DepthCue>::default(),
        ))
        .add_systems(Update, (sync_depth_cue, toggle_depth_cue));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(RenderStartup, init_pipeline)
            .add_render_graph_node::<ViewNodeRunner<DepthCueNode>>(Core3d, DepthCueLabel)
            // After `EndMainPass`, so the OIT resolve — which sits between the
            // transparent pass and that label — has already composited, and
            // before tonemapping, so the work happens in linear HDR. A cue that
            // darkens belongs on the linear side of the curve; past the
            // tonemapper the same factor would bend with the curve's slope and
            // read differently in the bright parts of the frame than in the
            // dark ones.
            .add_render_graph_edges(
                Core3d,
                (Node3d::EndMainPass, DepthCueLabel, Node3d::Tonemapping),
            );
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct DepthCueLabel;

#[derive(Resource)]
struct DepthCuePipeline {
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
    pipeline_id_hdr: CachedRenderPipelineId,
}

fn init_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "depth_cue_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                // The frame as the main pass left it.
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                // The scene's depth. Plain rather than multisampled: OIT
                // refuses to run with MSAA on — it panics outright — so the
                // camera is `Msaa::Off` and this texture has a single sample.
                texture_depth_2d(),
                uniform_buffer::<DepthCue>(true),
            ),
        ),
    );

    let mut desc = RenderPipelineDescriptor {
        label: Some("depth_cue_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: DEPTH_CUE_SHADER_HANDLE,
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::bevy_default(),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    };
    let pipeline_id = pipeline_cache.queue_render_pipeline(desc.clone());

    // The camera renders HDR, so this is the one that actually runs. The
    // other is kept so that turning HDR off stays a one-line change rather
    // than a format mismatch at pipeline creation.
    desc.fragment.as_mut().unwrap().targets[0]
        .as_mut()
        .unwrap()
        .format = ViewTarget::TEXTURE_FORMAT_HDR;
    let pipeline_id_hdr = pipeline_cache.queue_render_pipeline(desc);

    commands.insert_resource(DepthCuePipeline {
        layout,
        sampler: render_device.create_sampler(&SamplerDescriptor::default()),
        pipeline_id,
        pipeline_id_hdr,
    });
}

#[derive(Default)]
struct DepthCueNode;

impl ViewNode for DepthCueNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static DynamicUniformIndex<DepthCue>,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_target, view_depth, cue_index): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let cue_pipeline = world.resource::<DepthCuePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let pipeline_id = if view_target.is_hdr() {
            cue_pipeline.pipeline_id_hdr
        } else {
            cue_pipeline.pipeline_id
        };
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
            return Ok(());
        };

        let Some(cue_binding) = world
            .resource::<ComponentUniforms<DepthCue>>()
            .uniforms()
            .binding()
        else {
            return Ok(());
        };

        // Claims the other half of the view's ping-pong: `source` is what the
        // scene was drawn into, `destination` is what the rest of the pipeline
        // will read. Both are held at once, which is why the pass cannot simply
        // read and write one texture.
        let post_process = view_target.post_process_write();

        let bind_group = render_context.render_device().create_bind_group(
            "depth_cue_bind_group",
            &pipeline_cache.get_bind_group_layout(&cue_pipeline.layout),
            &BindGroupEntries::sequential((
                post_process.source,
                &cue_pipeline.sampler,
                view_depth.view(),
                cue_binding.clone(),
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("depth_cue_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            // No depth attachment: the depth texture is bound for reading here,
            // and a texture cannot be an attachment and a binding in the same
            // pass.
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[cue_index.index()]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}
