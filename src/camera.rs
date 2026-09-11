use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
/// Camera controller for the 3D scene.
///
/// Two modes, and they are two purposes rather than two settings of one camera:
///
/// - **Bound** (the default) ties the view to the caret. The orientation is
///   fixed, the projection is orthographic, and the only thing the user sets is
///   the scale. Moving the caret moves the camera along a path — addressing is
///   discrete, the camera is continuous, and the travel is what keeps the
///   reader oriented.
/// - **Free** releases the camera entirely: orbit, pan and zoom by hand, in
///   perspective. The guarantees of the bound mode are deliberately suspended.
///
/// Leaving the bound mode is an explicit act (the camera-mode button), never a
/// side effect of dragging.
use bevy::prelude::*;

/// Which of the two camera purposes is active.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum CameraMode {
    #[default]
    Bound,
    Free,
}

/// Orbit camera state stored as a resource.
#[derive(Resource)]
pub struct OrbitCamera {
    /// Horizontal angle (radians).
    pub theta: f32,
    /// Vertical angle (radians, 0 = top, PI = bottom).
    pub phi: f32,
    /// Distance from target. In the bound mode this only places the camera —
    /// an orthographic projection draws the same size whatever the distance —
    /// so it stays put there and the scale does the zooming instead.
    pub radius: f32,
    /// Look-at target.
    pub target: Vec3,
    pub mode: CameraMode,
    /// How many pixels one cell is drawn at: the user's scale setting, in the
    /// same sense and for the same reason as a font size in a text editor.
    ///
    /// It feeds `OrthographicProjection::scale` as world units per pixel, so
    /// resizing the window changes *how much* of the graph is visible and never
    /// *how large* it is drawn. That is what keeps the bound mode's legibility
    /// from depending on the frame size.
    pub cell_pixels: f32,
    /// Where the view stands between the two projections: 0 is the bound
    /// mode's oblique-orthographic picture, 1 the free mode's perspective one.
    /// It runs on the same clock as every other camera transition, so the
    /// switch is a movement and not a cut.
    pub blend: f32,
    /// Draw the bound mode with a little convergence instead of none. A
    /// parallel projection is exact but flat; a shallow perspective keeps the
    /// axis mapping and hands the eye back the depth cue it normally gets for
    /// free.
    pub semi_ortho: bool,
    /// The same, as a position between the two, on the same clock.
    pub semi_blend: f32,
    /// The free camera's field of view. Fixed when the free mode takes over,
    /// at whatever makes the caret's plane the size it already was — after
    /// that it belongs to the free camera and nothing recomputes it.
    pub free_fov: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            theta: RESET_THETA,
            phi: RESET_PHI,
            radius: RESET_RADIUS,
            target: Vec3::new(0.0, 0.0, 0.0),
            mode: CameraMode::Bound,
            cell_pixels: DEFAULT_CELL_PIXELS,
            blend: 0.0,
            semi_ortho: false,
            semi_blend: 0.0,
            free_fov: FREE_FOV,
        }
    }
}

/// Marker component for the orbit-controlled camera entity.
#[derive(Component)]
pub struct OrbitCameraTag;

/// Duration of a camera transition, in seconds. Short enough not to be
/// experienced as waiting, long enough to be followed — nothing about the view
/// is allowed to change abruptly, so every framing change runs through this.
pub const CAMERA_TWEEN_DURATION: f32 = 0.4;

/// The bound mode's orientation: straight down the conjunction axis, level.
/// Not a matter of taste — this is what puts evaluation order (layout +Z)
/// exactly left to right across the screen and the disjunction axis (+Y)
/// exactly downward. The obliqueness that shows more than one face of a body at
/// once, and puts greater X nearer the viewer, comes from the projection's
/// shear rather than from turning the camera; see `ObliqueOrthographic` for why
/// it has to.
pub const RESET_THETA: f32 = std::f32::consts::FRAC_PI_2;
pub const RESET_PHI: f32 = std::f32::consts::FRAC_PI_2;
/// Distance the camera keeps in the bound mode. Fixed, because under the
/// orthographic projection it changes nothing about apparent size — it only
/// decides how the distance fog grades the depth.
pub const RESET_RADIUS: f32 = 20.0;

/// Default and range of the scale setting, in pixels per cell.
pub const DEFAULT_CELL_PIXELS: f32 = 40.0;
/// The lower bound is not comfort but geometry: the base grid fades to nothing
/// 34 cells from the origin, and an orthographic view zoomed out past that
/// would show the fade end as a hard disc edge, which perspective used to hide.
const MIN_CELL_PIXELS: f32 = 20.0;
const MAX_CELL_PIXELS: f32 = 160.0;

/// How far the view box reaches either side of the caret's plane.
///
/// Measured from that plane rather than from the camera, because the camera's
/// standoff follows the zoom: a fixed box would clip the whole graph away as
/// soon as the camera stood further back than the box was deep. Wide enough to
/// swallow the base grid, tight enough that the depth buffer is not spread over
/// a kilometre of nothing — the transparent planes are coincident often enough
/// to care.
///
/// The near side comes out negative whenever the standoff is smaller than this,
/// which is deliberate: an orthographic box has parallel sides, so the half of
/// the base grid behind the camera would otherwise be sliced off along a hard
/// straight line at the camera's own position.
const VIEW_HALF_DEPTH: f32 = 200.0;

/// A field of view to stand in until a real one is worked out: Bevy's default
/// 45°. Both places that use it derive their own within a frame — the free
/// camera when it takes over, the semi-orthographic one on every viewport
/// change — so this is never what is actually drawn.
const FREE_FOV: f32 = std::f32::consts::FRAC_PI_4;

/// How far the bound camera stands back, as a multiple of the world height the
/// window shows.
///
/// It is the strength of the semi-orthographic perspective — the further away,
/// the flatter — and four and a half is about a thirteen-degree field of view:
/// enough convergence to read depth by, far too little to bend the picture. The
/// parallel projection stands at exactly the same distance even though nothing
/// about it depends on distance, and that is the point: the two projections
/// then differ *only* in whether they converge, so blending between them is a
/// straight fade instead of a fade plus a move.
const SEMI_ORTHO_DEPTH_RATIO: f32 = 4.5;

/// How deep the fog grades, in world units either side of the caret's plane.
const FOG_HALF_SPAN: f32 = 20.0;

#[derive(Clone, Copy, Default)]
struct OrbitState {
    theta: f32,
    phi: f32,
    radius: f32,
    target: Vec3,
}

impl OrbitState {
    fn snapshot(orbit: &OrbitCamera) -> Self {
        Self {
            theta: orbit.theta,
            phi: orbit.phi,
            radius: orbit.radius,
            target: orbit.target,
        }
    }
}

/// Sigmoidal transition for the orbit camera. `from = Some(_)` indicates an
/// active tween.
#[derive(Resource, Default)]
pub struct CameraTween {
    from: Option<OrbitState>,
    to: OrbitState,
    elapsed: f32,
}

impl CameraTween {
    /// Follow the caret to `world_target`, keeping everything else.
    ///
    /// Orientation is the bound mode's guarantee and the scale is the user's
    /// setting, so neither is touched here: the camera travels, it does not
    /// re-frame.
    pub fn focus_on(&mut self, orbit: &OrbitCamera, world_target: Vec3) {
        self.from = Some(OrbitState::snapshot(orbit));
        self.to = OrbitState {
            theta: orbit.theta,
            phi: orbit.phi,
            radius: orbit.radius,
            target: world_target,
        };
        self.elapsed = 0.0;
    }

    /// Move the whole view somewhere else. Used by the mode switch in both
    /// directions, so handing the camera over and taking it back are journeys
    /// rather than cuts.
    pub fn to_view(
        &mut self,
        orbit: &OrbitCamera,
        theta: f32,
        phi: f32,
        radius: f32,
        target: Vec3,
    ) {
        self.from = Some(OrbitState::snapshot(orbit));
        self.to = OrbitState {
            theta,
            phi,
            radius,
            target,
        };
        self.elapsed = 0.0;
    }

    pub fn cancel(&mut self) {
        self.from = None;
    }
}

/// Plugin that registers the orbit camera systems.
pub struct OrbitCameraPlugin;

impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<OrbitCamera>()
            .init_resource::<CameraTween>()
            .add_systems(
                Update,
                (
                    orbit_input,
                    camera_tween_apply,
                    sync_bound_radius,
                    orbit_apply,
                    apply_projection,
                    sync_fog,
                )
                    .chain(),
            );
    }
}

/// Handle mouse input.
///
/// All camera interactions require a held Ctrl key — this keeps
/// left-drag/right-drag/scroll free for grid and node interactions.
///
/// In the bound mode only the scale responds: orbiting and panning would
/// detach the view from the caret, and detaching is what the mode switch is
/// for. Zooming does not detach anything — the scale is a setting the user is
/// meant to keep.
fn orbit_input(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut motion_events: MessageReader<MouseMotion>,
    mut scroll_events: MessageReader<MouseWheel>,
    mut orbit: ResMut<OrbitCamera>,
    mut tween: ResMut<CameraTween>,
    drag: Res<crate::DragState>,
) {
    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);

    // Always drain the event queues so they don't accumulate stale
    // deltas while Ctrl is released.
    let mut total_motion = Vec2::ZERO;
    for ev in motion_events.read() {
        total_motion += ev.delta;
    }

    let mut scroll_delta = 0.0;
    for ev in scroll_events.read() {
        scroll_delta += match ev.unit {
            MouseScrollUnit::Line => ev.y * 1.2,
            MouseScrollUnit::Pixel => ev.y * 0.01,
        };
    }

    if !ctrl {
        return;
    }

    let free = orbit.mode == CameraMode::Free;

    // Any actual mouse-driven camera motion cancels a running transition.
    // Zooming does not, since it leaves the framing alone.
    let has_drag =
        mouse_buttons.pressed(MouseButton::Left) || mouse_buttons.pressed(MouseButton::Right);
    if free && has_drag && total_motion != Vec2::ZERO {
        tween.cancel();
    }

    // Ctrl + left drag → orbit
    if free
        && mouse_buttons.pressed(MouseButton::Left)
        && total_motion != Vec2::ZERO
        && drag.active.is_none()
    {
        orbit.theta -= total_motion.x * 0.007;
        orbit.phi = (orbit.phi - total_motion.y * 0.007).clamp(0.15, std::f32::consts::PI - 0.15);
    }

    // Ctrl + right drag → pan
    if free && mouse_buttons.pressed(MouseButton::Right) && total_motion != Vec2::ZERO {
        let theta = orbit.theta;
        let right = Vec3::new(theta.cos(), 0.0, -theta.sin());
        let up = Vec3::Y;
        orbit.target -= right * total_motion.x * 0.01;
        orbit.target += up * total_motion.y * 0.01;
    }

    // Ctrl + scroll → zoom. Bound mode zooms the scale setting, free mode the
    // distance, because that is what apparent size follows in each projection.
    if scroll_delta != 0.0 {
        if free {
            // Multiplicative here too: the free camera can stand a hundred times
            // further out than it used to, so a fixed step would be unusable at
            // one end of that range and violent at the other.
            orbit.radius = (orbit.radius * 1.1_f32.powf(-scroll_delta))
                .clamp(crate::render::CELL * 0.7, 10_000.0);
        } else {
            // Multiplicative, so a step feels the same size at every zoom.
            orbit.cell_pixels = (orbit.cell_pixels * 1.1_f32.powf(scroll_delta))
                .clamp(MIN_CELL_PIXELS, MAX_CELL_PIXELS);
        }
    }
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Progress an active `CameraTween` and write interpolated values back
/// into `OrbitCamera`.
fn camera_tween_apply(
    time: Res<Time>,
    mut tween: ResMut<CameraTween>,
    mut orbit: ResMut<OrbitCamera>,
) {
    // The projection follows the mode on the same clock as the framing, so both
    // halves of a mode switch arrive together.
    let blend_target = match orbit.mode {
        CameraMode::Bound => 0.0,
        CameraMode::Free => 1.0,
    };
    let step = time.delta_secs() / CAMERA_TWEEN_DURATION;
    if orbit.blend != blend_target {
        orbit.blend = if blend_target > orbit.blend {
            (orbit.blend + step).min(blend_target)
        } else {
            (orbit.blend - step).max(blend_target)
        };
    }
    let semi_target = if orbit.semi_ortho { 1.0 } else { 0.0 };
    if orbit.semi_blend != semi_target {
        orbit.semi_blend = if semi_target > orbit.semi_blend {
            (orbit.semi_blend + step).min(semi_target)
        } else {
            (orbit.semi_blend - step).max(semi_target)
        };
    }

    let Some(from) = tween.from else {
        return;
    };
    tween.elapsed += time.delta_secs();
    let t = (tween.elapsed / CAMERA_TWEEN_DURATION).clamp(0.0, 1.0);
    let e = smoothstep(t);

    orbit.theta = lerp(from.theta, tween.to.theta, e);
    orbit.phi = lerp(from.phi, tween.to.phi, e);
    orbit.radius = lerp(from.radius, tween.to.radius, e);
    orbit.target = from.target.lerp(tween.to.target, e);

    if t >= 1.0 {
        tween.from = None;
    }
}

/// Apply orbit state to camera transform.
fn orbit_apply(orbit: Res<OrbitCamera>, mut query: Query<&mut Transform, With<OrbitCameraTag>>) {
    for mut transform in query.iter_mut() {
        let x = orbit.target.x + orbit.radius * orbit.phi.sin() * orbit.theta.sin();
        let y = orbit.target.y + orbit.radius * orbit.phi.cos();
        let z = orbit.target.z + orbit.radius * orbit.phi.sin() * orbit.theta.cos();

        transform.translation = Vec3::new(x, y, z);
        transform.look_at(orbit.target, Vec3::Y);
    }
}

/// Direction and length in which the depth axis is drawn, as screen units per
/// world unit along it (screen +y is up): down and to the left, and shortened —
/// between the halved depth of a cabinet drawing and the full length of a
/// cavalier one, so the depth reads as depth without shouting over the two axes
/// that are drawn true.
///
/// **Downward is what makes this a view from above.** The angle points along
/// *increasing* X, which is the direction toward the viewer, so a cell's body
/// extends the opposite way: up and to the left. That is where its top face
/// ends up, above the near face — exactly the cabinet drawing of a box seen
/// from above. Pointing this axis upward instead would put the bottom face on
/// show and the whole arrangement would read as seen from below.
///
/// **Leftward is what puts the viewer on the source side.** A body drawn
/// receding up and to the left shows its front, top and *left* faces, which is
/// the picture one gets standing above and to the left of it — at the low end
/// of the evaluation axis, looking along it. Receding to the right would put
/// the viewer at the sink end instead, reading the program from its result
/// backwards.
///
/// The elevation cannot come from the camera here. Tilting it would cost the
/// exact axis mapping (see `ObliqueOrthographic`), so in an oblique projection
/// the *shear direction* is what says whether one looks from above or below —
/// which is also why it makes no difference whether the projection converges.
///
/// Steeper than the 45° of the classic construction, because the evaluation
/// axis runs exactly horizontal and the more the depth axis leans away from it
/// the less the two can be read for one another.
const DEPTH_ANGLE: f32 = std::f32::consts::PI * 295.0 / 180.0;
const DEPTH_FORESHORTENING: f32 = 0.7;

fn depth_shear() -> Vec2 {
    Vec2::new(
        DEPTH_FORESHORTENING * DEPTH_ANGLE.cos(),
        DEPTH_FORESHORTENING * DEPTH_ANGLE.sin(),
    )
}

/// Slide each depth slice sideways before projecting it, measured from the
/// plane the caret is on: `v'x = vx + dx·(vz + d)`.
///
/// The offset is what keeps the caret in the middle of the picture. Shearing
/// from the camera origin instead would displace the whole image by `d` times
/// the shear — at a twenty-unit standoff, half a screen — and it would also
/// make the obliqueness measure depth from the camera rather than from the
/// plane the legibility guarantee is about.
fn shear_matrix(depth: Vec2, focal_distance: f32) -> Mat4 {
    Mat4::from_cols(
        Vec4::X,
        Vec4::Y,
        Vec4::new(depth.x, depth.y, 1.0, 0.0),
        Vec4::new(depth.x * focal_distance, depth.y * focal_distance, 0.0, 1.0),
    )
}

/// What is visible is the projection box pulled back through the shear, so the
/// frustum corners move the other way.
fn unshear_corners(
    corners: [bevy::math::Vec3A; 8],
    depth: Vec2,
    focal_distance: f32,
) -> [bevy::math::Vec3A; 8] {
    corners.map(|corner| {
        let along = corner.z + focal_distance;
        bevy::math::Vec3A::new(
            corner.x - depth.x * along,
            corner.y - depth.y * along,
            corner.z,
        )
    })
}

/// The bound mode's parallel projection: orthographic, then sheared along the
/// depth axis — the oblique projection of technical drawing.
///
/// The shear is what makes the axis mapping exact, and it is not decoration.
/// Under a plain orthographic projection the three axes cannot all be read at
/// once: with the camera's up at world Y, the axes land on screen at
/// `Z ↦ (−sin θ, −cos θ cos φ)` and `X ↦ (cos θ, −sin θ cos φ)`, so asking Z to
/// be exactly horizontal means `cos θ cos φ = 0` — and either choice drives X
/// onto the vertical or onto the horizontal as well, where it collides with one
/// of the other two. Shearing instead points the camera straight down the depth
/// axis, which leaves Y exactly vertical and Z exactly horizontal, and draws X
/// off at an angle. Two axes true, the third oblique: the reason cabinet
/// drawings are built this way.
#[derive(Debug, Clone)]
pub struct ObliqueOrthographic {
    pub ortho: OrthographicProjection,
    /// Screen displacement per world unit along the depth axis.
    pub depth: Vec2,
    /// Distance to the plane the caret is on.
    pub focal_distance: f32,
}

impl bevy::camera::CameraProjection for ObliqueOrthographic {
    fn get_clip_from_view(&self) -> Mat4 {
        self.ortho.get_clip_from_view() * shear_matrix(self.depth, self.focal_distance)
    }

    fn get_clip_from_view_for_sub(&self, sub_view: &bevy::camera::SubCameraView) -> Mat4 {
        self.ortho.get_clip_from_view_for_sub(sub_view)
            * shear_matrix(self.depth, self.focal_distance)
    }

    fn update(&mut self, width: f32, height: f32) {
        self.ortho.update(width, height);
    }

    fn far(&self) -> f32 {
        self.ortho.far()
    }

    fn get_frustum_corners(&self, z_near: f32, z_far: f32) -> [bevy::math::Vec3A; 8] {
        unshear_corners(
            self.ortho.get_frustum_corners(z_near, z_far),
            self.depth,
            self.focal_distance,
        )
    }
}

/// The same picture with a little convergence in it: the identical shear, but
/// over a perspective projection seen from far enough away that the distortion
/// is a depth cue rather than a distortion.
///
/// The field of view is not a setting — it is derived, every time the viewport
/// changes, from how much world the window has to hold at the caret's plane.
/// That is what carries "scale is a setting" over into the perspective: a cell
/// on that plane is `cell_pixels` pixels tall whatever the window does, and
/// only cells in front of or behind it are drawn bigger or smaller.
#[derive(Debug, Clone)]
pub struct ObliquePerspective {
    pub perspective: PerspectiveProjection,
    pub depth: Vec2,
    pub focal_distance: f32,
    pub cell_pixels: f32,
}

impl bevy::camera::CameraProjection for ObliquePerspective {
    fn get_clip_from_view(&self) -> Mat4 {
        self.perspective.get_clip_from_view() * shear_matrix(self.depth, self.focal_distance)
    }

    fn get_clip_from_view_for_sub(&self, sub_view: &bevy::camera::SubCameraView) -> Mat4 {
        self.perspective.get_clip_from_view_for_sub(sub_view)
            * shear_matrix(self.depth, self.focal_distance)
    }

    fn update(&mut self, width: f32, height: f32) {
        self.perspective.aspect_ratio = width / height;
        let visible_world_height = height / self.cell_pixels;
        self.perspective.fov = 2.0 * (visible_world_height * 0.5 / self.focal_distance).atan();
    }

    fn far(&self) -> f32 {
        self.perspective.far()
    }

    fn get_frustum_corners(&self, z_near: f32, z_far: f32) -> [bevy::math::Vec3A; 8] {
        unshear_corners(
            self.perspective.get_frustum_corners(z_near, z_far),
            self.depth,
            self.focal_distance,
        )
    }
}

/// Two projections and a position between them, so a mode switch is a
/// transition rather than a cut.
///
/// Bevy has no way to interpolate one projection kind into another, so this
/// mixes the clip matrices directly. That is not a projection anyone would
/// design — it is a picture on its way from one to the other, which is exactly
/// what it is asked to be, and it is only ever used while `t` is strictly
/// between the two ends.
#[derive(Debug, Clone)]
pub struct BlendedProjection {
    pub from: Projection,
    pub to: Projection,
    pub t: f32,
    /// Distance to the caret's plane, needed to put a parallel matrix and a
    /// projective one on the same footing before mixing them — see
    /// `matched_scale`.
    pub focal_distance: f32,
}

fn lerp_mat4(a: Mat4, b: Mat4, t: f32) -> Mat4 {
    Mat4::from_cols(
        a.x_axis.lerp(b.x_axis, t),
        a.y_axis.lerp(b.y_axis, t),
        a.z_axis.lerp(b.z_axis, t),
        a.w_axis.lerp(b.w_axis, t),
    )
}

/// Put a parallel clip matrix on the same footing as a projective one before
/// mixing them entry by entry.
///
/// A clip matrix is homogeneous: scaling all of it changes nothing at all,
/// because the divide by `w` undoes it again. But that freedom is exactly what
/// wrecks a naive mix. A parallel projection divides by a constant 1; a
/// perspective one divides by the distance, here well over a hundred. Mixing
/// the two entry by entry mixes a 1 with a 120, and the 120 wins almost at
/// once — nine tenths of the visible change happen in the first two percent of
/// the transition, which is why it looked like a jump at one end and a snap at
/// the other.
///
/// Scaling the parallel matrix so that its divisor starts at the same distance
/// the perspective one is measuring makes the two divisors interpolate evenly.
/// The picture at either end is untouched, because scaling never changed it.
fn matched_scale(matrix: Mat4, focal_distance: f32) -> Mat4 {
    if matrix.w_axis.w == 0.0 {
        matrix
    } else {
        matrix * (focal_distance / matrix.w_axis.w)
    }
}

impl bevy::camera::CameraProjection for BlendedProjection {
    fn get_clip_from_view(&self) -> Mat4 {
        lerp_mat4(
            matched_scale(self.from.get_clip_from_view(), self.focal_distance),
            matched_scale(self.to.get_clip_from_view(), self.focal_distance),
            self.t,
        )
    }

    fn get_clip_from_view_for_sub(&self, sub_view: &bevy::camera::SubCameraView) -> Mat4 {
        lerp_mat4(
            matched_scale(
                self.from.get_clip_from_view_for_sub(sub_view),
                self.focal_distance,
            ),
            matched_scale(
                self.to.get_clip_from_view_for_sub(sub_view),
                self.focal_distance,
            ),
            self.t,
        )
    }

    fn update(&mut self, width: f32, height: f32) {
        self.from.update(width, height);
        self.to.update(width, height);
    }

    fn far(&self) -> f32 {
        self.from.far().max(self.to.far())
    }

    fn get_frustum_corners(&self, z_near: f32, z_far: f32) -> [bevy::math::Vec3A; 8] {
        // Culling only needs to be generous, not exact: take whichever end
        // reaches further at each corner.
        let from = self.from.get_frustum_corners(z_near, z_far);
        let to = self.to.get_frustum_corners(z_near, z_far);
        std::array::from_fn(|i| from[i].abs().max(to[i].abs()) * from[i].signum())
    }
}

/// The bound mode's projection at the given scale and standoff.
///
/// `semi` slides between the parallel picture and the shallow-perspective one.
/// Both are built from the *current* standoff, so the caret's plane keeps its
/// size all the way through and the only thing that changes on screen is how
/// much the depth converges.
pub fn bound_projection(cell_pixels: f32, focal_distance: f32, semi: f32) -> Projection {
    let parallel = || {
        Projection::custom(ObliqueOrthographic {
            ortho: OrthographicProjection {
                // World units per pixel: one cell is `cell_pixels` pixels wide,
                // whatever the window does.
                scale: 1.0 / cell_pixels,
                near: focal_distance - VIEW_HALF_DEPTH,
                far: focal_distance + VIEW_HALF_DEPTH,
                ..OrthographicProjection::default_3d()
            },
            depth: depth_shear(),
            focal_distance,
        })
    };
    let converging = || {
        Projection::custom(ObliquePerspective {
            perspective: PerspectiveProjection {
                // `update` derives the field of view from the viewport; this is
                // only what stands there until the first one arrives.
                fov: FREE_FOV,
                near: (focal_distance - VIEW_HALF_DEPTH).max(focal_distance * 0.01),
                far: focal_distance + VIEW_HALF_DEPTH,
                ..default()
            },
            depth: depth_shear(),
            focal_distance,
            cell_pixels,
        })
    };
    if semi <= 0.0 {
        parallel()
    } else if semi >= 1.0 {
        converging()
    } else {
        Projection::custom(BlendedProjection {
            from: parallel(),
            to: converging(),
            t: semi,
            focal_distance,
        })
    }
}

/// The free mode's projection.
pub fn free_projection(fov: f32) -> Projection {
    Projection::Perspective(PerspectiveProjection { fov, ..default() })
}

/// Where the oblique projection implies the viewer stands, as the orbit angles
/// that would put a real camera there.
///
/// A parallel projection has no viewpoint — that is what makes it parallel —
/// but its rays do travel along one direction, and a camera on that line sees
/// very nearly the same picture. Handing the free camera those angles is what
/// makes the switch show its one real difference (that the axes stop being
/// exactly aligned) instead of drowning it in a change of viewpoint.
///
/// In view space the rays run along `(dx, dy, −1)`; the view basis here is
/// right = world −Z, up = world +Y, back = world +X, which turns that into
/// `(−1, dy, −dx)` — and the camera stands the other way along it.
pub fn oblique_view_angles() -> (f32, f32) {
    let depth = depth_shear();
    let toward_camera = Vec3::new(1.0, -depth.y, depth.x).normalize();
    let phi = toward_camera.y.acos();
    let theta = toward_camera.x.atan2(toward_camera.z);
    (theta, phi)
}

/// How far the camera stands back in the bound mode.
///
/// The same distance whether or not the picture converges. Only the
/// semi-orthographic one needs a real distance — that distance *is* its
/// perspective strength, and following the scale setting keeps the amount of
/// convergence the same at every zoom — but the parallel one is indifferent to
/// it, so standing both at the same place costs nothing and buys everything:
/// the two pictures then differ in exactly one respect, and the transition
/// between them is a fade rather than a fade over a moving camera.
pub fn bound_radius(orbit: &OrbitCamera, window_height: f32) -> f32 {
    SEMI_ORTHO_DEPTH_RATIO * window_height / orbit.cell_pixels
}

/// Hold the bound camera at the standoff its projection needs.
///
/// Written straight rather than tweened: under both bound projections the
/// caret's plane keeps its size whatever the standoff, so a change here is
/// invisible — what the eye follows is the projection blend, and that is on the
/// clock. The free camera owns its own distance and is left alone.
fn sync_bound_radius(mut orbit: ResMut<OrbitCamera>, windows: Query<&Window>) {
    // Only once the mode change has finished arriving. While it is still
    // blending, the free half of the picture is built from this same distance,
    // so moving it here would zoom the scene in the middle of the transition —
    // the tween that started the switch owns it until then.
    if orbit.mode != CameraMode::Bound || orbit.blend > 0.0 {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let wanted = bound_radius(&orbit, window.height());
    if orbit.radius != wanted {
        orbit.radius = wanted;
    }
}

/// Keep the projection in step with the mode, the scale and the standoff.
fn apply_projection(
    orbit: Res<OrbitCamera>,
    mut query: Query<&mut Projection, With<OrbitCameraTag>>,
) {
    // Taking `&mut` on a `Mut` marks the component changed, and a projection
    // marked changed every frame makes Bevy rebuild the clip matrix every frame
    // for nothing. The camera resource is written exactly when something moves,
    // so following it is both cheaper and easier to reason about.
    if !orbit.is_changed() {
        return;
    }
    let blend = smoothstep(orbit.blend);
    let semi = smoothstep(orbit.semi_blend);
    for mut projection in query.iter_mut() {
        *projection = if blend <= 0.0 {
            bound_projection(orbit.cell_pixels, orbit.radius, semi)
        } else if blend >= 1.0 {
            free_projection(orbit.free_fov)
        } else {
            Projection::custom(BlendedProjection {
                from: bound_projection(orbit.cell_pixels, orbit.radius, semi),
                to: free_projection(orbit.free_fov),
                t: blend,
                focal_distance: orbit.radius,
            })
        };
    }
}

/// Grade the fog by depth around the caret's plane rather than by distance to
/// the camera.
///
/// The camera's standoff is an implementation detail — it is twenty units in
/// the parallel picture and ten times that in the semi-orthographic one — and a
/// fog measured from the camera would be a different fog in each. Anchoring it
/// to the focal plane keeps the haze telling the one thing it is here to tell:
/// how far a cell is in front of or behind the one the caret is on.
fn sync_fog(orbit: Res<OrbitCamera>, mut query: Query<&mut DistanceFog, With<OrbitCameraTag>>) {
    if !orbit.is_changed() {
        return;
    }
    for mut fog in query.iter_mut() {
        fog.falloff = FogFalloff::Linear {
            start: (orbit.radius - FOG_HALF_SPAN).max(0.0),
            end: orbit.radius + FOG_HALF_SPAN,
        };
    }
}
