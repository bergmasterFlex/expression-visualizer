use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
/// Orbit camera controller for 3D scene navigation.
///
/// - Left drag: orbit (rotate around target)
/// - Scroll: zoom in/out
/// - Right drag: pan
/// - Auto-rotates slowly until user interacts
use bevy::prelude::*;

/// Orbit camera state stored as a resource.
#[derive(Resource)]
pub struct OrbitCamera {
    /// Horizontal angle (radians).
    pub theta: f32,
    /// Vertical angle (radians, 0 = top, PI = bottom).
    pub phi: f32,
    /// Distance from target.
    pub radius: f32,
    /// Look-at target.
    pub target: Vec3,
    /// Auto-rotation speed (radians/sec).
    pub auto_speed: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            theta: 1.4,
            phi: 1.0,
            radius: 20.0,
            target: Vec3::new(0.0, 0.0, 0.0),
            auto_speed: 0.15,
        }
    }
}

/// Marker component for the orbit-controlled camera entity.
#[derive(Component)]
pub struct OrbitCameraTag;

/// Duration of the auto-focus camera transition, in seconds.
pub const CAMERA_TWEEN_DURATION: f32 = 0.4;

/// Reset-state the camera tweens toward. Matches the default angles
/// shown while the start menu is open (i.e. `OrbitCamera::default()`).
pub const RESET_THETA: f32 = 1.4;
pub const RESET_PHI: f32 = 1.0;
pub const RESET_RADIUS: f32 = 20.0;

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

/// Sigmoidal auto-focus transition for the orbit camera.
///
/// `from = Some(_)` indicates an active tween. `focus_on` starts a new
/// tween from the current camera state toward the reset defaults with
/// the given world-space look-at target.
#[derive(Resource, Default)]
pub struct CameraTween {
    from: Option<OrbitState>,
    to: OrbitState,
    elapsed: f32,
}

impl CameraTween {
    pub fn focus_on(&mut self, orbit: &OrbitCamera, world_target: Vec3) {
        self.from = Some(OrbitState::snapshot(orbit));
        self.to = OrbitState {
            theta: RESET_THETA,
            phi: RESET_PHI,
            radius: RESET_RADIUS,
            target: world_target,
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
                (orbit_input, camera_tween_apply, orbit_apply).chain(),
            );
    }
}

/// Handle mouse/touch input to update orbit state.
///
/// All camera interactions (orbit, pan, zoom) require a held Ctrl key —
/// this keeps left-drag/right-drag/scroll free for grid and node
/// interactions.
fn orbit_input(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut motion_events: MessageReader<MouseMotion>,
    mut scroll_events: MessageReader<MouseWheel>,
    mut orbit: ResMut<OrbitCamera>,
    mut tween: ResMut<CameraTween>,
    drag: Res<crate::DragState>,
    time: Res<Time>,
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

    // Any actual mouse-driven camera motion cancels a running auto-focus.
    let has_drag =
        mouse_buttons.pressed(MouseButton::Left) || mouse_buttons.pressed(MouseButton::Right);
    let user_input = (has_drag && total_motion != Vec2::ZERO) || scroll_delta != 0.0;
    if user_input {
        tween.cancel();
    }

    // Ctrl + left drag → orbit
    if mouse_buttons.pressed(MouseButton::Left)
        && total_motion != Vec2::ZERO
        && drag.active.is_none()
    {
        orbit.theta -= total_motion.x * 0.007;
        orbit.phi = (orbit.phi - total_motion.y * 0.007).clamp(0.15, std::f32::consts::PI - 0.15);
    }

    // Ctrl + right drag → pan
    if mouse_buttons.pressed(MouseButton::Right) && total_motion != Vec2::ZERO {
        let theta = orbit.theta;
        let right = Vec3::new(theta.cos(), 0.0, -theta.sin());
        let up = Vec3::Y;
        orbit.target -= right * total_motion.x * 0.01;
        orbit.target += up * total_motion.y * 0.01;
    }

    // Ctrl + scroll → zoom
    if scroll_delta != 0.0 {
        orbit.radius = (orbit.radius - scroll_delta).clamp(2.0, 52.5);
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
