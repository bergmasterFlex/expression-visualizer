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
            theta: 0.6,
            phi: 1.0,
            radius: 12.0,
            target: Vec3::new(0.0, 0.0, 0.0),
            auto_speed: 0.15,
        }
    }
}

/// Marker component for the orbit-controlled camera entity.
#[derive(Component)]
pub struct OrbitCameraTag;

/// Plugin that registers the orbit camera systems.
pub struct OrbitCameraPlugin;

impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<OrbitCamera>()
            .add_systems(Update, (orbit_input, orbit_apply).chain());
    }
}

/// Handle mouse/touch input to update orbit state.
fn orbit_input(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_events: EventReader<MouseMotion>,
    mut scroll_events: EventReader<MouseWheel>,
    mut orbit: ResMut<OrbitCamera>,
    drag: Res<crate::DragState>,
    time: Res<Time>,
) {
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

    // Left mouse drag → orbit
    if mouse_buttons.pressed(MouseButton::Left)
        && total_motion != Vec2::ZERO
        && drag.active.is_none()
    {
        orbit.theta -= total_motion.x * 0.007;
        orbit.phi = (orbit.phi - total_motion.y * 0.007).clamp(0.15, std::f32::consts::PI - 0.15);
    }

    // Right mouse drag → pan
    if mouse_buttons.pressed(MouseButton::Right) && total_motion != Vec2::ZERO {
        let theta = orbit.theta;
        let right = Vec3::new(theta.cos(), 0.0, -theta.sin());
        let up = Vec3::Y;
        orbit.target -= right * total_motion.x * 0.01;
        orbit.target += up * total_motion.y * 0.01;
    }

    // Scroll → zoom
    if scroll_delta != 0.0 {
        orbit.radius = (orbit.radius - scroll_delta).clamp(2.0, 35.0);
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
