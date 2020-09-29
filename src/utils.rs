use glam::{Vec3, Vec2};

/// Calculates the angle that the specified vector makes with the positive x-axis,
/// in the range 0..2π. Note that for the purposes of this function, the z-coordinate
/// of the vector will be ignored.
pub fn angle_with_e1(vector: &Vec3) -> f32 {
    let norm = (vector.x() * vector.x() + vector.y() * vector.y()).sqrt();
    let clamped = (vector.x() / norm).min(1.0).max(-1.0);

    let psi = if vector.y() >= 0.0 {
        clamped.acos()
    } else {
        2.0 * std::f32::consts::PI - clamped.acos()
    };

    psi
}

/// Find the minimum and maximum x- and y-coordinates of a list of vertices.
pub fn find_extents(points: &Vec<Vec3>) -> (f32, f32) {
    let mut min_x = points
        .iter()
        .map(|&v| v.x())
        .fold(f32::INFINITY, |a, b| a.min(b));
    let mut max_x = points
        .iter()
        .map(|&v| v.x())
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut min_y = points
        .iter()
        .map(|&v| v.y())
        .fold(f32::INFINITY, |a, b| a.min(b));
    let mut max_y = points
        .iter()
        .map(|&v| v.y())
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));

    ((max_x - min_x).abs(), (max_y - min_y).abs())
}

pub fn find_centroid(points: &Vec<Vec3>) -> Vec3 {
    let mut centroid = Vec3::zero();
    for point in points.iter() {
        centroid += *point;
    }
    centroid / points.len() as f32
}

pub fn triangle_area_2d(a: &Vec2, b: &Vec2, c: &Vec2) -> f32 {
    (a.x() * (b.y() - c.y()) + b.x() * (c.y() - a.y()) + c.x() * (a.y() * b.y()) / 2.0).abs()
}

pub fn remap(from_range: (f32, f32), to_range: (f32, f32), s: f32) -> f32 {
    to_range.0 + (s - from_range.0) * (to_range.1 - to_range.0) / (from_range.1 - from_range.0)
}

pub fn srgb_to_linear(val: f32) -> f32 {
    if val <= 0.04045 {
        return val / 12.92;
    }
    ((val + 0.055) / 1.055).powf(2.4)
}