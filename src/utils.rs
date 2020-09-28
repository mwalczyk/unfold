use glam::Vec3;

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
