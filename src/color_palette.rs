use glam::Vec3;
use serde::{Deserialize, Serialize};

/// A struct representing a color palette used for rendering an unfolded net.
#[derive(Serialize, Deserialize, Debug)]
pub struct ColorPalette {
    // The background color (RGB)
    pub background: Vec3,

    // A list of colors that will be applied to the faces of the net
    pub polygons: Vec<Vec3>,
}

impl ColorPalette {
    pub fn new(background: &Vec3, polygons: &Vec<Vec3>) -> ColorPalette {
        ColorPalette {
            background: *background,
            polygons: polygons.clone(),
        }
    }
}
