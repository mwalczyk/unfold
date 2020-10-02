use glam::Vec3;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct ColorPalette {
    pub background: Vec3,
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
