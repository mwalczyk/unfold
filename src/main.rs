mod goal_mesh;
mod gradient;
mod half_edge;
mod ids;
mod iterators;
mod utils;

use std::path::Path;

use crate::goal_mesh::GoalMesh;
use crate::gradient::Gradient;
use crate::half_edge::HalfEdgeMesh;
use crate::utils::*;

use bevy::prelude::*;
use bevy::render::pass::ClearColor;
use bevy_prototype_lyon::prelude::*;

const W: u32 = 1024;
const H: u32 = 1024;

fn main() {
    App::build()
        .add_resource(WindowDescriptor {
            width: W,
            height: H,
            title: String::from("durer"),
            ..Default::default()
        })
        .add_resource(ClearColor(Color::rgb(1.0, 0.98, 0.98)))
        .add_resource(Msaa { samples: 8 })
        .add_default_plugins()
        .add_startup_system(setup.system())
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let path_to_file = Path::new("goal_meshes/dodecahedron2.obj");
    let mut goal_mesh = GoalMesh::from_obj(path_to_file, 100.into());
    let mut unfolded_positions = goal_mesh.unfold();

    let (net_size_x, net_size_y) = find_extents(&unfolded_positions);
    let padding = 100.0;
    let net_center = find_centroid(&unfolded_positions);
    let net_scale = (W as f32 - padding) / net_size_x.max(net_size_y);
    println!("Net size: {:?} x {:?}", net_size_x, net_size_y);
    println!("Net center: {:?}", net_center);

    for point in unfolded_positions.iter_mut() {
        *point = (*point - net_center) * net_scale;
    }

    let gradient = Gradient::linear_spacing(&vec![
        Vec3::new(0.23921568627450981, 0.20392156862745098, 0.5450980392156862),
        Vec3::new(0.4627450980392157, 0.47058823529411764, 0.9294117647058824),
        Vec3::new(0.9686274509803922, 0.7215686274509804, 0.00392156862745098),
        Vec3::new(0.9450980392156862, 0.5294117647058824, 0.00392156862745098),
        Vec3::new(0.9529411764705882, 0.3568627450980392, 0.01568627450980392),
    ]);

    let to_linear = |val: f32| -> f32 {
        if val <= 0.04045 {
            return val / 12.92;
        }
        ((val + 0.055) / 1.055).powf(2.4)
    };

    let colors = vec![
        Vec3::new(0.5568627450980392, 0.792156862745098, 0.9019607843137255),
        Vec3::new(0.12941176470588237, 0.6196078431372549, 0.7372549019607844),
        Vec3::new(0.00784313725490196, 0.18823529411764706, 0.2784313725490196),
        Vec3::new(1.0, 0.7176470588235294, 0.011764705882352941),
        Vec3::new(0.984313725490196, 0.5215686274509804, 0.0),
    ];
    let mats = colors.iter().map(|color| {
        let color = Vec3::new(
            to_linear(color.x()),
            to_linear(color.y()),
            to_linear(color.z())
        );

        materials.add(Color::rgb(color.x(), color.y(), color.z()).into())
    }).collect::<Vec<_>>();

    // let mats = (0..5)
    //     .into_iter()
    //     .map(|i| {
    //         let c1 = colors[i];//gradient.color_at(i as f32 / 5.0);
    //         let c2 = Vec3::new(
    //             to_linear(c1.x()),
    //             to_linear(c1.y()),
    //             to_linear(c1.z())
    //         );
    //
    //         materials.add(Color::rgb(c2.x(), c2.y(), c2.z()).into())
    //     })
    //     .collect::<Vec<_>>();

    for triangle_index in (0..unfolded_positions.len()).step_by(3) {
        let a = unfolded_positions[triangle_index + 0];
        let b = unfolded_positions[triangle_index + 1];
        let c = unfolded_positions[triangle_index + 2];

        commands.spawn(primitive(
            mats[(triangle_index / 3) % mats.len()], // was just 3
            &mut meshes,
            ShapeType::Polyline {
                points: vec![
                    (a.x(), a.y()).into(),
                    (b.x(), b.y()).into(),
                    (c.x(), c.y()).into(),
                ],
                closed: true,
            },
            // TessellationMode::Stroke(&StrokeOptions::default()
            //     .with_line_width(2.0)
            //     .with_line_join(LineJoin::Round)
            //     .with_line_cap(LineCap::Round)
            // ),
            TessellationMode::Fill(&FillOptions::default()),
            Vec3::new(0.0, 0.0, 0.0),
        ));
    }

    // Add the camera
    commands.spawn(Camera2dComponents::default());
}
