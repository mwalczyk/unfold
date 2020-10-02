mod color_palette;
mod goal_mesh;
mod half_edge;
mod utils;

use std::path::Path;

use crate::color_palette::ColorPalette;
use crate::goal_mesh::GoalMesh;
use crate::utils::*;

use bevy::prelude::*;
use bevy::render::pass::ClearColor;
use bevy_prototype_lyon::prelude::*;
use clap;
use log::info;
use std::fs::File;

struct InputArgs {
    path_to_obj: String,
    resolution: u32,
    color_palette: ColorPalette,
    wireframe: bool,
}

fn main() {
    // Parse all of the commandline args
    let matches = clap::App::new("Unfold")
        .version("0.1")
        .author("Michael Walczyk")
        .about("📦 A program for unfolding arbitrary convex objects.")
        .short_flag('w')
        .long_flag("wireframe")
        .arg(
            clap::Arg::new("INPUT")
                .about("Sets the input .obj file, i.e. the goal mesh")
                .required(true),
        )
        .arg(
            clap::Arg::new("RESOLUTION")
                .about("Sets the resolution (width and height) of the renderer")
                .short('r')
                .long("resolution")
                .value_name("PIXELS")
                .default_value("1024")
                .takes_value(true),
        )
        .arg(
            clap::Arg::new("COLOR_PALETTE")
                .about("Sets the color palette based on the contents of the provided .json file")
                .short('c')
                .long("color_palette")
                .value_name("COLOR_PALETTE")
                .takes_value(true),
        )
        .arg(
            clap::Arg::new("WIREFRAME")
                .about("Sets the draw mode to wireframe (instead of filled)")
                .short('w')
                .long("wireframe"),
        )
        .get_matches();

    // This arg is required, so we can safely unwrap
    let path_to_obj = matches
        .value_of("INPUT")
        .expect("This parameter should always be provided")
        .to_owned();
    info!("Unfolding .obj: {:?}", path_to_obj);

    let resolution = matches
        .value_of("RESOLUTION")
        .unwrap()
        .parse::<u32>()
        .expect("Invalid resolution");
    info!(
        "Setting resolution to {:?}x{:?} pixels",
        resolution, resolution
    );

    // Parse and construct the color palette (or return the default color palette if none was provided)
    let color_palette = match matches.value_of("COLOR_PALETTE") {
        Some(path) => {
            let json_file_path = Path::new(path);
            let json_file = File::open(json_file_path).expect("File not found");
            let deserialized: ColorPalette =
                serde_json::from_reader(json_file).expect("Error while reading json");
            if deserialized.polygons.len() == 0 {
                panic!("Polygon color array is empty: must provide at least one color");
            }
            deserialized
        }
        _ => {
            // Default color palette
            ColorPalette::new(
                &Vec3::new(1.0, 0.98, 0.98),
                &vec![
                    Vec3::new(0.5568627450980392, 0.792156862745098, 0.9019607843137255),
                    Vec3::new(0.12941176470588237, 0.6196078431372549, 0.7372549019607844),
                    Vec3::new(0.00784313725490196, 0.18823529411764706, 0.2784313725490196),
                    Vec3::new(1.0, 0.7176470588235294, 0.011764705882352941),
                    Vec3::new(0.984313725490196, 0.5215686274509804, 0.0),
                ],
            )
        }
    };

    // Aggregate args
    let input_args = InputArgs {
        path_to_obj,
        resolution,
        color_palette,
        wireframe: matches.is_present("WIREFRAME"),
    };

    App::build()
        .add_resource(WindowDescriptor {
            width: resolution,
            height: resolution,
            title: String::from("unfold"),
            ..Default::default()
        })
        .add_resource(ClearColor(Color::from(
            input_args.color_palette.background.extend(1.0),
        )))
        .add_resource(Msaa { samples: 8 })
        .add_resource(input_args)
        .add_default_plugins()
        .add_startup_system(setup.system())
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    args: Res<InputArgs>,
) {
    // First, construct the goal mesh (and half-edge data structure)
    let mut goal_mesh = GoalMesh::from_obj(&Path::new(&args.path_to_obj[..]), 0.into());
    let mut unfolded_positions = goal_mesh.unfold();

    // Make sure that the unfolded net always fits into the specified canvas size
    // (with PADDING)
    const PADDING: f32 = 100.0;
    let (net_size_x, net_size_y) = find_extents(&unfolded_positions);
    let net_center = find_centroid(&unfolded_positions);
    let net_scale = (args.resolution as f32 - PADDING) / net_size_x.max(net_size_y);
    info!("Net size: {:?} x {:?}", net_size_x, net_size_y);
    info!("Net center: {:?}", net_center);
    for point in unfolded_positions.iter_mut() {
        *point = (*point - net_center) * net_scale;
    }
    debug_assert!(unfolded_positions.len() % 3 == 0);

    // Create materials based on the provided color palette
    let mats = args
        .color_palette
        .polygons
        .iter()
        .map(|color| {
            // Convert SRGB to linear (to compensate for Bevy's internal color system)
            let color = Vec3::new(
                srgb_to_linear(color.x()),
                srgb_to_linear(color.y()),
                srgb_to_linear(color.z()),
            );
            materials.add(Color::rgb(color.x(), color.y(), color.z()).into())
        })
        .collect::<Vec<_>>();

    for triangle_index in 0..unfolded_positions.len() / 3 {
        // Grab the 3 vertices that make up this triangle
        let a = unfolded_positions[triangle_index * 3 + 0];
        let b = unfolded_positions[triangle_index * 3 + 1];
        let c = unfolded_positions[triangle_index * 3 + 2];

        // Select one of the materials to use based on this triangle's index
        let material = mats[triangle_index % mats.len()];

        // Convert the triangle into a polyline primitive
        let shape_type = ShapeType::Polyline {
            points: vec![
                (a.x(), a.y()).into(),
                (b.x(), b.y()).into(),
                (c.x(), c.y()).into(),
            ],
            closed: true,
        };

        let translation = Vec3::zero();

        // Draw either filled or wireframe polygons, based on the provided flag
        if args.wireframe {
            commands.spawn(primitive(
                material,
                &mut meshes,
                shape_type,
                TessellationMode::Stroke(
                    &StrokeOptions::default()
                        .with_line_width(2.0)
                        .with_line_join(LineJoin::Round)
                        .with_line_cap(LineCap::Round),
                ),
                translation,
            ));
        } else {
            commands.spawn(primitive(
                material,
                &mut meshes,
                shape_type,
                TessellationMode::Fill(&FillOptions::default()),
                translation,
            ));
        }
    }

    // Add the camera
    commands.spawn(Camera2dComponents::default());
}
