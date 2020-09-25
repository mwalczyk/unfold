mod goal_mesh;
mod half_edge;
mod ids;
mod iterators;

use std::path::Path;

use crate::goal_mesh::GoalMesh;
use crate::half_edge::HalfEdgeMesh;

use bevy::prelude::*;
use bevy::render::mesh::VertexAttribute;
use bevy::render::pass::ClearColor;
use bevy::render::pipeline::PrimitiveTopology;

const W: u32 = 400;
const H: u32 = 400;

fn main() {
    App::build()
        .add_resource(WindowDescriptor {
            width: W,
            height: H,
            title: String::from("durer"),
            ..Default::default()
        })
        .add_resource(ClearColor(Color::rgb(1.0, 1.0, 1.0)))
        .add_resource(Msaa { samples: 4 })
        .add_default_plugins()
        .add_startup_system(setup.system())
        .run();
}

/// Find the minimum and maximum x- and y-coordinates of a list of vertices.
fn find_extents(points: &Vec<Vec3>) -> (f32, f32) {
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

fn find_centroid(points: &Vec<Vec3>) -> Vec3 {
    let mut centroid = Vec3::zero();
    for point in points.iter() {
        centroid += *point;
    }
    centroid / points.len() as f32
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let path_to_file = Path::new("goal_meshes/cube.obj");
    let mut goal_mesh = GoalMesh::from_obj(path_to_file, 0.into());
    let mut unfolded_positions = goal_mesh.unfold();

    let (net_size_x, net_size_y) = find_extents(&unfolded_positions);
    let padding = 60.0;
    let net_center = find_centroid(&unfolded_positions);
    let net_scale = (W as f32 - padding) / net_size_x.max(net_size_y);
    println!("Net size: {:?} x {:?}", net_size_x, net_size_y);
    println!("Net center: {:?}", net_center);

    for point in unfolded_positions.iter_mut() {
        *point = (*point - net_center) * net_scale; //+ Vec3::new(50.0, -50.0, 0.0);
    }

    // let mut positions = vec![];
    // for vertex in base_vertices.iter() {
    //     positions.push([vertex.x(), vertex.y(), vertex.z()]);
    // }
    //
    // let mut indices = vec![];
    // for triangle in base_faces.iter() {
    //     for i in 0..triangle.len() {
    //         indices.push(triangle[i] as u32);
    //         indices.push(triangle[(i + 1) % triangle.len()] as u32);
    //     }
    // }

    // let positions = goal_mesh
    //     .gather_lines()
    //     .iter()
    //     .map(|coord| [coord.x(), coord.y(), coord.z()])
    //     .collect::<Vec<_>>();
    // println!("Create mesh buffer with {} vertices", positions.len());
    //
    // let indices = (0..positions.len()).map(|i| i as u32).collect();

    let mut positions = vec![];
    let mut indices = vec![];

    for i in (0..unfolded_positions.len()).step_by(3) {
        let a = unfolded_positions[i + 0];
        let b = unfolded_positions[i + 1];
        let c = unfolded_positions[i + 2];

        positions.push([a.x(), a.y(), a.z()]);
        positions.push([b.x(), b.y(), b.z()]);
        positions.push([c.x(), c.y(), c.z()]);

        indices.push(i as u32 + 0);
        indices.push(i as u32 + 1);

        indices.push(i as u32 + 1);
        indices.push(i as u32 + 2);

        indices.push(i as u32 + 2);
        indices.push(i as u32 + 0);
    }

    let normals = vec![[0.0, 0.0, 0.0]; positions.len()];
    let uvs = vec![[0.0, 0.0]; positions.len()];

    let half_edge_mesh_to_bevy_mesh = Mesh {
        primitive_topology: PrimitiveTopology::LineList,
        attributes: vec![
            VertexAttribute::position(positions),
            VertexAttribute::normal(normals),
            VertexAttribute::uv(uvs),
        ],
        indices: Some(indices),
    };

    // add entities to the world
    commands
        .spawn(PbrComponents {
            mesh: meshes.add(half_edge_mesh_to_bevy_mesh),
            material: materials.add(Color::rgb(0.1, 0.2, 0.1).into()),
            ..Default::default()
        })
        // light
        .spawn(LightComponents {
            translation: Translation::new(4.0, 8.0, 4.0),
            ..Default::default()
        })
        // camera
        .spawn(Camera2dComponents {
            transform: Transform::new_sync_disabled(Mat4::face_toward(
                Vec3::new(0.0, 0.0, 100.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            )),
            translation: Translation::from(net_center),
            ..Default::default()
        });
}
