use crate::half_edge::HalfEdgeMesh;
use crate::ids::{FaceIndex, HalfEdgeIndex, VertexIndex, NO_FACE, NO_HALF_EDGE};

use glam::{Mat3, Mat4, Vec3};
use tobj;

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::ops::{Index, Mul};
use std::path::Path;

/// Calculates the angle that the specified vector makes with the positive x-axis,
/// in the range 0..2π. Note that for the purposes of this function, the z-coordinate
/// of the vector will be ignored.
fn angle_with_e1(v: &Vec3) -> f32 {
    let norm = (v.x() * v.x() + v.y() * v.y()).sqrt();
    let clamped = (v.x() / norm).min(1.0).max(-1.0);

    let psi = if v.y() >= 0.0 {
        clamped.acos()
    } else {
        2.0 * std::f32::consts::PI - clamped.acos()
    };

    psi
}

pub struct GoalMesh {
    // The internal HEM data structure, used for adjacency queries
    half_edge_mesh: HalfEdgeMesh,

    // The index of the face from which the unfolding map will be calculated
    reference_face: FaceIndex,

    // The total number of nodes (i.e. vertices)
    num_nodes: usize,

    // The total number of faces
    num_faces: usize,

    // The number of nodes surrounding each face
    num_nodes_in_face: Vec<usize>,

    face_edges: Vec<Vec<[VertexIndex; 2]>>,

    face_normals: Vec<Vec3>,

    node_normals: Vec<Vec3>,

    node_adjacency_matrix: Vec<Vec<VertexIndex>>,

    came_from: HashMap<FaceIndex, (FaceIndex, HalfEdgeIndex)>,

    branch_faces: Vec<FaceIndex>,

    leaf_faces: Vec<FaceIndex>,
}

impl GoalMesh {
    pub fn from_obj(path_to_file: &Path, reference_face: FaceIndex) -> GoalMesh {
        // Make sure to triangulate the model
        let (models, materials) = tobj::load_obj(&path_to_file, true).expect("Failed to load file");

        if models.len() > 1 {
            println!(
                "The specified .obj file has more than one model - only the first will be used"
            );
        }

        // Containers for storing vertices and faces
        let mut base_vertices = vec![];
        let mut base_faces = vec![];
        let mesh = &models[0].mesh;

        // Parse faces
        println!(
            "Number of triangular faces: {}",
            mesh.num_face_indices.len()
        );
        let mut next_face = 0;
        for f in 0..mesh.num_face_indices.len() {
            let end = next_face + mesh.num_face_indices[f] as usize;
            let face_indices: Vec<_> = mesh.indices[next_face..end].iter().collect();
            assert_eq!(face_indices.len(), 3);

            println!("\tFace[{}] = {:?}", f, face_indices);

            base_faces.push([
                *face_indices[0] as usize,
                *face_indices[1] as usize,
                *face_indices[2] as usize,
            ]);

            next_face = end;
        }

        // Parse vertices
        println!("Number of vertices: {}", mesh.positions.len() / 3);
        assert_eq!(mesh.positions.len() % 3, 0);
        for v in 0..mesh.positions.len() / 3 {
            base_vertices.push(Vec3::new(
                mesh.positions[3 * v + 0],
                mesh.positions[3 * v + 1],
                mesh.positions[3 * v + 2],
            ));
        }

        let mut goal_mesh = GoalMesh {
            half_edge_mesh: HalfEdgeMesh::from_faces(&base_faces, &base_vertices)
                .expect("Failed to create half-edge data structure"),
            reference_face,
            num_nodes: 0,
            num_faces: 0,
            num_nodes_in_face: vec![],
            face_edges: vec![],
            face_normals: vec![],
            node_normals: vec![],
            node_adjacency_matrix: vec![],
            came_from: HashMap::new(),
            branch_faces: vec![],
            leaf_faces: vec![],
        };

        goal_mesh.compute_properties();
        goal_mesh.compute_spanning_tree();
        goal_mesh
    }

    fn compute_properties(&mut self) {
        // The total number of nodes (i.e. vertices)
        self.num_nodes = self.half_edge_mesh.vertices().len();

        // The total number of faces
        self.num_faces = self.half_edge_mesh.faces().len();

        // Make sure that the provided reference face is valid
        assert!(self.reference_face >= 0.into() && self.reference_face < self.num_faces.into());

        // Calculate the number of nodes surrounding each face
        self.num_nodes_in_face = self
            .half_edge_mesh
            .face_id_iter()
            .map(|fid| self.half_edge_mesh.face_sides(fid))
            .collect::<Vec<_>>();

        // Enumerate all of the edges of each face in CCW order (for convenience later on)
        //
        // The k-th edge bounding the j-th face connects the j-th face's k-th and (k+1)-th
        // nodes (in that order) - therefore, the edges calculated below will always be
        // in CCW order
        for fid in self.half_edge_mesh.face_id_iter() {
            let adjacent_vids = self
                .half_edge_mesh
                .adjacent_vertices_to_face(fid)
                .collect::<Vec<_>>();

            let mut edges = vec![];

            for index in 0..adjacent_vids.len() + 1 {
                edges.push([
                    adjacent_vids[(index + 0) % adjacent_vids.len()],
                    adjacent_vids[(index + 1) % adjacent_vids.len()],
                ]);
            }
            self.face_edges.push(edges);
        }

        // Calculate all of the face normals
        self.face_normals = self
            .half_edge_mesh
            .face_id_iter()
            .map(|fid| self.half_edge_mesh.face_normal(fid))
            .collect();

        // Calculate all of the node (vertex) normals
        self.node_normals = self
            .half_edge_mesh
            .vertex_id_iter()
            .map(|vid| self.half_edge_mesh.vertex_normal(vid))
            .collect();

        // Find which nodes are neighbors to which other nodes
        for vid in self.half_edge_mesh.vertex_id_iter() {
            self.node_adjacency_matrix.push(
                self.half_edge_mesh
                    .adjacent_vertices_to_vertex(vid)
                    .collect(),
            );
        }

        // TODO stopped at interior + border edges: may need that additional Edge struct in the
        //  half-edge module to prevent duplicates (i.e. boundary half-edge pairs will be considered
        //  interior half-edges, which isn't really useful for our purposes)
    }

    fn compute_spanning_tree(&mut self) {
        // Calculate which faces are neighbors to each other
        let mut face_neighbors = vec![];

        for fid in self.half_edge_mesh.face_id_iter() {
            // Find all neighbor faces of the face at index `fid`, along with the index of the
            // half-edge that is between them
            println!(
                "Finding neighbors of face with ID: {:?}, which has {:?} neighbors",
                fid,
                self.half_edge_mesh.face_sides(fid)
            );
            let mut neighbors = vec![];

            for neighbor in self.half_edge_mesh.adjacent_faces_to_face(fid) {
                // Ignore "infinite" face outside of the mesh
                if let Some(neighbor) = neighbor {
                    let shared_edge = self
                        .half_edge_mesh
                        .find_half_edge_between_faces(fid, neighbor)
                        .expect("No shared edge found - this should never happen");

                    // let vids = self
                    //     .half_edge_mesh
                    //     .adjacent_vertices_to_half_edge(shared_edge);
                    // println!("\tNeighbor face with ID: {:?}, shares edge that connects vertices {:?} -> {:?}", neighbor, vids[0], vids[1]);

                    neighbors.push((neighbor, shared_edge));
                }
            }

            face_neighbors.push(neighbors);
        }

        println!(
            "{:?}",
            self.half_edge_mesh.half_edge(
                self.half_edge_mesh
                    .find_half_edge_between_faces(0.into(), 6.into())
                    .unwrap()
            )
        );

        // Now, construct the spanning tree
        let mut queue = vec![self.reference_face];
        let mut not_seen_faces = self.half_edge_mesh.face_id_iter().collect::<Vec<_>>();
        println!(
            "Starting spanning tree calculation at face with ID: {:?}",
            self.reference_face
        );

        // A dictionary that maps each face to the face it "comes from" in the spanning tree, along with
        // the edge that is shared between the pair of faces
        self.came_from
            .insert(self.reference_face, (NO_FACE, NO_HALF_EDGE));

        while !queue.is_empty() && !not_seen_faces.is_empty() {
            let curr_face = queue.pop().unwrap();

            for (neighbor, shared_edge) in face_neighbors[curr_face.0].iter() {
                if not_seen_faces.contains(neighbor) && !queue.contains(neighbor) {
                    // Update the spanning tree
                    self.came_from.insert(*neighbor, (curr_face, *shared_edge));
                    queue.push(*neighbor);
                }
            }
            // Remove `curr_face` from the list of not-seen faces, since we just examined its neighbors
            not_seen_faces.retain(|&fid| fid != curr_face);
        }

        // Edges that are crossed by the spanning tree
        let crossed_edges = self
            .came_from
            .values()
            .filter(|(fid, _)| *fid != NO_FACE)
            .map(|(_, eid)| *eid)
            .collect::<Vec<_>>();

        println!("Edges crossed by the spanning tree:");
        for eid in crossed_edges.iter() {
            let vids = self.half_edge_mesh.adjacent_vertices_to_half_edge(*eid);
            println!("\tEdge from {:?} to {:?}", vids[0], vids[1]);
        }
        for (fid, (neighbor, shared_edge)) in self.came_from.iter() {
            // Skip the reference face, whose neighbor index is invalid
            if *neighbor == NO_FACE {
                continue;
            }

            let vids = self
                .half_edge_mesh
                .adjacent_vertices_to_half_edge(*shared_edge);
            println!(
                "Face {:?} came from face {:?} via shared edge from {:?} to {:?}",
                fid, neighbor, vids[0], vids[1]
            );
        }

        // Faces that are in the "middle" of a path along the spanning tree
        self.branch_faces = self
            .came_from
            .values()
            .cloned()
            .filter(|(fid, _)| *fid != NO_FACE)
            .map(|(fid, _)| fid)
            .collect::<Vec<_>>();
        self.branch_faces.sort();
        self.branch_faces.dedup();

        // Faces that are at the "end" of a path along the spanning tree
        self.leaf_faces = self
            .half_edge_mesh
            .face_id_iter()
            .filter(|fid| !self.branch_faces.contains(fid))
            .collect::<Vec<_>>();

        println!("Branch faces:");
        for fid in self.branch_faces.iter() {
            println!("\tFace: {:?}", fid);
        }

        println!("Leaf faces:");
        for fid in self.leaf_faces.iter() {
            println!("\tFace: {:?}", fid);
        }
        assert_eq!(
            self.branch_faces.len() + self.leaf_faces.len(),
            self.half_edge_mesh.faces().len()
        );

        // Debug unfolding paths
        println!("Unfolding paths:");
        for fid in self.half_edge_mesh.face_id_iter() {
            let (faces_along_path, _) = self.get_unfolding_path_to(fid);
            println!(
                "\tPath to face {:?} from the reference face in the spanning tree: {:?}",
                fid, faces_along_path
            );
        }

        // TODO cut boundary and all edges lists

        // Debug incoming + outgoing edges for face 7, which lies between faces 2 and 1 in the spanning tree
        // (i.e. in the order 2, 7, 1)
        let (incoming, outgoing) = self.get_incoming_outgoing_edges(7.into(), 1.into());

        let vids = self.half_edge_mesh.adjacent_vertices_to_half_edge(incoming);
        println!("Incoming edge: {:?} -> {:?}", vids[0], vids[1]);

        if let Some(outgoing) = outgoing {
            let vids = self.half_edge_mesh.adjacent_vertices_to_half_edge(outgoing);
            println!("Outgoing edge: {:?} -> {:?}", vids[0], vids[1]);
        }
    }

    /// A helper function for finding the index of a vertex in the "global" array (i.e. m1, m2, or m3), given
    /// its face ID and vertex ID (from the half-edge data structure). This is necessary because the half-edge
    /// mesh essentially becomes "unwelded" during the unfolding process. In other words, we often end up with
    /// more vertices in the unfolded mesh than we started with. So, while we are calculating m1, m2, and m3, we
    /// need some way to find where a particular vertex belonging to a particular face lies in one or more of
    /// those arrays.
    fn get_global_vertex_index(&self, fid: FaceIndex, vid: VertexIndex) -> usize {
        let index_in_face = self
            .half_edge_mesh
            .adjacent_vertices_to_face(fid)
            .position(|adjacent| adjacent == vid)
            .expect("The specified vertex ID is not part of this face: this should never happen");
        // Assume triangular faces
        usize::from(fid) * 3 + index_in_face
    }

    /// A helper function for finding the properly oriented "version" of a half-edge belonging to a particular face.
    /// The specified half-edge might technically be adjacent to the specified face, but with the wrong orientation
    /// (i.e. we really want edge -> pair, so that the winding order is correct). This function assumes that the
    /// specified half-edge or its pair is, in fact, adjacent to the specified face. Otherwise, an error will be thrown.
    fn get_aligned_half_edge(&self, fid: FaceIndex, eid: HalfEdgeIndex) -> HalfEdgeIndex {
        let mut oriented_edge = eid;

        if !self.half_edge_mesh.face_contains_half_edge(fid, eid) {
            oriented_edge = self.half_edge_mesh.half_edge(oriented_edge).pair();
            assert!(self
                .half_edge_mesh
                .face_contains_half_edge(fid, oriented_edge));
        }

        oriented_edge
    }

    /// Determines the incoming and outgoing edges along the unfolding path connecting
    /// the target face to the reference face. Additionally, we must specify the
    /// index of the face that comes *after* the target face along the desired
    /// unfolding path (this is `towards_face`).
    ///
    /// This is necessary because the target face may be part of multiple unfolding
    /// paths for different leaf faces. In this scenario, the outgoing edge depends
    /// on which of these paths we are considering.
    fn get_incoming_outgoing_edges(
        &self,
        target_face: FaceIndex,
        towards_face: FaceIndex,
    ) -> (HalfEdgeIndex, Option<HalfEdgeIndex>) {
        // Not every face has an outgoing edge (for example, all of the leaf faces in the spanning tree)
        let mut outgoing = None;

        // Every face has an incoming edge - for the reference face, we can just choose one of its
        // edges arbitrarily
        let mut incoming = if target_face == self.reference_face {
            let reference_edge = self
                .half_edge_mesh
                .adjacent_half_edges_to_face(self.reference_face)
                .collect::<Vec<_>>()[0];
            reference_edge
        } else {
            // The target face's incoming edge is simply the shared edge between the target face and the
            // face that it came from in the spanning tree
            self.came_from[&target_face].1
        };

        // Find the outgoing edge: "leaf" faces are at the end of the branches of the spanning tree and thus,
        // don't have an outgoing edge
        if !self.leaf_faces.contains(&target_face) {
            // Search for the face that came from this one and grab the corresponding shared edge -
            // this is the outgoing edge for the target face
            for (node_to, (node_from, shared_edge)) in self.came_from.iter() {
                if *node_from == target_face && *node_to == towards_face {
                    outgoing = Some(*shared_edge);
                }
            }
        }

        // If the incoming edge isn't part of this face, that means that it isn't oriented correctly
        // (i.e. it is CW instead of CCW) - flip it here by grabbing its pair half-edge instead
        incoming = self.get_aligned_half_edge(target_face, incoming);
        assert!(self
            .half_edge_mesh
            .face_contains_half_edge(target_face, incoming));

        // Same as above but for the outgoing edge
        if let Some(maybe_outgoing) = outgoing {
            outgoing = Some(self.get_aligned_half_edge(target_face, maybe_outgoing));
            assert!(self
                .half_edge_mesh
                .face_contains_half_edge(target_face, outgoing.unwrap()));
        }

        (incoming, outgoing)
    }

    /// Returns the indices of the faces along the path from the reference face
    /// to the target face in the spanning tree. Note that the indices will be
    /// ordered in such a way that the target face is the first entry and the
    /// reference face is the last entry.
    fn get_unfolding_path_to(
        &self,
        target_face: FaceIndex,
    ) -> (Vec<FaceIndex>, Vec<HalfEdgeIndex>) {
        let mut faces_along_path = vec![target_face];
        let mut edges_along_path = vec![];

        let mut curr = target_face;

        while curr != self.reference_face {
            let (prev, shared_edge) = self.came_from[&curr];
            faces_along_path.push(prev);
            edges_along_path.push(shared_edge);
            curr = prev;
        }

        // The path will be ordered such that the target face is first and the reference face is last -
        // we want the opposite, so we reverse it below
        faces_along_path.reverse();
        edges_along_path.reverse();

        (faces_along_path, edges_along_path)
    }

    pub fn unfold(&mut self) -> Vec<Vec3> {
        // Basis vectors in R3
        let _e1 = Vec3::unit_x();
        let _e2 = Vec3::unit_y();
        let e3 = Vec3::unit_z();
        let r1_pi = Mat3::from_rotation_x(std::f32::consts::PI);

        // Grab the first edge of the reference face to serve as the "reference edge," i.e. the
        // incoming edge for the reference face
        let reference_edge = self
            .half_edge_mesh
            .adjacent_half_edges_to_face(self.reference_face)
            .collect::<Vec<_>>()[0];

        // (1) Rotating each mesh face to align its unit normal vector with e3
        let mut m1 = vec![];

        for fid in self.half_edge_mesh.face_id_iter() {
            // Precompute items that involve this face's normal vector
            let normal = self.half_edge_mesh.face_normal(fid);
            let normal_cross_e3 = normal.cross(e3);
            let normal_dot_e3 = normal.dot(e3);

            for vid in self.half_edge_mesh.adjacent_vertices_to_face(fid) {
                let coords = *self.half_edge_mesh.vertex(vid).coordinates();

                // If the normal vector is aligned with the negative z-axis, simply flip
                // all of the vertices 180-degrees about the x-axis
                if normal.cmpeq(-e3).all() {
                    let transformed = r1_pi.mul_vec3(coords);
                    m1.push(transformed);
                } else {
                    let transformed = normal_dot_e3 * coords
                        + normal_cross_e3.cross(coords)
                        + (normal_cross_e3 / (1.0 + normal_dot_e3)) * normal_cross_e3.dot(coords);
                    m1.push(transformed);
                }
            }
        }

        // (2) Translate and rotate each mesh face to place one of its nodes at the origin and one of its edges along e1
        let mut m2 = vec![];

        for fid in self.half_edge_mesh.face_id_iter() {
            // Since we only need the incoming edge at the moment, we can just pass an arbitrary face ID
            // as the `towards_face` in the function below
            let (incoming, _) = self.get_incoming_outgoing_edges(fid, NO_FACE);

            // The IDs of the 2 vertices that form the incoming edge of this face
            let incoming_vids = self.half_edge_mesh.adjacent_vertices_to_half_edge(incoming);
            let src = m1[self.get_global_vertex_index(fid, incoming_vids[0])];
            let dst = m1[self.get_global_vertex_index(fid, incoming_vids[1])];
            let along_incoming_edge = dst - src;
            let r3 = Mat3::from_rotation_z(-angle_with_e1(&along_incoming_edge));

            for vid in self.half_edge_mesh.adjacent_vertices_to_face(fid) {
                let coords = m1[self.get_global_vertex_index(fid, vid)];

                // Translate the face so that the "source" (first) vertex of its incoming edge is
                // coincident with the origin
                let translated = coords - src;

                // Perform a rotation around the z-axis that causes the incoming edge of this face
                // (w.r.t. the spanning tree) to be aligned with the positive x-axis
                let rotated = r3.mul_vec3(translated);
                m2.push(rotated);
            }
        }

        // (3) Translate and rotate each mesh face in the e1/e2 plane to its position in the net
        let mut m3 = vec![];

        for fid in self.half_edge_mesh.face_id_iter() {
            // The reference face is already in the correct position
            if fid == self.reference_face {
                for vid in self
                    .half_edge_mesh
                    .adjacent_vertices_to_face(self.reference_face)
                {
                    m3.push(m2[self.get_global_vertex_index(self.reference_face, vid)]);
                }
                continue;
            }

            // Get the unfolding path from the reference face to the target face
            let (path, _) = self.get_unfolding_path_to(fid);

            // Sanity check
            if path.len() < 2 {
                panic!("Unfolding path with 1 (or fewer) entries");
            }

            let mut cumulative_translation = Vec3::zero();
            let mut mu_history = vec![];

            // Go through all of the other faces along this path including the reference face and ignoring the last
            // face, which is the target face itself
            for path_index in 0..(path.len() - 1) {
                // The IDs of the current and next faces along the section of the unfolding path that connects
                // the face `fid` to the reference face
                let fid_curr = path[path_index];
                let fid_next = path[path_index + 1];

                // We should be able to safely unwrap `outgoing` here - if it is `None` something is seriously wrong
                let (incoming, mut maybe_outgoing) =
                    self.get_incoming_outgoing_edges(fid_curr, fid_next);
                let outgoing = maybe_outgoing.expect(
                    "Encountered branch face with no outgoing edge - this should never happen",
                );

                // A pair of vertex IDs for the incoming / outgoing edges
                let incoming_vids = self.half_edge_mesh.adjacent_vertices_to_half_edge(incoming);
                let outgoing_vids = self.half_edge_mesh.adjacent_vertices_to_half_edge(outgoing);

                // Vectors along the incoming and outgoing edges of this face
                let along_incoming = m1[self.get_global_vertex_index(fid_curr, incoming_vids[1])]
                    - m1[self.get_global_vertex_index(fid_curr, incoming_vids[0])];
                let along_outgoing = m1[self.get_global_vertex_index(fid_curr, outgoing_vids[1])]
                    - m1[self.get_global_vertex_index(fid_curr, outgoing_vids[0])];

                // The angle formed between the incoming / outgoing edges
                let mu = angle_with_e1(&-along_outgoing) - angle_with_e1(&along_incoming);

                // Calculate a translation vector
                let rhs = along_outgoing
                    + m1[self.get_global_vertex_index(fid_curr, outgoing_vids[0])]
                    - m1[self.get_global_vertex_index(fid_curr, incoming_vids[0])];
                let offset = Mat3::from_rotation_z(
                    mu_history.iter().sum::<f32>() - angle_with_e1(&along_incoming),
                )
                .mul_vec3(rhs);

                // Accumulate all such translation vectors
                cumulative_translation += offset;

                // Subsequent faces along this fold path will need to take all prior rotations into account
                mu_history.push(mu);
            }

            // Transform all of the points in this face to their final positions in the xy-plane
            let cumulative_rotation = Mat3::from_rotation_z(mu_history.iter().sum());
            for vid in self.half_edge_mesh.adjacent_vertices_to_face(fid) {
                let coords = m2[self.get_global_vertex_index(fid, vid)];
                let rotated = cumulative_rotation.mul_vec3(coords);
                let translated = rotated + cumulative_translation;
                m3.push(translated);
            }
        }

        m3
    }

    /// This function is primarily used to render the half-edge mesh as triangles (as in an OpenGL program).
    pub fn gather_triangles(&self) -> Vec<Vec3> {
        let mut triangles = vec![];

        for fid in self.half_edge_mesh.face_id_iter() {
            let coords = self
                .half_edge_mesh
                .adjacent_vertices_to_face(fid)
                .map(|vid| *self.half_edge_mesh.vertex(vid).coordinates())
                .collect::<Vec<_>>();
            triangles.extend_from_slice(&coords);
        }

        triangles
    }

    /// This function is primarily used to render the half-edge mesh as a wireframe (as in an OpenGL program).
    pub fn gather_lines(&self) -> Vec<Vec3> {
        let mut lines = vec![];

        for eid in self.half_edge_mesh.half_edge_id_iter() {
            if !self.half_edge_mesh.is_border_half_edge(eid) {
                let pair_vids = self.half_edge_mesh.adjacent_vertices_to_half_edge(eid);

                lines.extend_from_slice(&[
                    *self.half_edge_mesh.vertex(pair_vids[0]).coordinates(),
                    *self.half_edge_mesh.vertex(pair_vids[1]).coordinates(),
                ]);
            }
        }

        lines
    }
}
