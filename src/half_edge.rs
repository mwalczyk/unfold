use glam::Vec3;

use crate::ids::{FaceIndex, HalfEdgeIndex, VertexIndex};
use crate::iterators::{FaceEdgeLoop, VertexEdgeLoop};

use core::fmt;

/// Assuming a triangle mesh (i.e. one where all faces are triangles), each face
/// can be represented as a 3-tuple of vertex indices in CCW winding order. The
/// edges of this face, then, are the unique pairs of indices we generate as we traverse
/// the face in a CCW direction.
///
/// For example, say a particular face is specified by the vertex indices `[0, 2, 4]`.
/// This function would return an array of arrays:
///
/// ```
/// [
///     [0, 2],
///     [2, 4],
///     [4, 0],
/// ]
/// ```
fn get_edge_indices_for_face_indices(face_vids: &[usize; 3]) -> [[usize; 2]; 3] {
    // A (triangular) face has 3 unique edges:
    [
        [face_vids[0], face_vids[1]], // 1st edge
        [face_vids[1], face_vids[2]], // 2nd edge
        [face_vids[2], face_vids[0]], // 3rd edge
    ]
}

/// A simple half-edge data structure.
///
/// Note that face indices are assumed to be in CCW winding
/// order.
#[derive(Clone, Copy, Debug)]
pub struct HalfEdge {
    // The ID of the previous half-edge
    prev_id: HalfEdgeIndex,

    // The ID of the next half-edge
    next_id: HalfEdgeIndex,

    // The ID of the half-edge that runs parallel to this one but in the opposite direction
    // (sometimes called the "twin" of the half-edge)
    pair_id: HalfEdgeIndex,

    // The ID of the vertex from which this half-edge originates
    origin_vertex_id: VertexIndex,

    // The ID of the face that is adjacent to this half-edge
    face_id: Option<FaceIndex>,
}

impl HalfEdge {
    pub fn new() -> HalfEdge {
        HalfEdge {
            prev_id: HalfEdgeIndex(0),
            next_id: HalfEdgeIndex(0),
            pair_id: HalfEdgeIndex(0),
            origin_vertex_id: VertexIndex(0),
            face_id: None,
        }
    }

    pub fn prev(&self) -> HalfEdgeIndex {
        self.prev_id
    }

    pub fn next(&self) -> HalfEdgeIndex {
        self.next_id
    }

    pub fn pair(&self) -> HalfEdgeIndex {
        self.pair_id
    }

    pub fn origin_vertex(&self) -> VertexIndex {
        self.origin_vertex_id
    }

    pub fn face(&self) -> Option<FaceIndex> {
        self.face_id
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    // The XYZ-coordinates of this vertex, in 3-space
    coordinates: Vec3,

    // The ID of the half-edge that this vertex "belongs" to (i.e. is the origin vertex of)
    half_edge_id: HalfEdgeIndex,
}

impl Vertex {
    pub fn new(coordinates: Vec3, half_edge_id: HalfEdgeIndex) -> Vertex {
        Vertex {
            coordinates,
            half_edge_id,
        }
    }

    pub fn coordinates(&self) -> &Vec3 {
        &self.coordinates
    }

    pub fn half_edge(&self) -> HalfEdgeIndex {
        self.half_edge_id
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Face {
    // The ID of one of the half-edges that surround this face
    half_edge_id: HalfEdgeIndex,
}

impl Face {
    pub fn new(half_edge_id: HalfEdgeIndex) -> Face {
        Face { half_edge_id }
    }

    pub fn half_edge(&self) -> HalfEdgeIndex {
        self.half_edge_id
    }
}

#[derive(Clone)]
pub struct HalfEdgeMesh {
    // The half-edges of this mesh
    half_edges: Vec<HalfEdge>,

    // The vertices of this mesh
    vertices: Vec<Vertex>,

    // The (triangular) faces of this mesh
    faces: Vec<Face>,
}

impl HalfEdgeMesh {
    /// Returns a new, empty half-edge mesh.
    pub fn empty() -> HalfEdgeMesh {
        HalfEdgeMesh {
            half_edges: Vec::new(),
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Build a list of half-edges from the given face description. Currently, half-edge meshes
    /// can only represent triangular meshes, although this restriction can certainly be lifted
    /// in the future.
    pub fn from_faces(
        base_faces: &Vec<[usize; 3]>,
        base_vertices: &Vec<Vec3>,
    ) -> Result<HalfEdgeMesh, &'static str> {
        println!(
            "Building half-edges from {} faces and {} vertices",
            base_faces.len(),
            base_vertices.len()
        );
        let mut half_edges = vec![];
        let mut vertices = vec![];
        let mut faces = vec![];

        for (fid, f_vids) in base_faces.iter().enumerate() {
            println!(
                "Examining face ID {:?} with vertex indices {:?}",
                fid, f_vids
            );

            for (eid, e_vids) in get_edge_indices_for_face_indices(f_vids).iter().enumerate() {
                // Create a new half-edge for each existing edge of this face (there should always be 3)
                println!("\tExamining edge with vertex indices {:?}", e_vids);
                if eid > 2 {
                    return Err("Found face with more than 3 edges: triangulate this mesh before continuing");
                }

                // `fid` ranges from 0 -> # of faces in the mesh
                // `eid` loops between 0, 1, 2 (assuming triangle faces)
                //
                // `curr_id` is the index of the new half-edge in the list of half-edges (could
                // also be calculated as: `half_edges.len()`)
                let f_offset = fid * 3;

                let curr_id = eid + f_offset;
                assert_eq!(curr_id, half_edges.len());

                let prev_id = if eid == 0 {
                    2 + f_offset
                } else {
                    (eid - 1) + f_offset
                };

                let next_id = (eid + 1) % 3 + f_offset;

                // CCW winding order show below:
                //
                // 0
                // |\
                // | \
                // |  \
                // 1---2
                let mut he = HalfEdge::new();
                he.prev_id = HalfEdgeIndex(prev_id);
                he.next_id = HalfEdgeIndex(next_id);
                he.pair_id = HalfEdgeIndex(0); // This will be set below
                he.origin_vertex_id = VertexIndex(e_vids[0]);
                he.face_id = Some(FaceIndex(fid));

                // Add a new, tracked face to the mesh (only do this once per set of 3 half-edges)
                if eid == 0 {
                    faces.push(Face::new(HalfEdgeIndex(curr_id)));
                }

                // Finally, add this half-edge to the mesh
                half_edges.push(he);
            }
        }

        // Create vertices: we do this separately so that each vertex is only added once
        for vid in (0..base_vertices.len()).map(|i| VertexIndex(i)) {
            for (eid, edge) in half_edges.iter().enumerate() {
                // Is this a half-edge that originates from this vertex?
                if vid == edge.origin_vertex_id {
                    vertices.push(Vertex::new(
                        base_vertices[edge.origin_vertex_id.0],
                        HalfEdgeIndex(eid),
                    ));
                    // Stop after this vertex has been connected to a half-edge
                    break;
                }
            }
        }
        // Some quick sanity checks
        assert_eq!(half_edges.len(), base_faces.len() * 3);
        assert_eq!(vertices.len(), base_vertices.len());
        assert_eq!(faces.len(), base_faces.len());

        // Now, find each half-edge's pair (or "twin" / "opposite"): if one is not found, this means
        // that the half-edge is on the border (i.e. boundary) of the mesh, and a new, "dummy" half-edge
        // will need to be created alongside it
        let mut border_edges = vec![];
        for i in 0..half_edges.len() {
            let mut found_pair = false;

            for j in 0..half_edges.len() {
                if i != j {
                    if half_edges[i].origin_vertex_id
                        == half_edges[half_edges[j].next_id.0].origin_vertex_id
                        && half_edges[half_edges[i].next_id.0].origin_vertex_id
                            == half_edges[j].origin_vertex_id
                    {
                        // The vertex from which this half-edge originates from is the same as the other one's next
                        // (the conditions above should uniquely identify this edge)
                        half_edges[i].pair_id = HalfEdgeIndex(j);
                        found_pair = true;
                        break;
                    }
                }
            }

            if !found_pair {
                let mut border_edge = HalfEdge::new();
                border_edge.prev_id = HalfEdgeIndex(0); // This will be set later
                border_edge.next_id = HalfEdgeIndex(0); // This will be set later
                border_edge.pair_id = HalfEdgeIndex(i);
                border_edge.origin_vertex_id = half_edges[half_edges[i].next_id.0].origin_vertex_id;
                border_edge.face_id = None;

                // Set the half-edge's pair to the newly created border half-edge
                half_edges[i].pair_id = HalfEdgeIndex(half_edges.len() + border_edges.len());

                border_edges.push(border_edge);
            }
        }
        println!("\n{} border edges found in total\n", border_edges.len());

        // Now, assign next / previous pointers for the newly created half-edges along the border
        for i in 0..border_edges.len() {
            let mut found_next = false;
            let mut found_prev = false;

            for j in 0..border_edges.len() {
                if i != j {
                    if half_edges[border_edges[i].pair_id.0].origin_vertex_id
                        == border_edges[j].origin_vertex_id
                    {
                        // The vertex from which this half-edge's pair originates from is the same as the other one
                        border_edges[i].next_id = HalfEdgeIndex(half_edges.len() + j);
                        found_next = true;
                    } else if border_edges[i].origin_vertex_id
                        == half_edges[border_edges[j].pair_id.0].origin_vertex_id
                    {
                        // The vertex from which this half-edge originates from is the same as the other one's pair
                        border_edges[i].prev_id = HalfEdgeIndex(half_edges.len() + j);
                        found_prev = true;
                    }
                }
            }
            if !found_next || !found_prev {
                return Err("Couldn't find next (or maybe, previous) half-edge corresponding to one or more border half-edges");
            }
        }
        half_edges.extend_from_slice(&border_edges);

        let mut hem = HalfEdgeMesh {
            half_edges,
            vertices,
            faces,
        };

        // for fid in hem.face_id_iter() {
        //     println!("HEM face ID {:?} has half-edges:", fid);
        //     for eid in hem.adjacent_half_edges_to_face(fid) {
        //         let vids = hem.adjacent_vertices_to_half_edge(eid);
        //         println!("\tEdge from {:?} to {:?}", vids[0], vids[1]);
        //     }
        // }

        Ok(hem)
    }

    /// Returns an immutable reference to all of the half-edges that make up this mesh.
    pub fn half_edges(&self) -> &Vec<HalfEdge> {
        &self.half_edges
    }

    /// Returns an immutable reference to all of the vertices that make up this mesh.
    pub fn vertices(&self) -> &Vec<Vertex> {
        &self.vertices
    }

    /// Returns an immutable reference to all of the faces that make up this mesh.
    pub fn faces(&self) -> &Vec<Face> {
        &self.faces
    }

    /// Returns an immutable reference to the half-edge at `eid`.
    pub fn half_edge(&self, eid: HalfEdgeIndex) -> &HalfEdge {
        &self.half_edges[eid.0]
    }

    /// Returns a mutable reference to the half-edge at `eid`.
    pub fn half_edge_mut(&mut self, eid: HalfEdgeIndex) -> &mut HalfEdge {
        &mut self.half_edges[eid.0]
    }

    /// Returns an immutable reference to the vertex at `vid`.
    pub fn vertex(&self, vid: VertexIndex) -> &Vertex {
        &self.vertices[vid.0]
    }

    /// Returns a mutable reference to the vertex at `vid`.
    pub fn vertex_mut(&mut self, vid: VertexIndex) -> &mut Vertex {
        &mut self.vertices[vid.0]
    }

    /// Returns an immutable reference to the face at `fid`.
    pub fn face(&self, fid: FaceIndex) -> &Face {
        &self.faces[fid.0]
    }

    /// Returns a mutable reference to the face at `fid`.
    pub fn face_mut(&mut self, fid: FaceIndex) -> &mut Face {
        &mut self.faces[fid.0]
    }

    /// Returns `true` if the half-edge at `eid` is along the border (i.e. is a "dummy"
    /// half-edge) and `false` otherwise.
    pub fn is_border_half_edge(&self, eid: HalfEdgeIndex) -> bool {
        self.half_edge(eid).face_id.is_none()
    }

    /// Returns a list of all of the indices of the half-edges along the border of the mesh.
    pub fn boundary_edges(&self) -> Vec<HalfEdgeIndex> {
        self.half_edge_id_iter()
            .filter(|&eid| self.is_border_half_edge(eid))
            .collect()
    }

    /// Returns a list of all of the indices of the vertices along the border of the mesh.
    pub fn boundary_vertices(&self) -> Vec<VertexIndex> {
        self.boundary_edges()
            .iter()
            .map(|&eid| self.half_edge(eid).origin_vertex_id)
            .collect()
    }

    /// Returns the degree / valency of the specified vertex, i.e. the numbering of outgoing
    /// edges.
    pub fn vertex_degree(&self, vid: VertexIndex) -> usize {
        self.adjacent_half_edges_to_vertex(vid)
            .collect::<Vec<_>>()
            .len()
    }

    /// Returns the number of sides (edges) of the specified face.
    pub fn face_sides(&self, fid: FaceIndex) -> usize {
        self.adjacent_half_edges_to_face(fid)
            .collect::<Vec<_>>()
            .len()
    }

    /// Returns a direction vector (un-normalized) along the specified half-edge. The vector will
    /// always be oriented such that it points from the half-edge's origin vertex towards the
    /// half-edge's "destination" vertex (edge -> next -> origin).
    pub fn edge_vector(&self, eid: HalfEdgeIndex) -> Vec3 {
        let coordinates = self
            .adjacent_vertices_to_half_edge(eid)
            .to_vec()
            .iter()
            .map(|&vid| self.vertex(vid).coordinates)
            .collect::<Vec<_>>();
        let (origin, destination) = (coordinates[0], coordinates[1]);
        destination - origin
    }

    /// Returns the corner angles of all of the faces that surround the specified vertex.
    /// These are guaranteed to be in *CCW order*.
    ///
    /// Note that if a corner angle is obtuse (will this ever happen?), the smaller angle
    /// will be returned. This tends to happen when a vertex is along the border (i.e. the
    /// angle calculated between the two "neighboring" border edges might be obtuse).
    pub fn face_corner_angles(&self, vid: VertexIndex) -> Vec<f32> {
        // First, grab all of the indices of the half-edges that surround this vertex
        let adjacent_eids = self.adjacent_half_edges_to_vertex(vid).collect::<Vec<_>>();
        let mut corner_angles = vec![];

        for i in 0..adjacent_eids.len() {
            // These will always be oriented away from the vertex in question, so we don't have to
            // worry about flipping the sign of either vector here
            let mut curr = self.edge_vector(adjacent_eids[(i + 0) % adjacent_eids.len()]);
            let mut next = self.edge_vector(adjacent_eids[(i + 1) % adjacent_eids.len()]);
            curr = curr.normalize();
            next = next.normalize();

            let phi = curr.dot(next).acos();
            corner_angles.push(phi);
        }

        corner_angles
    }

    /// Returns the midpoint of the specified half-edge.
    pub fn edge_midpoint(&self, eid: HalfEdgeIndex) -> Vec3 {
        let mut sum = Vec3::zero();
        for vid in self.adjacent_vertices_to_half_edge(eid).iter() {
            sum += self.vertex(*vid).coordinates;
        }
        sum * 0.5
    }

    /// Returns the center of the specified face.
    pub fn face_center(&self, fid: FaceIndex) -> Vec3 {
        let mut sum = Vec3::zero();
        let mut count = 0;
        for vid in self.adjacent_vertices_to_face(fid) {
            sum += self.vertex(vid).coordinates;
            count += 1;
        }
        sum / (count as f32)
    }

    /// TODO
    pub fn edge_normal(&self) {}

    /// Returns the normal of the specified vertex.
    pub fn vertex_normal(&self, vid: VertexIndex) -> Vec3 {
        let mut sum = Vec3::zero();
        for fid in self.adjacent_faces_to_vertex(vid) {
            // Ignore invalid faces
            if let Some(fid) = fid {
                sum += self.face_normal(fid);
            }
        }
        sum = sum.normalize();
        sum
    }

    /// Returns the normal of the specified face.
    pub fn face_normal(&self, fid: FaceIndex) -> Vec3 {
        // The first 3 vertex coordinates (guaranteed to be in CCW order) around the face
        let triangle = self
            .adjacent_vertices_to_face(fid)
            .take(3)
            .map(|vid| self.vertex(vid).coordinates)
            .collect::<Vec<_>>();

        let v0 = triangle[0];
        let v1 = triangle[1];
        let v2 = triangle[2];

        // The normal is: (v1 - v0) x (v2 - v0)
        let mut normal = (v1 - v0).cross(v2 - v0);
        normal = normal.normalize();
        normal
    }

    /// TODO: probably need a different `Edge` struct to find unique edges
    pub fn euler_characteristic(&self) {
        // V - E + F
    }

    /// Returns the index of the half-edge that joins the vertices at `a` and `b` or `None`
    /// if such a half-edge doesn't exist.
    pub fn find_half_edge_between_vertices(
        &self,
        a: VertexIndex,
        b: VertexIndex,
    ) -> Option<HalfEdgeIndex> {
        // Is the second vertex at the other end of this half-edge?
        for eid in self.adjacent_half_edges_to_vertex(a) {
            if self.get_terminating_vertex_along_half_edge(eid) == b {
                return Some(eid);
            }
        }

        None
    }

    /// Returns the index of the half-edge that lies between faces `a` and `b` or `None` if such a half-edge
    /// doesn't exist (i.e. the faces are not neighbors). The half-edge returned lies in face `a` (the corresponding
    /// half-edge in face `b` is simply its pair).
    pub fn find_half_edge_between_faces(
        &self,
        a: FaceIndex,
        b: FaceIndex,
    ) -> Option<HalfEdgeIndex> {
        // Look for the half-edge such that edge -> pair -> face is `b`
        for eid in self.adjacent_half_edges_to_face(a) {
            if self.half_edge(self.half_edge(eid).pair_id).face_id == Some(b) {
                return Some(eid);
            }
        }
        // No half-edge was found
        None
    }

    /// Returns `true` if the specified faces are neighbors (i.e. share a common edge) and `false` otherwise.
    pub fn are_faces_neighbors(&self, a: FaceIndex, b: FaceIndex) -> bool {
        self.find_half_edge_between_faces(a, b).is_some()
    }

    /// Returns `true` if the specified face contains the specified half-edge and `false` otherwise.
    pub fn face_contains_half_edge(&self, fid: FaceIndex, eid: HalfEdgeIndex) -> bool {
        self.adjacent_half_edges_to_face(fid)
            .collect::<Vec<_>>()
            .contains(&eid)
    }

    /// Returns `true` if the specified face contains the specified vertex and `false` otherwise.
    pub fn face_contains_vertex(&self, fid: FaceIndex, vid: VertexIndex) -> bool {
        self.adjacent_vertices_to_face(fid)
            .collect::<Vec<_>>()
            .contains(&vid)
    }

    /// Returns the index of the vertex that is opposite to the half-edge at `index`. Assuming all faces
    /// are triangular, this function returns the index of the vertex that is *not* this half-edge's origin
    /// vertex and *not* its terminating vertex (i.e. the vertex at the "end" of the half-edge).
    pub fn vertex_opposite_to_half_edge(&self, eid: HalfEdgeIndex) -> Option<VertexIndex> {
        // The indices of the two vertices that form this edge
        let adjacent_vids = self.adjacent_vertices_to_half_edge(eid);

        // Remember that not all half-edges will actually point to a valid face
        if let Some(fid) = self.half_edge(eid).face_id {
            // Get the indices of the vertices that surround this face
            for other_eid in self.adjacent_half_edges_to_face(fid) {
                // Find the third vertex
                let vid = self.half_edge(other_eid).origin_vertex_id;
                if vid != adjacent_vids[0] && vid != adjacent_vids[1] {
                    return Some(vid);
                }
            }
        }
        None
    }

    pub fn insert_edge_between(
        &mut self,
        from: VertexIndex,
        to: VertexIndex,
    ) -> Option<HalfEdgeIndex> {
        // TODO: https://kaba.hilvi.org/homepage/blog/halfedge/halfedge.htm

        // Don't allow loop edges
        if from == to {
            return None;
        }

        // Don't allow parallel (i.e. duplicate) edges
        if let Some(existing) = self.find_half_edge_between_vertices(from, to) {
            return Some(existing);
        }

        None
    }

    /// Convenience function that constructs an iterator over all half-edge IDs, essentially mapping a
    /// numeric index to a `HalfEdgeIndex` unit struct.
    pub fn half_edge_id_iter(&self) -> impl Iterator<Item = HalfEdgeIndex> {
        (0..self.half_edges.len())
            .into_iter()
            .map(|i| HalfEdgeIndex(i))
    }

    /// Convenience function that constructs an iterator over all face IDs, essentially mapping a
    /// numeric index to a `FaceIndex` unit struct.
    pub fn face_id_iter(&self) -> impl Iterator<Item = FaceIndex> {
        (0..self.faces.len()).into_iter().map(|i| FaceIndex(i))
    }

    /// Convenience function that constructs an iterator over all vertex IDs, essentially mapping a
    /// numeric index to a `VertexIndex` unit struct.
    pub fn vertex_id_iter(&self) -> impl Iterator<Item = VertexIndex> {
        (0..self.vertices.len()).into_iter().map(|i| VertexIndex(i))
    }

    /// Convenience function for: edge -> pair -> origin
    ///
    /// Returns the index of the vertex opposite this half-edge's origin (i.e. where the half-edge
    /// ends).
    pub fn get_terminating_vertex_along_half_edge(&self, eid: HalfEdgeIndex) -> VertexIndex {
        self.half_edge(self.half_edge(eid).pair_id).origin_vertex_id
    }

    /// Convenience function for: (edge -> face, edge -> pair -> face)
    ///
    /// Returns the indices of the two faces that are adjacent to this half-edge. Note
    /// that one of these may be `None`, as is always the case when the half-edge is part
    /// of the boundary of the mesh.
    pub fn adjacent_faces_to_half_edge(&self, eid: HalfEdgeIndex) -> [Option<FaceIndex>; 2] {
        [
            self.half_edge(eid).face_id,
            self.half_edge(self.half_edge(eid).pair_id).face_id,
        ]
    }

    /// Convenience function for: (edge -> origin, edge -> pair -> origin)
    ///
    /// Returns the indices of the two vertices that are adjacent to this half-edge
    /// (i.e. the two vertices that form this particular edge).
    pub fn adjacent_vertices_to_half_edge(&self, eid: HalfEdgeIndex) -> [VertexIndex; 2] {
        [
            self.half_edge(eid).origin_vertex_id,
            self.half_edge(self.half_edge(eid).pair_id).origin_vertex_id,
        ]
    }

    /// Returns an iterator over the half-edges that bound the specified face.
    ///
    /// The half-edge indices returned are guaranteed to be in a counter-clockwise winding order,
    /// but there are no other guarantees about the order of the indices (i.e. they may not be
    /// sorted from lowest to highest index, for example).
    pub fn adjacent_half_edges_to_face(&self, fid: FaceIndex) -> FaceEdgeLoop {
        let start_at = self.face(fid).half_edge_id;
        let end_at = self.half_edge(start_at).prev_id;
        FaceEdgeLoop::new(self, start_at, end_at)
    }

    /// Returns the indices of all of the faces that are neighbors of the specified face.
    /// The faces are guaranteed to be in *CCW order*. Note that if the specified
    /// face is bounded by one (or more) boundary half-edges, `None` will be returned one
    /// or more times during iteration.
    ///
    /// This is a convenience function that provides functionality equivalent to:
    /// ```
    /// let face_loop_iterator = half_edge_mesh
    ///     .adjacent_half_edges_to_face(face_index)
    ///     .map(|eid| mesh.half_edge(mesh.half_edge(eid).pair).face_id);
    /// ```
    pub fn adjacent_faces_to_face(
        &self,
        fid: FaceIndex,
    ) -> impl Iterator<Item = Option<FaceIndex>> + '_ {
        self.adjacent_half_edges_to_face(fid)
            .map(move |eid| self.half_edge(self.half_edge(eid).pair_id).face_id)
    }

    /// Returns the indices of all of the vertices that bound (i.e. surround) the specified face.
    /// The vertices are guaranteed to be in *CCW order*.
    ///
    /// This is a convenience function that provides functionality equivalent to:
    /// ```
    /// let vertex_loop_iterator = half_edge_mesh
    ///     .adjacent_half_edges_to_face(face_index)
    ///     .map(|eid| mesh.half_edge(eid).origin_vertex_id);
    /// ```
    pub fn adjacent_vertices_to_face(
        &self,
        fid: FaceIndex,
    ) -> impl Iterator<Item = VertexIndex> + '_ {
        self.adjacent_half_edges_to_face(fid)
            .map(move |eid| self.half_edge(eid).origin_vertex_id)
    }

    /// Returns an iterator over the half-edges that are adjacent to the specified vertex.
    ///
    /// The half-edge indices returned are in *CCW order* and oriented such that they emanate
    /// (i.e. originate from) the specified vertex.
    pub fn adjacent_half_edges_to_vertex(&self, vid: VertexIndex) -> VertexEdgeLoop {
        let start_at = self.vertex(vid).half_edge_id;
        let end_at = self.half_edge(self.half_edge(start_at).pair_id).next_id;
        VertexEdgeLoop::new(self, start_at, end_at)
    }

    /// Returns an iterator over the faces that are neighbors of the specified vertex.
    /// The faces are guaranteed to be in *CCW order*. Note that if the specified
    /// vertex is part of the boundary of the mesh, one of the faces will be `None`,
    /// indicating the "infinite face" that surrounds the mesh.
    ///
    /// This is a convenience function that provides functionality equivalent to:
    /// ```
    /// let face_loop_iterator = half_edge_mesh
    ///     .adjacent_half_edges_to_vertex(vertex_index)
    ///     .map(|eid| half_edge_mesh.get_half_edge(eid).face_id);
    /// ```
    pub fn adjacent_faces_to_vertex(
        &self,
        vid: VertexIndex,
    ) -> impl Iterator<Item = Option<FaceIndex>> + '_ {
        self.adjacent_half_edges_to_vertex(vid)
            .map(move |eid| self.half_edge(eid).face_id)
    }

    /// Returns an iterator over the vertices that are neighbors to the specified vertex.
    /// The vertices are guaranteed to be in *CCW order*.
    ///
    /// This is a convenience function that provides functionality equivalent to:
    /// ```
    /// let vertex_loop_iterator = half_edge_mesh
    ///     .adjacent_half_edges_to_vertex(vertex_index)
    ///     .map(|eid| half_edge_mesh.get_terminating_vertex_along_half_edge(eid));
    /// ```
    pub fn adjacent_vertices_to_vertex(
        &self,
        vid: VertexIndex,
    ) -> impl Iterator<Item = VertexIndex> + '_ {
        self.adjacent_half_edges_to_vertex(vid)
            .map(move |eid| self.get_terminating_vertex_along_half_edge(eid))
    }
}

impl fmt::Debug for HalfEdgeMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for eid in self.half_edge_id_iter() {
            let he = self.half_edge(eid);
            if let None = he.face_id {
                write!(f, "Half-edge (part of boundary) #{}:\n", eid)?;
            } else {
                write!(f, "Half-edge #{}:\n", eid)?;
            }
            write!(f, "\tStart vertex ID: {:?}\n", he.origin_vertex_id)?;
            write!(
                f,
                "\tEnd vertex ID: {:?}\n",
                self.half_edge(he.pair_id).origin_vertex_id
            )?;
            write!(f, "\tPair half-edge ID: {:?}\n", he.pair_id)?;
            write!(f, "\tPrevious half-edge: {:?}\n", he.prev_id)?;
            write!(f, "\tNext half-edge: {:?}\n", he.next_id)?;

            if let Some(face_id) = he.face_id {
                write!(f, "\tFace ID: {:?}\n", face_id)?;
            } else {
                write!(f, "\tFace: None\n")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_faces() {
        // 0 -- 3
        // | \  |
        // |  \ |
        // 1 -- 2
        let base_faces = vec![
            [0, 1, 2], // 1st triangle
            [0, 2, 3], // 2nd triangle
        ];

        // These don't really matter for testing...
        let base_vertices = vec![
            Vec3::new(0.0, 1.0, 0.0), // Vertex #0
            Vec3::new(0.0, 0.0, 0.0), // Vertex #1
            Vec3::new(1.0, 0.0, 0.0), // Vertex #2
            Vec3::new(1.0, 1.0, 0.0), // Vertex #3
        ];

        // There should be 10 half-edges total: 3 for each interior face (of which there are 2) plus
        // 4 for the border (boundary) half-edges
        let hem = HalfEdgeMesh::from_faces(&base_faces, &base_vertices).unwrap();
        println!("{:?}", hem);

        // Print pre-computed properties
        for face in hem.faces() {
            println!("Face normal vector: {:?}", face.normal);
        }
        for vertex in hem.vertices() {
            println!("Vertex normal vector: {:?}", vertex.normal);
        }

        // First, some basic tests
        assert_eq!(10, hem.half_edges.len());
        assert_eq!(4, hem.vertices.len());
        assert_eq!(2, hem.faces.len());

        // Edge invariants
        for (index, &half_edge) in hem.half_edges().iter().enumerate() {
            let curr = HalfEdgeIndex(index);

            // edge -> next -> prev should equal edge
            assert_eq!(hem.half_edge(hem.half_edge(curr).next_id).prev_id, curr);

            // edge -> prev -> next should equal edge
            assert_eq!(hem.half_edge(hem.half_edge(curr).prev_id).next_id, curr);

            // edge -> next -> face should equal edge -> face
            assert_eq!(
                hem.half_edge(hem.half_edge(curr).next_id).face_id,
                half_edge.face_id
            );

            // edge -> prev -> face should equal edge -> face
            assert_eq!(
                hem.half_edge(hem.half_edge(curr).prev_id).face_id,
                half_edge.face_id
            );

            // edge -> pair -> pair should equal edge
            assert_eq!(hem.half_edge(hem.half_edge(curr).pair_id).pair_id, curr);

            // edge -> pair -> next -> origin should equal edge -> origin
            assert_eq!(
                hem.half_edge(hem.half_edge(hem.half_edge(curr).pair_id).next_id)
                    .origin_vertex_id,
                half_edge.origin_vertex_id
            );
        }

        // Test boundary edges and vertices
        let mut boundary_edges = hem.boundary_edges();
        boundary_edges.sort();
        assert_eq!(boundary_edges, vec![6.into(), 7.into(), 8.into(), 9.into()]);

        let mut boundary_vertices = hem.boundary_vertices();
        boundary_vertices.sort();
        assert_eq!(
            boundary_vertices,
            vec![0.into(), 1.into(), 2.into(), 3.into()]
        );

        // Should be 45, 45, 90
        println!("Face corner angles:");
        println!("{:?}", hem.face_corner_angles(0.into()));

        // Test adjacency query on a particular vertex: note that we sort the array before any
        // assertions, since we don't know what order the indices will be in
        let mut adjacent_to_vertex_0: Vec<_> =
            hem.adjacent_half_edges_to_vertex(0.into()).collect();
        let mut adjacent_to_vertex_1: Vec<_> =
            hem.adjacent_half_edges_to_vertex(1.into()).collect();
        let mut adjacent_to_vertex_2: Vec<_> =
            hem.adjacent_half_edges_to_vertex(2.into()).collect();
        let mut adjacent_to_vertex_3: Vec<_> =
            hem.adjacent_half_edges_to_vertex(3.into()).collect();
        adjacent_to_vertex_0.sort();
        adjacent_to_vertex_1.sort();
        adjacent_to_vertex_2.sort();
        adjacent_to_vertex_3.sort();
        assert_eq!(adjacent_to_vertex_0, vec![0.into(), 3.into(), 9.into()]);
        assert_eq!(adjacent_to_vertex_1, vec![1.into(), 6.into()]);
        assert_eq!(adjacent_to_vertex_2, vec![2.into(), 4.into(), 7.into()]);
        assert_eq!(adjacent_to_vertex_3, vec![5.into(), 8.into()]);
        println!(
            "Half-edges adjacent to vertex #0: {:?}",
            adjacent_to_vertex_0
        );
        println!(
            "Half-edges adjacent to vertex #1: {:?}",
            adjacent_to_vertex_1
        );
        println!(
            "Half-edges adjacent to vertex #2: {:?}",
            adjacent_to_vertex_2
        );
        println!(
            "Half-edges adjacent to vertex #3: {:?}",
            adjacent_to_vertex_3
        );

        // Test adjacency queries on a particular vertex
        let mut neighbors_of_vertex_0: Vec<_> = hem.adjacent_vertices_to_vertex(0.into()).collect();
        let mut neighbors_of_vertex_1: Vec<_> = hem.adjacent_vertices_to_vertex(1.into()).collect();
        let mut neighbors_of_vertex_2: Vec<_> = hem.adjacent_vertices_to_vertex(2.into()).collect();
        let mut neighbors_of_vertex_3: Vec<_> = hem.adjacent_vertices_to_vertex(3.into()).collect();
        neighbors_of_vertex_0.sort();
        neighbors_of_vertex_1.sort();
        neighbors_of_vertex_2.sort();
        neighbors_of_vertex_3.sort();
        assert_eq!(
            vec![VertexIndex(1), VertexIndex(2), VertexIndex(3)],
            neighbors_of_vertex_0
        );
        assert_eq!(vec![VertexIndex(0), VertexIndex(2)], neighbors_of_vertex_1);
        assert_eq!(
            vec![VertexIndex(0), VertexIndex(1), VertexIndex(3)],
            neighbors_of_vertex_2
        );
        assert_eq!(vec![VertexIndex(0), VertexIndex(2)], neighbors_of_vertex_3);

        // Test adjacency queries on a particular face

        let adjacent_to_face_0 = hem
            .adjacent_half_edges_to_face(FaceIndex(0))
            .collect::<Vec<_>>();
        let adjacent_to_face_1 = hem
            .adjacent_half_edges_to_face(FaceIndex(1))
            .collect::<Vec<_>>();
        assert_eq!(
            vec![HalfEdgeIndex(0), HalfEdgeIndex(1), HalfEdgeIndex(2)],
            adjacent_to_face_0
        );
        assert_eq!(
            vec![HalfEdgeIndex(3), HalfEdgeIndex(4), HalfEdgeIndex(5)],
            adjacent_to_face_1
        );

        let vertices_around_face_0 = hem
            .adjacent_vertices_to_face(FaceIndex(0))
            .collect::<Vec<_>>();
        let vertices_around_face_1 = hem
            .adjacent_vertices_to_face(FaceIndex(1))
            .collect::<Vec<_>>();
        assert_eq!(
            vec![VertexIndex(0), VertexIndex(1), VertexIndex(2)],
            vertices_around_face_0
        );
        assert_eq!(
            vec![VertexIndex(0), VertexIndex(2), VertexIndex(3)],
            vertices_around_face_1
        );

        // Test faces along interior edges
        assert_eq!(Some(FaceIndex(0)), hem.half_edge(HalfEdgeIndex(0)).face_id);
        assert_eq!(Some(FaceIndex(0)), hem.half_edge(HalfEdgeIndex(1)).face_id);
        assert_eq!(Some(FaceIndex(0)), hem.half_edge(HalfEdgeIndex(2)).face_id);
        assert_eq!(Some(FaceIndex(1)), hem.half_edge(HalfEdgeIndex(3)).face_id);
        assert_eq!(Some(FaceIndex(1)), hem.half_edge(HalfEdgeIndex(4)).face_id);
        assert_eq!(Some(FaceIndex(1)), hem.half_edge(HalfEdgeIndex(5)).face_id);

        // Test faces along border edges
        assert_eq!(None, hem.half_edge(HalfEdgeIndex(6)).face_id);
        assert_eq!(None, hem.half_edge(HalfEdgeIndex(7)).face_id);
        assert_eq!(None, hem.half_edge(HalfEdgeIndex(8)).face_id);
        assert_eq!(None, hem.half_edge(HalfEdgeIndex(9)).face_id);

        // Test opposite vertices
        assert_eq!(
            Some(VertexIndex(2)),
            hem.vertex_opposite_to_half_edge(HalfEdgeIndex(0))
        );
        assert_eq!(
            Some(VertexIndex(0)),
            hem.vertex_opposite_to_half_edge(HalfEdgeIndex(1))
        );
        assert_eq!(
            Some(VertexIndex(1)),
            hem.vertex_opposite_to_half_edge(HalfEdgeIndex(2))
        );
        assert_eq!(
            Some(VertexIndex(3)),
            hem.vertex_opposite_to_half_edge(HalfEdgeIndex(3))
        );
        assert_eq!(
            Some(VertexIndex(0)),
            hem.vertex_opposite_to_half_edge(HalfEdgeIndex(4))
        );
        assert_eq!(
            Some(VertexIndex(2)),
            hem.vertex_opposite_to_half_edge(HalfEdgeIndex(5))
        );
        assert_eq!(None, hem.vertex_opposite_to_half_edge(6.into()));
        assert_eq!(None, hem.vertex_opposite_to_half_edge(7.into()));
        assert_eq!(None, hem.vertex_opposite_to_half_edge(8.into()));
        assert_eq!(None, hem.vertex_opposite_to_half_edge(9.into()));

        // Test gather methods
        //println!("\n{:?}", half_edge_mesh.gather_triangles());
    }
}
