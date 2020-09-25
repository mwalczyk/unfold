use crate::half_edge::HalfEdgeMesh;
use crate::ids::HalfEdgeIndex;

/// An iterator for looping over all of the half-edges that bound a particular face.
pub struct FaceEdgeLoop<'l> {
    mesh: &'l HalfEdgeMesh,
    current: HalfEdgeIndex,
    last: HalfEdgeIndex,
    finished: bool,
}

impl<'l> Iterator for FaceEdgeLoop<'l> {
    type Item = HalfEdgeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        let res = self.current;

        // If this flag has been set, we are done with the loop - return `None`
        if self.finished {
            return None;
        }

        // If we've reached the last half-edge in this face, we want to return it
        // then signal to the program that the next iteration should return `None`
        if self.current == self.last {
            self.finished = true;
        }

        // Advance the pointer: edge -> next
        //
        // On the last iteration (i.e. when `current` is equal to the starting edge's
        // `prev`), this will be set to the half-edge that we started with, but it
        // won't be returns twice, since we have set the `finished` flag
        self.current = self.mesh.half_edge(self.current).next();

        Some(res)
    }
}

impl<'l> FaceEdgeLoop<'l> {
    pub fn new(
        mesh: &'l HalfEdgeMesh,
        start_at: HalfEdgeIndex,
        end_at: HalfEdgeIndex,
    ) -> FaceEdgeLoop<'l> {
        FaceEdgeLoop {
            mesh,
            current: start_at,
            last: end_at,
            finished: false,
        }
    }
}

/// An iterator for looping over all of the half-edges that are adjacent to a particular vertex.
pub struct VertexEdgeLoop<'l> {
    mesh: &'l HalfEdgeMesh,
    current: HalfEdgeIndex,
    last: HalfEdgeIndex,
    finished: bool,
}

impl<'l> Iterator for VertexEdgeLoop<'l> {
    type Item = HalfEdgeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        let res = self.current;

        // If this flag has been set, we are done with the loop - return `None`
        if self.finished {
            return None;
        }

        // If we've reached the last half-edge around this vertex, we want to return it
        // then signal to the program that the next iteration should return `None`
        if self.current == self.last {
            self.finished = true;
        }

        // Advance the pointer: edge -> prev -> pair (we want to move in CCW order)
        //
        // `current` will point to the next half-edge that emanates from this vertex
        self.current = self
            .mesh
            .half_edge(self.mesh.half_edge(self.current).prev())
            .pair();

        Some(res)
    }
}

impl<'l> VertexEdgeLoop<'l> {
    pub fn new(
        mesh: &'l HalfEdgeMesh,
        start_at: HalfEdgeIndex,
        end_at: HalfEdgeIndex,
    ) -> VertexEdgeLoop<'l> {
        VertexEdgeLoop {
            mesh,
            current: start_at,
            last: end_at,
            finished: false,
        }
    }
}
