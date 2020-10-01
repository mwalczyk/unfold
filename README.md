# Unfold
📦 A program for unfolding arbitrary convex objects. 

<p align="center">
  <img src="https://raw.githubusercontent.com/mwalczyk/durer/master/screenshots/screenshot.png" alt="screenshot" width="400" height="auto"/>
</p>

## Description
"Unfold" is an implementation of the paper _Unfolding Polyhedra Method for Design of Origami Structures with Creased Folds_, as described in the book _Active Origami_. The input to the program is a 3D model (in the form of an .obj file), and the output is a rendered image of the unfolded mesh. The resulting pattern often exhibits complex structure, but the algorithm to find such an unfolding is actually quite simple. The goal of this algorithm is to find a [net](https://en.wikipedia.org/wiki/Net_(polyhedron)) (i.e. an unfolding where no faces overlap) for the given _convex_ polyhedron.

First, we construct a spanning tree of the faces of the input mesh (also known as the **goal mesh**). By definition, the spanning tree touches each face of the goal mesh exactly once. To construct a spanning tree, we pick an arbitrary **reference face**. We add each of the reference face's neighbors to a queue. While the queue is not empty, we pop off the last face, add it to the spanning tree, and push each of its unvisited neighbors onto the queue. We continue in this fashion until all faces have been added to the tree. The edges of the mesh that are _not_ crossed by the spanning tree form the **cut boundary**. These are the edges that must be "cut" in order to unfold the mesh.

The faces of the goal mesh are rotated and translated into their positions in the net via a series of transformations:
1. Rotate each face to align its unit normal vector with the positive z-axis ("up" in our coordinate system)
2. Translate and rotate each face to place one of its vertices at the origin and one of its edges along the positive x-axis
3. Translate and rotate each face in the xy-plane to its final position in the net

All of these operations (including the calculation of the spanning tree) require extensive knowledge of how the vertices, edges, and faces of the goal mesh are connected to one another. To facilitate this, the mesh is stored as a half-edge data structure, which makes such adjacency queries fairly easy. There is a lot of existing literature on half-edge data structures, but essentially, each edge of the mesh stores a pair of directed _half-edges_. Every face stores a pointer to one of its half-edges. Every half-edge stores a pointer to its originating vertex, its face, the next half-edge, the previous half-edge, and its pair half-edge. Every vertex stores a pointer to one of its incident half-edges. 

### Limitations

Determining whether _every_ convex polyhedra has a net is still an unsolved problem (known as "Dürer's conjecture").

## Tested On
- Windows 10
- NVIDIA GeForce GTX 1660 Ti
- Rust compiler version `1.45.0`

## To Build
1. Clone this repo.
2. Make sure 🦀 [Rust](https://www.rust-lang.org/en-US/) installed and `cargo` is in your `PATH`.
3. Inside the repo, run: `cargo build --release`.

## To Use
Notes here.

## To Do
- [ ] Add support for non-triangular faces

## Credits
Notes here.

### License
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
