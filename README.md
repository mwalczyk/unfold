# Durer
📦 A program for unfolding arbitrary convex objects. 

<p align="center">
  <img src="https://raw.githubusercontent.com/mwalczyk/durer/master/screenshots/screenshot.png" alt="screenshot" width="400" height="auto"/>
</p>

## Description
Durer is an implementation of the paper _Unfolding Polyhedra Method for Design of Origami Structures with Creased Folds_ as described in the book _Active Origami_. The input to the program is a 3D model (in the form of an .obj file), and the output is a rendered image of the unfolded mesh. The resulting [net](https://en.wikipedia.org/wiki/Net_(polyhedron)) often exhibits complex structure, but the algorithm to find such an unfolding is actually quite simple.

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
