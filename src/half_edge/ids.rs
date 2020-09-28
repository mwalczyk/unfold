/// Define different "types" of indices using the `newtype` pattern.
///
/// Reference: `https://doc.rust-lang.org/1.0.0/style/features/types/newtype.html`
use typed_index_derive::TypedIndex;

use core::fmt;

pub struct HalfEdgeIDs(Vec<HalfEdgeIndex>);
pub struct VertexIDs(Vec<VertexIndex>);
pub struct FaceIDs(Vec<FaceIndex>);

/// An index type corresponding to a half-edge.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, TypedIndex)]
#[typed_index(HalfEdgeIDs)]
pub struct HalfEdgeIndex(pub usize);

/// An index type corresponding to a vertex.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, TypedIndex)]
#[typed_index(VertexIDs)]
pub struct VertexIndex(pub usize);

/// An index type corresponding to a face.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, TypedIndex)]
#[typed_index(FaceIDs)]
pub struct FaceIndex(pub usize);

/// A placeholder for invalid half-edges.
pub const NO_HALF_EDGE: HalfEdgeIndex = HalfEdgeIndex(std::usize::MAX);

/// A placeholder for invalid vertices.
pub const NO_VERTEX: VertexIndex = VertexIndex(std::usize::MAX);

/// A placeholder for invalid faces.
pub const NO_FACE: FaceIndex = FaceIndex(std::usize::MAX);

macro_rules! impl_display {
    ($name:ty) => {
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{:?}", self.0)
            }
        }
    };
}

impl_display!(HalfEdgeIndex);
impl_display!(VertexIndex);
impl_display!(FaceIndex);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversions() {
        assert_eq!(HalfEdgeIndex(0), 0.into());
        assert_eq!(VertexIndex(0), 0.into());
        assert_eq!(FaceIndex(0), 0.into());

        assert_eq!(HalfEdgeIndex(0) + 5, 5.into());
        assert_eq!(VertexIndex(0) + 5, 5.into());
        assert_eq!(FaceIndex(0) + 5, 5.into());
    }
}
