// ─────────────────────────────────────────────────────────────
// VECTRIX · core: the dense Vector type
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! The [`Vector`] type: a dense, owned `f32` vector.

/// A dense `f32` vector. Stored boxed so a `Vector` moves by a single
/// pointer-sized word and stays cache-friendly when packed into a
/// `Vec<Node>` inside the graph indexes.
#[derive(Clone, Debug)]
pub struct Vector {
    dim: usize,
    data: Box<[f32]>,
}

impl Vector {
    /// Wraps owned data into a [`Vector`], capturing the dimension.
    pub fn new(data: impl Into<Vec<f32>>) -> Self {
        let data = data.into().into_boxed_slice();
        let dim = data.len();
        Self { dim, data }
    }

    /// A zero-filled vector of the given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            dim,
            data: vec![0.0; dim].into_boxed_slice(),
        }
    }

    /// Borrows the underlying `f32` data.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// The number of components.
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.dim == 0
    }

    /// Squared Euclidean norm.
    pub fn norm2(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum()
    }

    /// Unit-normalizes in place. Returns `None` when the vector is
    /// (near-)zero and cannot be normalized.
    pub fn normalize(&mut self) -> Option<()> {
        let n = self.norm2().sqrt();
        if n <= f32::EPSILON {
            return None;
        }
        for x in self.data.iter_mut() {
            *x /= n;
        }
        Some(())
    }

    /// Owned copy of the components.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.to_vec()
    }
}

impl From<Vec<f32>> for Vector {
    fn from(v: Vec<f32>) -> Self {
        Self::new(v)
    }
}

impl From<&[f32]> for Vector {
    fn from(v: &[f32]) -> Self {
        Self::new(v.to_vec())
    }
}

impl AsRef<[f32]> for Vector {
    fn as_ref(&self) -> &[f32] {
        &self.data
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.data == other.data
    }
}

impl Eq for Vector {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_captures_dimension() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.dim(), 3);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn normalize_produces_unit_norm() {
        let mut v = Vector::new(vec![3.0, 4.0]);
        assert!(v.normalize().is_some());
        assert!((v.norm2() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_rejects_zero_vector() {
        let mut v = Vector::zeros(4);
        assert!(v.normalize().is_none());
    }

    #[test]
    fn eq_is_by_component() {
        assert_eq!(Vector::new(vec![1.0]), Vector::new(vec![1.0]));
        assert_ne!(Vector::new(vec![1.0]), Vector::new(vec![2.0]));
        assert_ne!(Vector::new(vec![1.0, 2.0]), Vector::new(vec![1.0]));
    }
}
