// ─────────────────────────────────────────────────────────────
// VECTRIX · core: the Metric enum and distance surface
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Public distance surface. The heavy lifting lives in [`crate::simd`];
//! this module maps a [`Metric`] to the right kernel and fixes the
//! convention that **all metrics are smaller-is-better**.

use crate::simd;

/// The similarity metric an index is scored under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Metric {
    /// Squared Euclidean distance.
    L2 = 0,
    /// Cosine distance, computed from a normalized dot product.
    Cosine = 1,
    /// Negative inner product, so that smaller-is-better still holds.
    Dot = 2,
}

impl Metric {
    /// Distance between two equal-length vectors. Always smaller-is-better.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => simd::squared_l2(a, b),
            Metric::Cosine => simd::cosine(a, b),
            Metric::Dot => -simd::dot(a, b),
        }
    }

    /// Stable, lowercase identifier (used in file headers and CLI configs).
    pub const fn as_str(self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::Cosine => "cosine",
            Metric::Dot => "dot",
        }
    }

    /// Decodes the on-disk encoding (see `docs/storage-format.md`).
    pub const fn from_u8(v: u8) -> Option<Metric> {
        match v {
            0 => Some(Metric::L2),
            1 => Some(Metric::Cosine),
            2 => Some(Metric::Dot),
            _ => None,
        }
    }
}

/// Convenience re-exports of the individual kernels.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    simd::dot(a, b)
}

#[inline]
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    simd::squared_l2(a, b)
}

#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    simd::cosine(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_distance_is_positive_and_smaller_for_similar_vectors() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let c = [3.0f32, 4.0, 0.0];
        assert!(Metric::L2.distance(&a, &b) < Metric::L2.distance(&a, &c));
    }

    #[test]
    fn cosine_of_parallel_vectors_is_zero() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [2.0f32, 4.0, 6.0];
        let d = Metric::Cosine.distance(&a, &b);
        assert!(d.abs() < 1e-6, "got {d}");
    }

    #[test]
    fn dot_metric_is_negative_inner_product() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert_eq!(Metric::Dot.distance(&a, &b), -32.0);
    }

    #[test]
    fn from_u8_roundtrips() {
        for m in [Metric::L2, Metric::Cosine, Metric::Dot] {
            assert_eq!(Metric::from_u8(m as u8), Some(m));
        }
        assert_eq!(Metric::from_u8(7), None);
    }
}
