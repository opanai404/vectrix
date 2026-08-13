// ─────────────────────────────────────────────────────────────
// VECTRIX · quant: int8 scalar quantization
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Symmetric int8 (`SQ8`) quantizer. Maps each `f32` component to a
//! signed byte via a single shared scale, cutting memory 4× with much
//! lower reconstruction error than 8-bit product quantization on uniform
//! data. Intended for pre-filtering and near-exact low-footprint storage.

use vectrix_core::vector::Vector;

/// A symmetric int8 scalar quantizer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarQuantizer {
    scale: f32,
}

impl Default for ScalarQuantizer {
    fn default() -> Self {
        // Unit scale: bytes stay numerically close to the source values.
        Self { scale: 1.0 }
    }
}

impl ScalarQuantizer {
    /// Derives the scale from a sample so that the extreme magnitude maps
    /// to `127` (symmetric, sign-preserving).
    pub fn fit(&mut self, samples: &[Vector]) -> bool {
        let mut max_abs = 0.0f32;
        for v in samples {
            for x in v.as_slice() {
                max_abs = max_abs.max(x.abs());
            }
        }
        if max_abs <= f32::EPSILON {
            self.scale = 1.0;
            return false;
        }
        self.scale = 127.0 / max_abs;
        true
    }

    /// The derived scale; `1.0` before `fit` or for an all-zero sample.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Quantizes a vector to `i8`, clipping to `[-127, 127]`.
    pub fn quantize(&self, vector: &[f32]) -> Vec<i8> {
        vector
            .iter()
            .map(|x| {
                let q = (x * self.scale).round().clamp(-127.0, 127.0) as i8;
                q
            })
            .collect()
    }

    /// Dequantizes `i8` bytes back to `f32`.
    pub fn dequantize(&self, bytes: &[i8]) -> Vec<f32> {
        bytes.iter().map(|&b| b as f32 / self.scale).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_error_is_small_for_in_band_values() {
        let mut sq = ScalarQuantizer::default();
        assert!(sq.fit(&[Vector::new(vec![0.0, 1.0, -0.5])]));
        let src = [0.25f32, -0.1, 0.9];
        let back = sq.dequantize(&sq.quantize(&src));
        for (a, b) in src.iter().zip(&back) {
            assert!((a - b).abs() < 1.0 / 127.0 + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn fit_rejects_all_zero_samples() {
        let mut sq = ScalarQuantizer::default();
        assert!(!sq.fit(&[Vector::zeros(4)]));
        assert_eq!(sq.scale(), 1.0);
    }

    #[test]
    fn extreme_values_clip_to_127() {
        let mut sq = ScalarQuantizer::default();
        sq.fit(&[Vector::new(vec![5.0, -3.0])]);
        let q = sq.quantize(&[100.0, -100.0]);
        assert_eq!(q, vec![127, -127]);
    }
}
