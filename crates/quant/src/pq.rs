// ─────────────────────────────────────────────────────────────
// VECTRIX · quant: product quantization
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! 8-bit product quantizer with per-subvector k-means codebooks.
//!
//! A vector of dimension `d = m · s` is split into `m` contiguous
//! subvectors of `s` components; each is encoded as the index of its
//! nearest codebook centroid, so a vector compresses to `m` bytes.
//!
//! Search never decodes: [`ProductQuantizer::dist_table`] builds an
//! `m × k` table of squared distances from the query residual to every
//! codebook centroid, and [`ProductQuantizer::approx_distance`] scores an
//! encoded vector with `m` table lookups. See `docs/quantization.md`.

use std::fmt;

use vectrix_core::rng::XorShift64;
use vectrix_core::vector::Vector;

use crate::kmeans::{kmeans, sq_dist};

/// Errors produced when configuring or training a quantizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductQuantizerError {
    /// Dimension is not divisible by the number of subquantizers.
    UnevenSplit { dim: usize, m: usize },
    /// Codebook size (`1 << k_bits`) exceeds the training set.
    TooFewTrainingVectors { codebook: usize, samples: usize },
    /// `k_bits` must be in `1..=8`.
    BadBits(u8),
}

impl fmt::Display for ProductQuantizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProductQuantizerError::UnevenSplit { dim, m } => {
                write!(f, "dimension {dim} not divisible by m={m}")
            }
            ProductQuantizerError::TooFewTrainingVectors { codebook, samples } => {
                write!(f, "codebook needs {codebook} centroids but only {samples} samples")
            }
            ProductQuantizerError::BadBits(b) => {
                write!(f, "k_bits={b} outside supported range 1..=8")
            }
        }
    }
}

impl std::error::Error for ProductQuantizerError {}

/// An 8-bit product quantizer.
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    m: usize,
    k: usize,
    sub_dim: usize,
    /// Flattened codebooks, layout `[m][k][sub_dim]`.
    codebooks: Vec<f32>,
}

impl ProductQuantizer {
    /// Builds a quantizer for `dim`-dimensional vectors split into `m`
    /// subquantizers with `1 << k_bits` centroids each.
    pub fn new(dim: usize, m: usize, k_bits: u8) -> Result<Self, ProductQuantizerError> {
        if !(1..=8).contains(&k_bits) {
            return Err(ProductQuantizerError::BadBits(k_bits));
        }
        if dim == 0 || dim % m != 0 {
            return Err(ProductQuantizerError::UnevenSplit { dim, m });
        }
        let k = 1usize << k_bits;
        let sub_dim = dim / m;
        Ok(Self {
            m,
            k,
            sub_dim,
            codebooks: vec![0.0; m * k * sub_dim],
        })
    }

    /// Number of subquantizers.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Codebook size per subquantizer.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Bytes per encoded vector (equal to `m` for 8-bit codes).
    pub fn code_len(&self) -> usize {
        self.m
    }

    /// Dimension of the vectors this quantizer was configured for.
    pub fn dim(&self) -> usize {
        self.sub_dim * self.m
    }

    /// Trains codebooks on `residuals` (each of dimension `dim()`).
    pub fn train(&mut self, residuals: &[Vector], iters: usize, seed: u64) -> Result<(), ProductQuantizerError> {
        let n = residuals.len();
        if n < self.k {
            return Err(ProductQuantizerError::TooFewTrainingVectors {
                codebook: self.k,
                samples: n,
            });
        }
        let mut rng = XorShift64::new(seed);
        for sub in 0..self.m {
            // Gather the subvectors of dimension `sub_dim` into one buffer.
            let lo = sub * self.sub_dim;
            let hi = (sub + 1) * self.sub_dim;
            let mut data = vec![0.0f32; n * self.sub_dim];
            for (i, r) in residuals.iter().enumerate() {
                let s = r.as_slice();
                debug_assert_eq!(s.len(), self.dim());
                data[i * self.sub_dim..(i + 1) * self.sub_dim].copy_from_slice(&s[lo..hi]);
            }
            let cb = kmeans(&data, n, self.sub_dim, self.k, iters, &mut rng);
            let start = sub * self.k * self.sub_dim;
            self.codebooks[start..start + cb.len()].copy_from_slice(&cb);
        }
        Ok(())
    }

    /// Encodes a vector (or residual) into `m` bytes.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert_eq!(vector.len(), self.dim());
        let mut code = vec![0u8; self.m];
        for sub in 0..self.m {
            let lo = sub * self.sub_dim;
            let hi = lo + self.sub_dim;
            let sub_vec = &vector[lo..hi];
            let cb_lo = sub * self.k * self.sub_dim;
            let cb = &self.codebooks[cb_lo..cb_lo + self.k * self.sub_dim];

            let mut best = 0u8;
            let mut best_d = f32::INFINITY;
            for j in 0..self.k {
                let c = &cb[j * self.sub_dim..(j + 1) * self.sub_dim];
                let d = sq_dist(sub_vec, c);
                if d < best_d {
                    best_d = d;
                    best = j as u8;
                }
            }
            code[sub] = best;
        }
        code
    }

    /// Decodes an `m`-byte code back into a `dim()`-dimensional vector.
    /// Used for diagnostics and for the exact re-rank path.
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        debug_assert_eq!(code.len(), self.m);
        let mut out = vec![0.0f32; self.dim()];
        for sub in 0..self.m {
            let j = code[sub] as usize;
            let src_lo = (sub * self.k + j) * self.sub_dim;
            let src = &self.codebooks[src_lo..src_lo + self.sub_dim];
            let lo = sub * self.sub_dim;
            out[lo..lo + self.sub_dim].copy_from_slice(src);
        }
        out
    }

    /// Builds the `m × k` asymmetric distance table for a query residual.
    /// `table[i * k + j]` is the squared distance between subvector `i` of
    /// the query and codebook centroid `j` of subquantizer `i`.
    pub fn dist_table(&self, query_residual: &[f32]) -> Vec<f32> {
        debug_assert_eq!(query_residual.len(), self.dim());
        let mut table = vec![0.0f32; self.m * self.k];
        for sub in 0..self.m {
            let lo = sub * self.sub_dim;
            let q = &query_residual[lo..lo + self.sub_dim];
            let cb_lo = sub * self.k * self.sub_dim;
            let cb = &self.codebooks[cb_lo..cb_lo + self.k * self.sub_dim];
            for j in 0..self.k {
                let c = &cb[j * self.sub_dim..(j + 1) * self.sub_dim];
                table[sub * self.k + j] = sq_dist(q, c);
            }
        }
        table
    }

    /// Scores an encoded vector against a precomputed distance table.
    pub fn approx_distance(&self, table: &[f32], code: &[u8]) -> f32 {
        debug_assert_eq!(table.len(), self.m * self.k);
        debug_assert_eq!(code.len(), self.m);
        let mut acc = 0.0f32;
        for sub in 0..self.m {
            acc += table[sub * self.k + code[sub] as usize];
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_uneven_split() {
        let err = ProductQuantizer::new(10, 3, 8).unwrap_err();
        assert_eq!(
            err,
            ProductQuantizerError::UnevenSplit { dim: 10, m: 3 }
        );
    }

    #[test]
    fn rejects_bad_bit_width() {
        assert_eq!(ProductQuantizer::new(8, 4, 9).unwrap_err(), ProductQuantizerError::BadBits(9));
        assert_eq!(ProductQuantizer::new(8, 4, 0).unwrap_err(), ProductQuantizerError::BadBits(0));
    }

    #[test]
    fn code_len_is_m() {
        let pq = ProductQuantizer::new(64, 8, 8).unwrap();
        assert_eq!(pq.code_len(), 8);
        assert_eq!(pq.dim(), 64);
        assert_eq!(pq.k(), 256);
    }

    #[test]
    fn encode_decode_roundtrips_approximately() {
        // With one subquantizer over 2-d data, 16 centroids reproduce a
        // well-separated cluster closely.
        let pq = ProductQuantizer::new(2, 1, 4).unwrap();
        let pts: Vec<Vector> = (0..64)
            .map(|i| Vector::new(vec![(i % 8) as f32, (i / 8) as f32]))
            .collect();
        pq.train(&pts, 12, 3).unwrap();

        for p in &pts {
            let code = pq.encode(p.as_slice());
            let out = pq.decode(&code);
            let err = sq_dist(p.as_slice(), &out);
            assert!(err < 1.0, "reconstruction error too high: {err}");
        }
    }
}
