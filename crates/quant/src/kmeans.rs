// ─────────────────────────────────────────────────────────────
// VECTRIX · quant: k-means clustering (Lloyd)
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! A small, deterministic Lloyd's k-means used both for IVF centroid
//! training and for per-subvector PQ codebook training.
//!
//! Initialization picks `k` distinct data rows uniformly at random
//! (seeded), and empty clusters keep their previous centroid — a common
//! simplification that keeps the loop divergence-free.

use vectrix_core::rng::XorShift64;

/// Clusters `n` rows of `dim` floats into `k` centroids.
///
/// `data` is row-major (`n × dim`). Returns `k × dim` centroids in the
/// same layout. Runs `iters` assignment/update passes.
pub fn kmeans(data: &[f32], n: usize, dim: usize, k: usize, iters: usize, rng: &mut XorShift64) -> Vec<f32> {
    assert!(k > 0 && k <= n, "need 0 < k <= n, got k={k}, n={n}");
    assert!(data.len() == n * dim, "data length mismatch");

    // ---- initialization: k distinct random rows ----
    let mut centroids: Vec<f32> = Vec::with_capacity(k * dim);
    let mut seen = vec![false; n];
    let mut placed = 0usize;
    while placed < k {
        let i = rng.gen_range(0, n);
        if !seen[i] {
            seen[i] = true;
            centroids.extend_from_slice(&data[i * dim..(i + 1) * dim]);
            placed += 1;
        }
    }

    // ---- Lloyd iterations ----
    let mut assign = vec![0usize; n];
    let mut sums = vec![0.0f32; k * dim];
    let mut counts = vec![0usize; k];

    for _ in 0..iters {
        sums.iter_mut().for_each(|v| *v = 0.0);
        counts.iter_mut().for_each(|v| *v = 0);

        // assignment
        for i in 0..n {
            let row = &data[i * dim..(i + 1) * dim];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let d = sq_dist(row, &centroids[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assign[i] = best;
            for j in 0..dim {
                sums[best * dim + j] += row[j];
            }
            counts[best] += 1;
        }

        // update (empty clusters keep their centroid)
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            for j in 0..dim {
                centroids[c * dim + j] = sums[c * dim + j] / counts[c] as f32;
            }
        }
    }

    centroids
}

/// Squared L2 between two equal-length slices.
#[inline]
pub(crate) fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        acc = f32::mul_add(d, d, acc);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_cluster_centroid_is_the_mean() {
        // All points are identical apart from the first, so one cluster
        // must converge to the arithmetic mean.
        let data = [0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let mut rng = XorShift64::new(5);
        let c = kmeans(&data, 3, 3, 1, 5, &mut rng);
        assert!((c[0] - 0.0).abs() < 1e-4);
        assert!((c[1] - 1.0).abs() < 1e-4);
        assert!((c[2] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn separated_clusters_are_recovered() {
        // Two tight clusters on opposite sides of the origin.
        let data: Vec<f32> = (0..40)
            .flat_map(|i| {
                let side = if i < 20 { -10.0f32 } else { 10.0 };
                [side, side, 0.0f32]
            })
            .collect();
        let mut rng = XorShift64::new(1);
        let mut centroids = kmeans(&data, 40, 3, 2, 20, &mut rng);
        centroids.sort_by(f32::total_cmp);
        // Expect cluster means near (-10,-10,0) and (10,10,0).
        assert!((centroids[0] + 10.0).abs() < 1e-2, "got {}", centroids[0]);
        assert!((centroids[3] - 10.0).abs() < 1e-2, "got {}", centroids[3]);
    }

    #[test]
    #[should_panic(expected = "0 < k <= n")]
    fn rejects_empty_or_oversized_k() {
        let data = [1.0f32, 2.0];
        let mut rng = XorShift64::new(2);
        let _ = kmeans(&data, 2, 1, 3, 2, &mut rng);
    }
}
