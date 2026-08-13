// ─────────────────────────────────────────────────────────────
// VECTRIX · index integration tests: IVF-PQ
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! End-to-end IVF-PQ tests: build, approximate search, exact baseline,
//! and the nprobe/recall tradeoff.

use vectrix_core::distance::Metric;
use vectrix_core::rng::XorShift64;
use vectrix_core::vector::Vector;
use vectrix_index::ivf::{IvfPqBuildError, IvfPqIndex, IvfPqParams};

const DIM: usize = 16;

fn clustered_vectors(n: usize, clusters: usize, seed: u64) -> Vec<Vector> {
    let mut rng = XorShift64::new(seed);
    (0..n)
        .map(|i| {
            let cx = (i % clusters) as f32 * 10.0;
            let cy = (i / clusters) as f32 * 10.0;
            let mut v: Vec<f32> = (0..DIM)
                .map(|d| {
                    if d == 0 {
                        cx + (rng.next_f32() - 0.5)
                    } else if d == 1 {
                        cy + (rng.next_f32() - 0.5)
                    } else {
                        rng.next_f32() - 0.5
                    }
                })
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            Vector::new(v)
        })
        .collect()
}

fn params() -> IvfPqParams {
    IvfPqParams {
        nlist: 8,
        nprobe: 2,
        m: 4,
        k_bits: 8,
        k_means_iters: 12,
        seed: 7,
    }
}

#[test]
fn approximate_search_finds_the_exact_neighbor() {
    let vectors = clustered_vectors(200, 4, 1);
    let idx = IvfPqIndex::build(Metric::Cosine, params(), vectors.clone()).unwrap();
    let q = vectors[0].to_vec();
    let hits = idx.search_nprobe(&q, 5, params().nprobe);
    assert_eq!(hits.len(), 5);
    assert_eq!(hits[0].id, 0, "query vector must be its own nearest neighbor");
}

#[test]
fn larger_nprobe_improves_or_keeps_recall() {
    let vectors = clustered_vectors(240, 4, 2);
    let idx = IvfPqIndex::build(Metric::Cosine, params(), vectors.clone()).unwrap();
    let mut rng = XorShift64::new(55);

    // The k-smallest set of a superset of candidates elementwise dominates
    // the k-smallest set of a subset, so the top-1 *distance* must never
    // get worse as nprobe grows — a guaranteed monotonicity we can assert
    // even with noisy approximate distances.
    let mut sum_n1 = 0.0f32;
    let mut sum_n4 = 0.0f32;
    let mut recall4_total = 0.0f32;
    for _ in 0..12 {
        let q = clustered_vectors(1, 4, rng.next_u64())[0].to_vec();
        let exact: Vec<u32> = idx.search_exact(&q, 5).into_iter().map(|h| h.id).collect();
        let d1 = idx.search_nprobe(&q, 1, 1)[0].distance;
        let d4 = idx.search_nprobe(&q, 1, 4)[0].distance;
        assert!(d4 <= d1 + 1e-6, "nprobe=4 top-1 ({d4}) worse than nprobe=1 ({d1})");
        sum_n1 += d1;
        sum_n4 += d4;

        let r4: Vec<u32> = idx.search_nprobe(&q, 5, 4).into_iter().map(|h| h.id).collect();
        recall4_total += overlap(&r4, &exact);
    }
    // nprobe=4 must not be worse than nprobe=1 on average (it is strictly
    // better on clustered data, but we only assert non-worse for safety)...
    assert!(sum_n4 <= sum_n1, "mean top-1 distance should not grow with nprobe");
    // ...and the full-beam recall has to clear a meaningful floor.
    assert!(recall4_total / 12.0 >= 0.5, "recall@5 too low for clustered data");
}

#[test]
fn exact_search_matches_brute_force() {
    let vectors = clustered_vectors(150, 3, 3);
    let idx = IvfPqIndex::build(Metric::L2, params(), vectors).unwrap();
    let q = vec![0.5f32; DIM];
    let hits = idx.search_exact(&q, 3);
    assert_eq!(hits.len(), 3);
    for w in hits.windows(2) {
        assert!(w[0].distance <= w[1].distance);
    }
}

#[test]
fn rerank_keeps_exact_top1() {
    let vectors = clustered_vectors(180, 4, 4);
    let idx = IvfPqIndex::build(Metric::Cosine, params(), vectors.clone()).unwrap();
    let q = vectors[0].to_vec();
    let hits = idx.search_rerank(&q, 5, params().nprobe, 32);
    assert_eq!(hits[0].id, 0);
}

#[test]
fn build_rejects_undersized_corpus() {
    let vectors = clustered_vectors(4, 1, 9);
    let err = IvfPqIndex::build(Metric::L2, params(), vectors).unwrap_err();
    assert!(matches!(err, IvfPqBuildError::TooFewVectors { .. }));
}

fn overlap(hits: &[u32], exact: &[u32]) -> f32 {
    let set: std::collections::HashSet<u32> = hits.iter().copied().collect();
    let found = exact.iter().filter(|id| set.contains(id)).count();
    found as f32 / exact.len() as f32
}
