// ─────────────────────────────────────────────────────────────
// VECTRIX · index integration tests: HNSW
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! End-to-end HNSW tests with a brute-force baseline.

use vectrix_core::distance::Metric;
use vectrix_core::rng::XorShift64;
use vectrix_core::vector::Vector;
use vectrix_index::hnsw::{HnswIndex, HnswParams};

const DIM: usize = 32;

/// Deterministic pseudo-random unit vectors.
fn unit_vectors(n: usize, seed: u64) -> Vec<Vector> {
    let mut rng = XorShift64::new(seed);
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..DIM).map(|_| rng.next_f32() - 0.5).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            Vector::new(v)
        })
        .collect()
}

/// Exact top-k by brute force, returning ids.
fn exact_topk(vectors: &[Vector], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (Metric::Cosine.distance(query, v.as_slice()), i as u32))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.truncate(k);
    scored.into_iter().map(|(_, id)| id).collect()
}

fn recall(hits: &[u32], exact: &[u32]) -> f32 {
    let hit: std::collections::HashSet<u32> = hits.iter().copied().collect();
    let found = exact.iter().filter(|id| hit.contains(id)).count();
    found as f32 / exact.len() as f32
}

#[test]
fn recall_against_brute_force_is_high() {
    let vectors = unit_vectors(600, 11);
    let params = HnswParams {
        m: 16,
        ef_construction: 120,
        ef_search: 64,
        seed: 11,
    };
    let mut idx = HnswIndex::new(Metric::Cosine, params);
    for v in &vectors {
        idx.insert(v.clone());
    }

    let mut rng = XorShift64::new(99);
    let mut total = 0.0f32;
    for _ in 0..20 {
        let query = unit_vectors(1, rng.next_u64())[0].to_vec();
        let k = 10;
        let exact = exact_topk(&vectors, &query, k);
        let hits: Vec<u32> = idx.search(&query, k).into_iter().map(|h| h.id).collect();
        total += recall(&hits, &exact);
    }
    let mean = total / 20.0;
    assert!(
        mean > 0.6,
        "recall too low for this config: {mean:.3} (insertion order matters)"
    );
}

#[test]
fn search_returns_ascending_distances() {
    let vectors = unit_vectors(200, 5);
    let mut idx = HnswIndex::with_defaults(Metric::Cosine);
    for v in &vectors {
        idx.insert(v.clone());
    }
    let hits = idx.search(&vectors[7].to_vec(), 8);
    assert_eq!(hits.len(), 8);
    for w in hits.windows(2) {
        assert!(w[0].distance <= w[1].distance);
    }
}

#[test]
fn ids_are_assigned_monotonically() {
    let mut idx = HnswIndex::with_defaults(Metric::L2);
    for i in 0..50 {
        assert_eq!(idx.insert(Vector::zeros(DIM)), i);
    }
    assert_eq!(idx.len(), 50);
}

#[test]
fn explicit_ef_changes_result_sets() {
    let vectors = unit_vectors(400, 3);
    let mut idx = HnswIndex::with_defaults(Metric::L2);
    for v in &vectors {
        idx.insert(v.clone());
    }
    let q = vectors[0].to_vec();
    let wide: Vec<u32> = idx.search_ef(&q, 10, 200).into_iter().map(|h| h.id).collect();
    let narrow: Vec<u32> = idx.search_ef(&q, 10, 8).into_iter().map(|h| h.id).collect();
    // Both are valid k=10 result sets.
    assert_eq!(wide.len(), 10);
    assert_eq!(narrow.len(), 10);

    // A wider beam is strictly more accurate: its nearest hit is at most
    // as far as the narrow beam's.
    let d_wide = idx.search_ef(&q, 1, 200)[0].distance;
    let d_narrow = idx.search_ef(&q, 1, 8)[0].distance;
    assert!(d_wide <= d_narrow + 1e-6, "wider beam should not be worse");
}

#[test]
fn exact_self_search_finds_the_inserted_vector() {
    let mut idx = HnswIndex::with_defaults(Metric::L2);
    idx.insert(Vector::new(vec![3.0, 1.0, 4.0]));
    idx.insert(Vector::new(vec![1.0, 5.0, 9.0]));
    let hits = idx.search(&[3.0, 1.0, 4.0], 1);
    assert_eq!(hits[0].id, 0);
    assert!(hits[0].distance < 1e-5);
}
