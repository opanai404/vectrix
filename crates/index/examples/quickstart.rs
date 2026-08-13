// ─────────────────────────────────────────────────────────────
// VECTRIX · index example: quickstart
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! End-to-end walkthrough: generate synthetic 64-d unit vectors, build an
//! HNSW index and an IVF-PQ index over them, and compare their top-10
//! results against an exact brute-force scan.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release -p vectrix-index --example quickstart
//! ```

use vectrix_core::distance::Metric;
use vectrix_core::rng::XorShift64;
use vectrix_core::vector::Vector;
use vectrix_index::hnsw::{HnswIndex, HnswParams};
use vectrix_index::ivf::{IvfPqIndex, IvfPqParams};

const DIM: usize = 64;
const N: usize = 5_000;

fn main() {
    // 1. Synthetic embeddings, roughly unit length (like real model output
    //    after normalization).
    let mut rng = XorShift64::new(0xC0FFEE);
    let vectors: Vec<Vector> = (0..N)
        .map(|_| {
            let mut v: Vec<f32> = (0..DIM).map(|_| rng.next_f32() - 0.5).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            Vector::new(v)
        })
        .collect();

    // 2. HNSW index.
    let params = HnswParams {
        m: 16,
        ef_construction: 200,
        ef_search: 64,
        seed: 42,
    };
    let mut hnsw = HnswIndex::new(Metric::Cosine, params);
    for v in &vectors {
        hnsw.insert(v.clone());
    }

    // 3. IVF-PQ index over the same corpus.
    let ivf_params = IvfPqParams {
        nlist: 64,
        nprobe: 8,
        m: 16,
        k_bits: 8,
        k_means_iters: 24,
        seed: 7,
    };
    let ivf = IvfPqIndex::build(Metric::Cosine, ivf_params, vectors.clone()).expect("build failed");

    // 4. Query with the first vector; print all three views.
    let query = vectors[0].to_vec();

    println!("query: vector #0 (dim={DIM})\n");

    println!("HNSW top-10 (ef={}):", params.ef_search);
    for hit in hnsw.search(&query, 10) {
        println!("  id={:>5}  distance={:.6}", hit.id, hit.distance);
    }

    println!("\nIVF-PQ top-10 (nprobe={}):", ivf_params.nprobe);
    for hit in ivf.search_nprobe(&query, 10, ivf_params.nprobe) {
        println!("  id={:>5}  distance={:.6}", hit.id, hit.distance);
    }

    println!("\nIVF-PQ top-10 (exact re-rank):");
    for hit in ivf.search_rerank(&query, 10, ivf_params.nprobe, 32) {
        println!("  id={:>5}  distance={:.6}", hit.id, hit.distance);
    }

    println!("\nexact baseline top-10:");
    for hit in ivf.search_exact(&query, 10) {
        println!("  id={:>5}  distance={:.6}", hit.id, hit.distance);
    }
}
