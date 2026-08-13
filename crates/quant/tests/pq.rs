// ─────────────────────────────────────────────────────────────
// VECTRIX · quant integration tests: PQ + scalar quantizers
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Property and behavior tests for the quantizers.

use proptest::prelude::*;
use vectrix_core::vector::Vector;
use vectrix_quant::{ProductQuantizer, ProductQuantizerError, ScalarQuantizer};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn sq8_roundtrip_error_bounded_by_quantum(src in prop::collection::vec(-1.0f32..1.0, 1..32)) {
        let mut sq = ScalarQuantizer::default();
        sq.fit(&[Vector::new(src.clone())]);
        let back = sq.dequantize(&sq.quantize(&src));
        let quantum = 1.0 / sq.scale();
        for (a, b) in src.iter().zip(&back) {
            prop_assert!((a - b).abs() <= quantum / 2.0 + 1e-6, "{a} vs {b}");
        }
    }
}

#[test]
fn pq_dist_table_equals_decode_then_l2() {
    // The asymmetric-distance table must agree with decoding a code and
    // computing the exact squared L2 against the query residual.
    let mut pq = ProductQuantizer::new(8, 2, 4).unwrap(); // 2 sub-quants, 16 centroids
    let pts: Vec<Vector> = (0..48)
        .map(|i| {
            Vector::new(vec![
                (i % 6) as f32,
                (i / 6) as f32,
                (i % 4) as f32,
                (i % 3) as f32,
                (i % 8) as f32,
                (i % 2) as f32,
                (i % 5) as f32,
                (i % 7) as f32,
            ])
        })
        .collect();
    pq.train(&pts, 16, 99).unwrap();

    let query: Vec<f32> = vec![0.7, 1.3, 2.1, 0.9, 4.2, 0.1, 3.3, 2.4];
    let table = pq.dist_table(&query);
    for p in &pts {
        let code = pq.encode(p.as_slice());
        let decoded = pq.decode(&code);
        let exact = decoded
            .iter()
            .zip(&query)
            .map(|(d, q)| (d - q) * (d - q))
            .sum::<f32>();
        let approx = pq.approx_distance(&table, &code);
        assert!(
            (exact - approx).abs() < 1e-3,
            "exact {exact} vs approx {approx}"
        );
    }
}

#[test]
fn pq_train_requires_enough_samples() {
    let mut pq = ProductQuantizer::new(4, 2, 8).unwrap(); // needs 256 samples
    let samples: Vec<Vector> = (0..10).map(|i| Vector::new(vec![i as f32; 4])).collect();
    let err = pq.train(&samples, 4, 1).unwrap_err();
    assert_eq!(
        err,
        ProductQuantizerError::TooFewTrainingVectors {
            codebook: 256,
            samples: 10,
        }
    );
}

#[test]
fn pq_encode_length_matches_m() {
    let pq = ProductQuantizer::new(12, 3, 8).unwrap();
    let v = Vector::new(vec![0.0; 12]);
    let code = pq.encode(v.as_slice());
    assert_eq!(code.len(), 3);
}
