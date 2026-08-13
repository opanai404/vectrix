// ─────────────────────────────────────────────────────────────
// VECTRIX · core integration tests: distance kernels
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Cross-checks the SIMD-dispatched kernels against the portable scalar
//! reference over many random shapes.

use proptest::prelude::*;
use vectrix_core::distance::{cosine, dot, squared_l2};
use vectrix_core::simd::{dot_scalar, l2_scalar, KernelKind, kernel_for};

/// Vector of up to 160 components so the AVX2 (8-lane) and NEON (4-lane)
/// tails are exercised with a variety of remainders.
fn any_vector() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(any::<f32>(), 0..160)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn dot_dispatched_matches_scalar(a in any_vector(), b in any_vector()) {
        prop_assume!(a.len() == b.len());
        let want = dot_scalar(&a, &b);
        let got = dot(&a, &b);
        let scale = want.abs().max(1.0);
        prop_assert!((want - got).abs() < 1e-2 * scale, "dot {got} != scalar {want}");
    }

    #[test]
    fn l2_dispatched_matches_scalar(a in any_vector(), b in any_vector()) {
        prop_assume!(a.len() == b.len());
        let want = l2_scalar(&a, &b);
        let got = squared_l2(&a, &b);
        let scale = want.abs().max(1.0);
        prop_assert!((want - got).abs() < 1e-2 * scale, "l2 {got} != scalar {want}");
    }

    #[test]
    fn cosine_is_bounded(a in any_vector(), b in any_vector()) {
        prop_assume!(a.len() == b.len());
        prop_assume!(a.iter().any(|x| *x != 0.0));
        prop_assume!(b.iter().any(|x| *x != 0.0));
        let c = cosine(&a, &b);
        prop_assert!((-0.001..=1.001).contains(&c), "cosine out of range: {c}");
    }
}

#[test]
fn dispatched_dot_agrees_on_avx2_aligned_lengths() {
    // 512 is exactly 64 AVX2 lanes; the dispatch and scalar must agree
    // to a tight tolerance (differences only from reassociation).
    let a: Vec<f32> = (0..512).map(|i| ((i % 17) as f32).sin()).collect();
    let b: Vec<f32> = (0..512).map(|i| ((i % 23) as f32).cos()).collect();
    let want = dot_scalar(&a, &b);
    let got = dot(&a, &b);
    let tol = 1e-3 * want.abs().max(1.0);
    assert!((want - got).abs() < tol, "dot {got} != scalar {want}");
}

#[test]
fn kernel_for_reports_scalar_for_tiny_inputs() {
    // Sub-lane buffers always fall back to the scalar kernel.
    assert_eq!(kernel_for(0), KernelKind::Scalar);
    assert_eq!(kernel_for(3), KernelKind::Scalar);
}

#[test]
fn empty_vectors_produce_zero_distances() {
    let a: [f32; 0] = [];
    assert_eq!(dot(&a, &a), 0.0);
    assert_eq!(squared_l2(&a, &a), 0.0);
}
