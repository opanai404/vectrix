// ─────────────────────────────────────────────────────────────
// VECTRIX · core: hand-written SIMD distance kernels
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Distance kernels with runtime dispatch:
//!
//! ```text
//!     dot / squared_l2
//!        ├── x86_64   → AVX2 + FMA (runtime-detected)     · 8 × f32
//!        ├── aarch64  → NEON (architectural baseline)     · 4 × f32
//!        └── fallback → portable scalar with tail loop
//! ```
//!
//! All kernels require `a.len() == b.len()`; the public wrappers assert it
//! in debug builds. Vectors are consumed in whole SIMD lanes and the
//! remainder is folded in scalar-wise, so no lane is skipped and no
//! alignment is assumed.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vdupq_n_f32, vfmaq_f32, vgetq_lane_f32, vld1q_f32, vpaddq_f32, vsubq_f32,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Which kernel the next `dot`/`squared_l2` call will dispatch to.
///
/// Variants are cfg-gated so the enum and the dispatch `match` stay
/// exhaustive on every target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Scalar,
}

/// `true` when the current CPU exposes AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// The kernel chosen for a buffer of length `len`.
pub fn kernel_for(len: usize) -> KernelKind {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && len >= 8 {
            return KernelKind::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if len >= 4 {
        return KernelKind::Neon;
    }
    KernelKind::Scalar
}

/// Dot product `a · b`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "kernel operands must have equal length");
    match kernel_for(a.len()) {
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => {
            // SAFETY: lengths are equal (asserted above) and at least 8.
            unsafe { dot_avx2(a, b) }
        }
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => {
            // SAFETY: lengths are equal (asserted above) and at least 4.
            unsafe { dot_neon(a, b) }
        }
        KernelKind::Scalar => dot_scalar(a, b),
    }
}

/// Squared Euclidean distance `||a - b||²`.
#[inline]
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "kernel operands must have equal length");
    match kernel_for(a.len()) {
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => {
            // SAFETY: lengths are equal (asserted above) and at least 8.
            unsafe { l2_avx2(a, b) }
        }
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => {
            // SAFETY: lengths are equal (asserted above) and at least 4.
            unsafe { l2_neon(a, b) }
        }
        KernelKind::Scalar => l2_scalar(a, b),
    }
}

/// Cosine distance `1 - cos`, computed from a normalized dot product.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "kernel operands must have equal length");
    let ab = dot(a, b);
    let na = norm2(a);
    let nb = norm2(b);
    if na == 0.0 || nb == 0.0 {
        // Degenerate: a zero vector has no direction to agree on.
        return 1.0;
    }
    1.0 - ab / (na * nb).sqrt()
}

/// Squared Euclidean norm, reusing the dot kernel.
#[inline]
pub fn norm2(a: &[f32]) -> f32 {
    dot(a, a)
}

// ─────────────────────────────────────────────────────────────
// scalar kernels
// ─────────────────────────────────────────────────────────────

/// Portable reference kernels, exposed for testing and micro-benchmarking.
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        acc = f32::mul_add(x, y, acc);
    }
    acc
}

pub fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        acc = f32::mul_add(d, d, acc);
    }
    acc
}

// ─────────────────────────────────────────────────────────────
// x86_64 AVX2 + FMA kernels
// ─────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + LANES <= n {
        // SAFETY: `i + LANES <= n`, and both buffers are at least `n` long.
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(av, bv, acc);
        i += LANES;
    }
    let mut tail = 0.0f32;
    while i < n {
        tail = f32::mul_add(a[i], b[i], tail);
        i += 1;
    }
    horizontal_add8(acc) + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_avx2(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + LANES <= n {
        // SAFETY: `i + LANES <= n`, and both buffers are at least `n` long.
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        let d = _mm256_sub_ps(av, bv);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += LANES;
    }
    let mut tail = 0.0f32;
    while i < n {
        let d = a[i] - b[i];
        tail = f32::mul_add(d, d, tail);
        i += 1;
    }
    horizontal_add8(acc) + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_add8(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum = _mm_add_ps(lo, hi);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    let mut out = 0.0f32;
    _mm_store_ss(&mut out as *mut f32, sum);
    out
}

// ─────────────────────────────────────────────────────────────
// aarch64 NEON kernels
// ─────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 4;
    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + LANES <= n {
        // SAFETY: `i + LANES <= n`, and both buffers are at least `n` long.
        let av = vld1q_f32(a.as_ptr().add(i));
        let bv = vld1q_f32(b.as_ptr().add(i));
        acc = vfmaq_f32(acc, av, bv);
        i += LANES;
    }
    let mut s = vpaddq_f32(acc, acc);
    s = vpaddq_f32(s, s);
    let mut tail = vgetq_lane_f32::<0>(s);
    while i < n {
        tail = f32::mul_add(a[i], b[i], tail);
        i += 1;
    }
    tail
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l2_neon(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 4;
    let n = a.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + LANES <= n {
        // SAFETY: `i + LANES <= n`, and both buffers are at least `n` long.
        let av = vld1q_f32(a.as_ptr().add(i));
        let bv = vld1q_f32(b.as_ptr().add(i));
        let d = vsubq_f32(av, bv);
        acc = vfmaq_f32(acc, d, d);
        i += LANES;
    }
    let mut s = vpaddq_f32(acc, acc);
    s = vpaddq_f32(s, s);
    let mut tail = vgetq_lane_f32::<0>(s);
    while i < n {
        let d = a[i] - b[i];
        tail = f32::mul_add(d, d, tail);
        i += 1;
    }
    tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_matches_reference_sum() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        assert_eq!(dot_scalar(&a, &b), 70.0);
    }

    #[test]
    fn cosine_of_identical_vectors_is_zero() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        assert!(cosine(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_one() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dispatched_dot_matches_scalar() {
        let a: Vec<f32> = (0..513).map(|i| i as f32 * 0.001 - 0.5).collect();
        let b: Vec<f32> = (0..513).map(|i| i as f32 * 0.0007).collect();
        let want = dot_scalar(&a, &b);
        let got = dot(&a, &b);
        assert!(
            (want - got).abs() < 1e-3 * want.abs().max(1.0),
            "dispatched {got} != scalar {want}"
        );
    }

    #[test]
    fn dispatched_l2_matches_scalar() {
        let a: Vec<f32> = (0..1027).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..1027).map(|i| (i as f32).cos()).collect();
        let want = l2_scalar(&a, &b);
        let got = squared_l2(&a, &b);
        assert!((want - got).abs() < 1e-2 * want.max(1.0));
    }

    #[test]
    fn handles_short_and_odd_lengths() {
        // L2 of a vector against itself is always zero, for every length
        // the dispatch has to cope with (including lane-remainder tails).
        for len in [0usize, 1, 3, 7, 8, 9, 31, 32, 100, 1001] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let d = squared_l2(&a, &a);
            assert!(d.abs() < 1e-3, "len {len}: self-distance was {d}");
        }
    }
}
