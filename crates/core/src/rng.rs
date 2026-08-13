// ─────────────────────────────────────────────────────────────
// VECTRIX · core: deterministic xorshift PRNG
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Deterministic pseudo-randomness for index construction.
//!
//! The workspace deliberately avoids a PRNG dependency: [`XorShift64`] is a
//! compact xorshift64* generator, more than sufficient for HNSW level
//! assignment and IVF-PQ centroid seeding, and reproducible across runs for
//! a fixed seed.

/// xorshift64* PRNG. Deterministic for a fixed seed.
#[derive(Clone, Debug)]
pub struct XorShift64(u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        // A zero state would never escape the zero fixed point.
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform integer in `[lo, hi)`.
    pub fn gen_range(&mut self, lo: usize, hi: usize) -> usize {
        debug_assert!(hi > lo, "empty range");
        lo + (self.next_u64() % (hi - lo) as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_for_a_seed() {
        let mut a = XorShift64::new(42);
        let mut b = XorShift64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn next_f32_stays_in_unit_interval() {
        let mut rng = XorShift64::new(7);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn gen_range_respects_bounds() {
        let mut rng = XorShift64::new(1);
        for _ in 0..10_000 {
            let v = rng.gen_range(3, 9);
            assert!((3..9).contains(&v));
        }
    }

    #[test]
    fn zero_seed_does_not_trap() {
        // The constructor ORs in 1, so this must not loop forever.
        let mut rng = XorShift64::new(0);
        assert!(rng.next_u64() != 0);
    }
}
