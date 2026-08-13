// ─────────────────────────────────────────────────────────────
// VECTRIX · index: shared query result type
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! The result type shared by every index family.

/// A single search result. `distance` follows the workspace convention:
/// smaller is better (squared L2, cosine distance, or negated dot).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchHit {
    /// The stored vector id, assigned at insert/build time.
    pub id: u32,
    /// The distance from the query to this hit.
    pub distance: f32,
}

impl SearchHit {
    /// Ordering used to sort a hit list ascending by distance.
    pub fn cmp_by_distance(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.distance.total_cmp(&b.distance)
    }
}

/// Runtime knobs shared by the index families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchConfig {
    /// HNSW beam width (recall/throughput tradeoff).
    pub ef: usize,
    /// IVF-PQ number of inverted lists to probe.
    pub nprobe: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self { ef: 64, nprobe: 16 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_ordering_is_by_distance() {
        let a = SearchHit { id: 1, distance: 2.0 };
        let b = SearchHit { id: 2, distance: 1.0 };
        assert_eq!(SearchHit::cmp_by_distance(&b, &a), std::cmp::Ordering::Less);
    }
}
