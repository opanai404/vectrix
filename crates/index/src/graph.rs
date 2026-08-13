// ─────────────────────────────────────────────────────────────
// VECTRIX · index: graph primitives
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Shared graph machinery for the HNSW index:
//!
//! - [`Neighbor`] — a node id plus its distance to some query.
//! - [`select_diverse`] — the neighbor-selection heuristic that prefers
//!   close candidates but rejects ones too similar to what is already
//!   selected, keeping the graph navigable instead of star-shaped.
//! - [`VisitedPool`] — a reusable visited marker set so the beam search
//!   avoids revisiting nodes without a per-query `HashSet` allocation.

use std::cmp::Ordering;

/// A node reachable during graph traversal, with its distance to the
/// current query. `Ord` sorts by distance ascending.
#[derive(Clone, Copy, Debug)]
pub struct Neighbor {
    pub id: u32,
    pub distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.distance == other.distance
    }
}
impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Selects up to `m` neighbors from `candidates` (already sorted by
/// distance to the query). A candidate is kept only if it is closer to the
/// query than it is to every already-selected neighbor — the standard HNSW
/// "diversity" heuristic that prevents redundant links.
///
/// `dist` measures the distance between two *node ids*.
pub fn select_diverse(
    candidates: Vec<Neighbor>,
    m: usize,
    dist: impl Fn(u32, u32) -> f32,
) -> Vec<Neighbor> {
    if candidates.is_empty() || m == 0 {
        return Vec::new();
    }
    let mut selected: Vec<Neighbor> = Vec::with_capacity(m.min(candidates.len()));
    for cand in candidates {
        if selected.len() >= m {
            break;
        }
        let redundant = selected
            .iter()
            .any(|s| cand.distance >= dist(cand.id, s.id));
        if !redundant {
            selected.push(cand);
        }
    }
    selected
}

/// A compact visited marker set with O(1) amortized reset.
///
/// A single `epoch` counter is bumped on [`VisitedPool::reset`]; a node is
/// considered visited when its mark equals the current epoch. This avoids
/// re-zeroing the backing array between queries.
#[derive(Debug, Clone)]
pub struct VisitedPool {
    marks: Vec<u32>,
    epoch: u32,
}

impl VisitedPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            marks: vec![0; capacity],
            epoch: 1,
        }
    }

    /// Marks `idx` as visited.
    pub fn mark(&mut self, idx: usize) {
        self.marks[idx] = self.epoch;
    }

    /// Marks `idx` visited and returns `true` if this was the first visit
    /// in the current epoch.
    pub fn visit(&mut self, idx: usize) -> bool {
        if self.marks[idx] == self.epoch {
            return false;
        }
        self.marks[idx] = self.epoch;
        true
    }

    /// Starts a new epoch. If the counter would wrap, the array is
    /// re-zeroed so stale marks can never alias a future epoch.
    pub fn reset(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    pub fn len(&self) -> usize {
        self.marks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visited_marks_exactly_once_per_epoch() {
        let mut pool = VisitedPool::new(4);
        assert!(pool.visit(2));
        assert!(!pool.visit(2));
        assert!(pool.visit(0));
        pool.reset();
        assert!(pool.visit(2), "reset must clear marks");
    }

    #[test]
    fn diverse_selection_skips_redundant_neighbors() {
        // dist = manhattan between ids (a stand-in for real vector dist).
        let dist = |a: u32, b: u32| (a as f32 - b as f32).abs();
        let candidates = vec![
            Neighbor { id: 10, distance: 0.0 },
            Neighbor { id: 11, distance: 0.1 }, // 11 is adjacent to 10
            Neighbor { id: 20, distance: 0.5 }, // far from 10 -> kept
            Neighbor { id: 21, distance: 0.6 }, // adjacent to 20 -> dropped
        ];
        let selected = select_diverse(candidates, 4, dist);
        let ids: Vec<u32> = selected.iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![10, 20]);
    }

    #[test]
    fn selection_respects_m() {
        let dist = |a: u32, b: u32| (a as f32 - b as f32).abs();
        let candidates = vec![
            Neighbor { id: 0, distance: 0.0 },
            Neighbor { id: 1, distance: 1.0 },
            Neighbor { id: 2, distance: 2.0 },
            Neighbor { id: 3, distance: 3.0 },
        ];
        let selected = select_diverse(candidates, 2, dist);
        assert_eq!(selected.len(), 2);
    }
}
