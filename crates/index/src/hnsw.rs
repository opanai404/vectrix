// ─────────────────────────────────────────────────────────────
// VECTRIX · index: HNSW graph index
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Hierarchical Navigable Small World index.
//!
//! The graph is a set of layers: higher layers are sparser and let the
//! search "teleport" toward the region of interest; level 0 holds every
//! node and is traversed by the final bounded beam. New nodes draw a
//! random level `floor(-ln(U) / ln(m))`, connect to `m` diverse neighbors
//! per level under an `ef_construction`-wide beam, and links are pruned to
//! `M = m` (level 0) / `Mmax = 2m` (upper levels).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use vectrix_core::distance::Metric;
use vectrix_core::heap::TopK;
use vectrix_core::rng::XorShift64;
use vectrix_core::vector::Vector;

use crate::graph::{Neighbor, VisitedPool, select_diverse};
use crate::query::SearchHit;

/// Hard cap on the number of graph layers per node.
pub const MAX_LEVEL: usize = 32;

/// Configuration for the HNSW graph.
#[derive(Clone, Copy, Debug)]
pub struct HnswParams {
    /// Maximum out-degree per node per level.
    pub m: usize,
    /// Beam width used while inserting (recall at build time).
    pub ef_construction: usize,
    /// Default beam width used while searching.
    pub ef_search: usize,
    /// Seed for the level-assignment PRNG.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x5EED_2019,
        }
    }
}

#[derive(Debug)]
struct Node {
    id: u32,
    vector: Vector,
    level: usize,
    /// One neighbor list per layer the node participates in; `links[l]`
    /// exists for `l <= level`, sorted ascending by distance to `id`.
    links: Vec<Vec<Neighbor>>,
}

/// The HNSW index. Read-only access (`search`) is `Send + Sync` and can
/// be shared across threads; `insert` requires `&mut self`.
#[derive(Debug)]
pub struct HnswIndex {
    metric: Metric,
    params: HnswParams,
    rng: XorShift64,
    nodes: Vec<Node>,
    entry: Option<u32>,
    /// `1 / ln(m)` used to draw random levels.
    level_mult: f32,
}

impl HnswIndex {
    /// Constructs an index for `metric` with explicit graph parameters.
    pub fn new(metric: Metric, params: HnswParams) -> Self {
        let m = params.m.max(2);
        Self {
            metric,
            params,
            rng: XorShift64::new(params.seed),
            nodes: Vec::new(),
            entry: None,
            level_mult: 1.0 / (m as f32).ln(),
        }
    }

    /// Constructs an index with [`HnswParams::default`].
    pub fn with_defaults(metric: Metric) -> Self {
        Self::new(metric, HnswParams::default())
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The top-layer entry node, if any.
    pub fn entry_point(&self) -> Option<u32> {
        self.entry
    }

    /// Appends a vector, returning its id (`0..len`).
    pub fn insert(&mut self, vector: Vector) -> u32 {
        let id = self.nodes.len() as u32;
        self.insert_with_id(id, vector)
    }

    /// Appends a vector under an explicit id. Ids are indices into the
    /// node table, so they must be appended in order: `id` must equal the
    /// current length. The index does not deduplicate.
    pub fn insert_with_id(&mut self, id: u32, vector: Vector) -> u32 {
        debug_assert_eq!(
            id as usize,
            self.nodes.len(),
            "insert_with_id ids must be appended in order"
        );

        let level = self.random_level();
        let node = Node {
            id,
            vector,
            level,
            links: vec![Vec::new(); level + 1],
        };
        // Clone the query before `node` moves into the index: `connect`
        // needs `&mut self` while the per-layer loop borrows it.
        let query = node.vector.to_vec();

        let Some(entry) = self.entry else {
            self.entry = Some(id);
            self.nodes.push(node);
            return id;
        };

        let mut curr = entry;
        let top = self.nodes[curr as usize].level;

        // Descend greedily from the top layer to just above the new
        // node's level, one best-neighbor hop at a time.
        for l in (level + 1..=top).rev() {
            curr = self.greedy_descend(curr, &query, l);
        }

        // The node must be reachable by `connect` before any edge is
        // added, so it is pushed first.
        self.nodes.push(node);

        // At each layer from the new node's top down to 0: beam-search,
        // select diverse neighbors, wire both directions, prune.
        for l in (0..=level.min(top)).rev() {
            let found = self.search_layer(curr, &query, self.params.ef_construction, l);
            let dist = |a: u32, b: u32| {
                self.distance(
                    &self.nodes[a as usize].vector.as_slice(),
                    &self.nodes[b as usize].vector.as_slice(),
                )
            };
            let selected = select_diverse(found, self.params.m, dist);
            for nb in &selected {
                self.connect(id, nb.id, l);
            }
            if let Some(closest) = selected.first() {
                curr = closest.id;
            }
        }

        if level > top {
            self.entry = Some(id);
        }
        id
    }

    /// Searches with the configured `ef_search` beam.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchHit> {
        self.search_ef(query, k, self.params.ef_search)
    }

    /// Searches with an explicit beam width `ef`. Returns up to `k` hits
    /// ascending by distance; an empty index yields an empty result.
    pub fn search_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchHit> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }
        let entry = self.entry.expect("non-empty index has an entry");
        let top = self.nodes[entry as usize].level;
        let mut curr = entry;
        for l in (1..=top).rev() {
            curr = self.greedy_descend(curr, query, l);
        }
        let found = self.search_layer(curr, query, ef, 0);
        found
            .into_iter()
            .take(k)
            .map(|n| SearchHit { id: n.id, distance: n.distance })
            .collect()
    }

    // ─────────────────────────────────────────────
    // internals
    // ─────────────────────────────────────────────

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.metric.distance(a, b)
    }

    fn random_level(&mut self) -> usize {
        // `floor(-ln(U) / ln(m))`, clamped to a sane ceiling.
        let u = self.rng.next_f32().clamp(1e-6, 1.0 - 1e-6);
        let l = (-u.ln() * self.level_mult) as usize;
        l.min(MAX_LEVEL)
    }

    /// Moves one step to the nearest neighbor of `query` at `level`,
    /// repeating until a local minimum.
    fn greedy_descend(&self, mut curr: u32, query: &[f32], level: usize) -> u32 {
        loop {
            let mut best = curr;
            let mut best_d = self.distance(&self.nodes[curr as usize].vector.as_slice(), query);
            let mut changed = false;
            for nb in &self.nodes[curr as usize].links[level] {
                let d = self.distance(&self.nodes[nb.id as usize].vector.as_slice(), query);
                if d < best_d {
                    best_d = d;
                    best = nb.id;
                    changed = true;
                }
            }
            if !changed {
                return best;
            }
            curr = best;
        }
    }

    /// Bounded best-first search at one layer. Returns up to `ef`
    /// neighbors sorted ascending by distance to `query`.
    fn search_layer(&self, entry: u32, query: &[f32], ef: usize, level: usize) -> Vec<Neighbor> {
        debug_assert!(level < self.nodes[entry as usize].links.len());
        let ef = ef.max(1);

        let mut visited = VisitedPool::new(self.nodes.len());
        let mut candidates: BinaryHeap<Reverse<Neighbor>> = BinaryHeap::new();
        let mut topk = TopK::new(ef);

        let e_dist = self.distance(&self.nodes[entry as usize].vector.as_slice(), query);
        let e = Neighbor { id: entry, distance: e_dist };
        candidates.push(Reverse(e));
        topk.push(e_dist, entry);
        visited.mark(entry as usize);

        while let Some(Reverse(cur)) = candidates.pop() {
            let worst = topk.peek_key().unwrap_or(f32::INFINITY);
            if cur.distance > worst {
                break;
            }
            for &nb in &self.nodes[cur.id as usize].links[level] {
                if visited.visit(nb.id as usize) {
                    let d = self.distance(&self.nodes[nb.id as usize].vector.as_slice(), query);
                    if d < worst || topk.len() < ef {
                        candidates.push(Reverse(Neighbor { id: nb.id, distance: d }));
                        topk.push(d, nb.id);
                    }
                }
            }
        }

        topk.into_items()
            .into_iter()
            .map(|(distance, id)| Neighbor { id, distance })
            .collect()
    }

    /// Adds an undirected edge between `from` and `to` at `level`,
    /// trimming both neighbor lists to their capacity.
    fn connect(&mut self, from: u32, to: u32, level: usize) {
        let cap = if level == 0 { self.params.m * 2 } else { self.params.m };
        let d = self.distance(
            &self.nodes[from as usize].vector.as_slice(),
            &self.nodes[to as usize].vector.as_slice(),
        );
        self.add_link(from, to, d, level, cap);
        self.add_link(to, from, d, level, cap);
    }

    fn add_link(&mut self, node: u32, other: u32, dist: f32, level: usize, cap: usize) {
        let links = &mut self.nodes[node as usize].links[level];
        if links.iter().any(|n| n.id == other) {
            return;
        }
        links.push(Neighbor { id: other, distance: dist });
        links.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
        links.truncate(cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_insert_becomes_entry() {
        let mut idx = HnswIndex::with_defaults(Metric::L2);
        let id = idx.insert(Vector::new(vec![1.0, 0.0]));
        assert_eq!(id, 0);
        assert_eq!(idx.entry_point(), Some(0));
    }

    #[test]
    fn empty_index_returns_no_hits() {
        let idx = HnswIndex::with_defaults(Metric::L2);
        assert!(idx.search(&[1.0, 2.0], 5).is_empty());
    }

    #[test]
    fn exact_self_search_is_zero_distance() {
        let mut idx = HnswIndex::with_defaults(Metric::L2);
        idx.insert(Vector::new(vec![0.5, 0.25, 0.125]));
        let hits = idx.search(&[0.5, 0.25, 0.125], 1);
        assert_eq!(hits[0].id, 0);
        assert!(hits[0].distance.abs() < 1e-6);
    }
}
