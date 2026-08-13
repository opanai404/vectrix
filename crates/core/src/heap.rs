// ─────────────────────────────────────────────────────────────
// VECTRIX · core: bounded top-k accumulator
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! [`TopK`] keeps the `capacity` smallest `(key, item)` pairs seen so far.
//! It backs both HNSW's beam search and IVF-PQ's list scoring, and is the
//! single place where "the worst of the current best-k" is decided.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// `f32` wrapper with a total order so it can live inside a `BinaryHeap`.
#[derive(Clone, Copy, Debug)]
pub struct Key(pub f32);

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Accumulates the `capacity` smallest `(key, item)` pairs.
///
/// The internal `BinaryHeap` is a max-heap, so its root is always the
/// *worst* kept candidate — exactly what search needs to prune.
#[derive(Debug)]
pub struct TopK<V> {
    capacity: usize,
    heap: BinaryHeap<(Key, V)>,
}

impl<V> TopK<V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::new(),
        }
    }

    /// Pushes a candidate, evicting the current worst when full.
    pub fn push(&mut self, key: f32, item: V) {
        if self.capacity == 0 {
            return;
        }
        let k = Key(key);
        if self.heap.len() < self.capacity {
            self.heap.push((k, item));
        } else if key < self.peek_key().unwrap_or(f32::INFINITY) {
            // SAFETY: len is non-zero here (capacity > 0 and len >= capacity).
            let mut top = self.heap.peek_mut().expect("heap is non-empty");
            *top = (k, item);
        }
    }

    /// Number of candidates currently retained.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// The largest (worst) retained key, if any.
    pub fn peek_key(&self) -> Option<f32> {
        self.heap.peek().map(|(k, _)| k.0)
    }

    /// Drains the accumulator, returning items sorted ascending by key.
    pub fn into_items(self) -> Vec<(f32, V)> {
        let mut items: Vec<(f32, V)> = self
            .heap
            .into_iter()
            .map(|(Key(k), v)| (k, v))
            .collect();
        items.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_only_the_smallest() {
        let mut topk = TopK::new(3);
        for (k, v) in [(5.0, "e"), (1.0, "a"), (4.0, "d"), (2.0, "b"), (3.0, "c")] {
            topk.push(k, v);
        }
        let items = topk.into_items();
        assert_eq!(
            items,
            vec![(1.0, "a"), (2.0, "b"), (3.0, "c")]
        );
    }

    #[test]
    fn zero_capacity_discards_everything() {
        let mut topk = TopK::new(0);
        topk.push(0.0, ());
        assert!(topk.is_empty());
    }

    #[test]
    fn peek_key_is_the_worst_retained() {
        let mut topk = TopK::new(2);
        topk.push(0.5, ());
        topk.push(0.1, ());
        topk.push(2.0, ());
        assert_eq!(topk.peek_key(), Some(0.5));
    }

    #[test]
    fn sort_order_is_ascending() {
        let mut topk = TopK::new(5);
        topk.push(9.0, 9);
        topk.push(1.0, 1);
        topk.push(5.0, 5);
        let keys: Vec<f32> = topk.into_items().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![1.0, 5.0, 9.0]);
    }
}
