// ─────────────────────────────────────────────────────────────
// VECTRIX · core integration tests: bounded top-k heap
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

use proptest::prelude::*;
use vectrix_core::heap::TopK;

fn any_keys() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(any::<f32>(), 0..64)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn topk_returns_the_k_smallest_in_ascending_order(keys in any_keys(), capacity in 0usize..8) {
        let mut topk = TopK::new(capacity);
        for (i, k) in keys.iter().enumerate() {
            topk.push(*k, i);
        }
        let got: Vec<f32> = topk.into_items().into_iter().map(|(k, _)| k).collect();
        let mut want: Vec<f32> = keys;
        want.sort_by(f32::total_cmp);
        want.truncate(capacity);
        prop_assert_eq!(got, want, "capacity {capacity}");
    }
}

#[test]
fn push_evicts_only_the_worst() {
    let mut topk = TopK::new(3);
    topk.push(3.0, "c");
    topk.push(1.0, "a");
    topk.push(2.0, "b");
    topk.push(0.0, "z"); // beats 3.0
    topk.push(9.0, "nope"); // loses to everything
    let items = topk.into_items();
    assert_eq!(items, vec![(0.0, "z"), (1.0, "a"), (2.0, "b")]);
}

#[test]
fn len_tracks_retained_candidates() {
    let mut topk = TopK::new(4);
    for i in 0..10 {
        topk.push(i as f32, i);
    }
    assert_eq!(topk.len(), 4);
    assert_eq!(topk.peek_key(), Some(3.0));
}

#[test]
fn duplicates_are_allowed() {
    let mut topk = TopK::new(3);
    topk.push(1.0, ());
    topk.push(1.0, ());
    topk.push(1.0, ());
    topk.push(1.0, ());
    assert_eq!(topk.len(), 3);
}
