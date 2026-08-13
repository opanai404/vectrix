# API — `vectrix-index::hnsw`

This page documents the public surface of the HNSW index. It is the
reference implementation of the graph index and the default choice for
datasets up to a few million vectors in RAM.

## Types

### `HnswParams`

| Field             | Type     | Default | Meaning                                        |
|-------------------|----------|---------|------------------------------------------------|
| `m`               | `usize`  | `16`    | Maximum out-degree per node per level          |
| `ef_construction` | `usize`  | `200`   | Beam width used during insert                  |
| `ef_search`       | `usize`  | `64`    | Default beam width used during search          |
| `seed`            | `u64`    | `0x5EED_2019` | PRNG seed for level assignment            |

Set `m` to 12–32 for most workloads; raise `ef_construction` at build time
for recall, `ef_search` at query time for recall.

### `HnswIndex`

| Method | Signature | Notes |
|--------|-----------|-------|
| `new` | `(Metric, HnswParams) -> Self` | Construct with an explicit config |
| `with_defaults` | `(Metric) -> Self` | `HnswParams::default()` |
| `insert` | `(&mut self, Vector) -> u32` | Appends a node, returns its id |
| `insert_with_id` | `(&mut self, u32, Vector) -> u32` | Insert with a caller-chosen id |
| `search` | `(&self, &[f32], usize) -> Vec<SearchHit>` | Uses `ef_search` |
| `search_ef` | `(&self, &[f32], usize, usize) -> Vec<SearchHit>` | Explicit `ef` |
| `len` / `is_empty` | — | Node count |
| `entry_point` | `(&self) -> Option<u32>` | Top-level entry node |
| `metric` | `(&self) -> Metric` | Metric the index is scored under |

### `SearchHit`

```rust
pub struct SearchHit {
    pub id: u32,
    pub distance: f32,
}
```

Distance is always *smaller is better*, per the metric convention:
squared L2 for `Metric::L2`, cosine distance for `Metric::Cosine`, and
negated dot product for `Metric::Dot`.

## Insertion

```rust
let mut index = HnswIndex::with_defaults(Metric::L2);
index.insert(Vector::new(vec![1.0, 0.0, 0.0]));
```

Insertion is `O(ef_construction · log M)` distance evaluations amortized.
Ids are assigned sequentially from 0 unless `insert_with_id` is used;
re-inserting an existing id is undefined behavior by contract — the caller
must deduplicate.

## Search

```rust
let hits = index.search(&[1.0, 0.0, 0.0], 5);
for hit in &hits {
    println!("{}  {:.6}", hit.id, hit.distance);
}
```

`search` returns at most `k` hits in ascending distance order. An empty
index returns `Vec::new()`. For recall-sensitive queries, raise `ef`
beyond `k` with `search_ef` and keep `k` hits yourself.

## Threading

`HnswIndex` is `Send + Sync` for reads: call `search` from many threads
concurrently. `insert` takes `&mut self` and is not re-entrant.

## Example

```rust
use vectrix_core::distance::Metric;
use vectrix_core::vector::Vector;
use vectrix_index::hnsw::{HnswIndex, HnswParams};

let mut index = HnswIndex::new(Metric::Cosine, HnswParams::default());
for i in 0..1_000 {
    let v = Vector::new(vec![i as f32, 1.0, 1.0, 1.0]);
    index.insert(v);
}
let hits = index.search(&[999.5, 1.0, 1.0, 1.0], 3);
assert_eq!(hits[0].id, 999);
```
