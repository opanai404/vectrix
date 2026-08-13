# Getting started

Vectrix is a Rust workspace; you can either use it as a library dependency
or run the bundled examples.

## Prerequisites

- Rust 1.90+ (the `rust-toolchain.toml` pins the exact channel)
- For the browser target: `rustup target add wasm32-unknown-unknown`

## Build and test

```sh
cargo build --workspace --all-targets
cargo test --workspace --all-features
make lint   # rustfmt + clippy -D warnings
```

## Quickstart (local, native)

```sh
cargo run --release -p vectrix-index --example quickstart
```

The example generates 5,000 synthetic 64-d unit vectors, builds an HNSW
index and an IVF-PQ index over them, and prints the top-10 neighbors of one
query for each index plus the exact brute-force baseline.

## Using Vectrix as a dependency

```toml
[dependencies]
vectrix-index = { version = "0.1", default-features = false }
```

```rust
use vectrix_core::distance::Metric;
use vectrix_core::vector::Vector;
use vectrix_index::hnsw::{HnswIndex, HnswParams};

let mut index = HnswIndex::new(
    Metric::Cosine,
    HnswParams { m: 16, ef_construction: 200, ef_search: 64, ..Default::default() },
);

index.insert(Vector::new(vec![0.1, 0.2, 0.3, 0.4]));
let hits = index.search(&[0.11, 0.21, 0.29, 0.41], 1);
assert_eq!(hits[0].id, 0);
assert!(hits[0].distance < 1e-3);
```

## Persisting an index

Write vectors to a segment, then reopen it later from the mapped file:

```rust
use vectrix_storage::segment::Segment;
use vectrix_storage::writer::SegmentWriter;

let mut writer = SegmentWriter::new("ann.seg", 4, Metric::Cosine);
writer.push(0, &[0.1, 0.2, 0.3, 0.4])?;
writer.finish()?;

let segment = Segment::open("ann.seg")?;
let v = segment.read_vector(0)?;
```

See [storage-format.md](storage-format.md) for the on-disk layout.

## Running in the browser

```sh
cargo build -p vectrix-wasm --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/vectrix_wasm.wasm --out-dir wasm/pkg --target web
```

```js
import init, { Vectrix } from "./wasm/pkg/vectrix_wasm.js";

await init();
const vx = new Vectrix(1 /* cosine */);
vx.add(new Float32Array([0.1, 0.2, 0.3, 0.4]));
const hits = vx.search(new Float32Array([0.11, 0.21, 0.29, 0.41]), 1);
console.log(hits); // [{ id: 0, distance: ... }]
```

## Benchmarks

```sh
make bench   # cargo bench -p vectrix-core
```
