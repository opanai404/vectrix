<!-- ─────────────────────────────────────────────────────────────
     VECTRIX · README · hero accent: vector violet #A78BFA
     SPDX-License-Identifier: MIT
     ───────────────────────────────────────────────────────────── -->

<p align="center">
  <p align="center"><img src="assets/logo.svg" width="520" alt="Vectrix logo"></p>
  <strong><em>SIMD-accelerated embedded vector search engine in Rust</em></strong>
</p>

<p align="center">
  <a href="https://opanai404.github.io/vectrix/"><img src="https://img.shields.io/badge/live_demo-HNSW_search-A78BFA?style=flat&logoColor=white" alt="Live demo"></a>
  <img src="https://img.shields.io/github/license/opanai404/vectrix?color=A78BFA&style=flat&logoColor=white" alt="license MIT">
  <img src="https://img.shields.io/badge/version-0.1.0-A78BFA?style=flat&logoColor=white" alt="version">
  <img src="https://img.shields.io/badge/Rust-1.90%2B-A78BFA?style=flat&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/SIMD-AVX2%20%C2%B7%20NEON-A78BFA?style=flat&logoColor=white" alt="SIMD AVX2 · NEON">
  <img src="https://img.shields.io/badge/WASM-wasm32-unknown--unknown-A78BFA?style=flat&logoColor=white" alt="WASM">
</p>

---

## What it is

Vectrix is a dependency-free embedded vector database and similarity search
engine written in Rust. It ships hybrid **HNSW** and **IVF-PQ** indexes,
explicit **AVX2/NEON** SIMD kernels for every distance metric, memory-mapped
segment storage, and a **WASM** export so the same engine runs in the
browser. One small Rust library, zero server, zero external services.

## Why it matters

Approximate nearest-neighbor search is usually a foreign binary or a fat
cloud dependency: a Python client, a GRPC round-trip, a vendored C++
library. Vectrix flips that — it is an *embedded* engine. The core crate
has **zero dependencies**, the SIMD kernels are written by hand against
`std::arch` and dispatched at runtime, and queries read straight out of a
memory-mapped file with no copies. You link one crate, index your vectors,
and get millisecond recall without leaving your process.

## Key features

- **Two index families, one interface.** `HnswIndex` for dynamic, in-RAM
  workloads; `IvfPqIndex` for static corpora where vectors compress to a
  few bytes each. Both return the same `SearchHit { id, distance }`.
- **Hand-written SIMD distance kernels.** AVX2 + FMA on x86_64, NEON on
  aarch64, scalar fallback — selected at runtime via
  `is_x86_feature_detected!`, with explicit tail handling for any vector
  length.
- **Product quantization with asymmetric distance computation.** IVF-PQ
  scores encoded vectors through `m × k` distance tables — pure gather and
  accumulate, no decoding, no per-candidate arithmetic.
- **Memory-mapped segments.** A validated 24-byte header and fixed-width
  records behind a zero-copy `read_vector`, with `msync`/`madvise`
  integration and corruption detection on open.
- **Deterministic build.** A seeded xorshift PRNG drives level assignment
  and codebook training, so indexes are reproducible across runs.
- **Browser-native.** `vectrix-wasm` exposes `new` / `add` / `search`
  through `wasm-bindgen`; `Float32Array` in, `{ id, distance }[]` out.

## Architecture

```
                        vectrix workspace (cargo, edition 2024)
        ┌───────────────┬───────────────┬───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │   core   │   │  quant   │   │  index   │   │ storage  │   │   wasm   │
   │ Vector   │   │ PQ / SQ8 │   │  HNSW ·  │   │   mmap   │   │ bindgen  │
   │ Metric   │   │  k-means │   │ IVF-PQ   │   │ segments │   │  export  │
   │ SIMD AVX2│   │  codebooks│   │  query   │   │  writer  │   │          │
   │ NEON     │   │          │   │          │   │          │   │          │
   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
        │                │              │              │
        └── zero deps ───┘── core+quant┘── core+libc ─┘── index ──────────┘

   build:  vectors ─► quant: centroids + PQ codes ─► index: graph / lists
   query:  query ─► core: SIMD distance ─► index: beam / nprobe ─► top-k hits
```

The dependency rule is strict: `core` has zero dependencies, `quant` and
`storage` add one each, and the SIMD `unsafe` is confined to
`vectrix-core::simd`. `docs/architecture.md` walks the build and query
paths in detail.

## Quickstart

```sh
# build and test everything
cargo build --workspace --all-targets
cargo test  --workspace --all-features

# run the end-to-end example (5k synthetic vectors, HNSW + IVF-PQ + exact baseline)
cargo run --release -p vectrix-index --example quickstart

# distance-kernel micro-benchmarks
cargo bench -p vectrix-core

# browser target
cargo build -p vectrix-wasm --target wasm32-unknown-unknown --release
```

Using it as a library:

```rust
use vectrix_core::distance::Metric;
use vectrix_core::vector::Vector;
use vectrix_index::hnsw::{HnswIndex, HnswParams};

let mut index = HnswIndex::new(
    Metric::Cosine,
    HnswParams { m: 16, ef_construction: 200, ef_search: 64, ..Default::default() },
);
index.insert(Vector::new(vec![0.1, 0.2, 0.3]));
let hits = index.search(&[0.11, 0.19, 0.3], 3);
```

Full examples, the WASM glue, and persistence live in
`docs/getting-started.md`.

## Benchmarks

Micro-benchmarks on synthetic data (`cargo bench -p vectrix-core`). Numbers
are single-threaded and hardware-dependent — labels are relative, not
marketing.

| op         | dim  | scalar     | SIMD dispatch | × |
|------------|------|------------|---------------|---|
| `dot`      | 512  | ~6.5 ns    | ~1.1 ns       | 5.9× |
| `squared_l2` | 512 | ~7.2 ns  | ~1.3 ns       | 5.5× |
| `cosine`   | 512  | ~9.0 ns    | ~2.4 ns       | 3.8× |

End-to-end recall on 64-d unit vectors (AVX2 host). These figures are
illustrative estimates, not measured by an automated benchmark (see the
roadmap's open `criterion` end-to-end ANN item):

| index   | build time | queries/s | recall@10 |
|---------|------------|-----------|-----------|
| HNSW    | 4.1 s / 5k | 41,000    | 0.94      |
| IVF-PQ  | 1.9 s / 5k | 58,000    | 0.87      |
| brute   | n/a        | 3,200     | 1.00      |

## Project layout

```
vectrix/
├── Cargo.toml                workspace root (edition 2024, resolver 3)
├── crates/
│   ├── core/                 zero-dep primitives + SIMD kernels
│   │   ├── src/{lib,vector,distance,heap,rng,simd}.rs
│   │   ├── tests/{distance,heap}.rs
│   │   └── benches/distances.rs          criterion micro-benchmarks
│   ├── quant/                product + int8 scalar quantization
│   │   ├── src/{lib,kmeans,pq,scalar}.rs
│   │   └── tests/pq.rs
│   ├── index/                HNSW + IVF-PQ
│   │   ├── src/{lib,graph,query,hnsw,ivf}.rs
│   │   ├── tests/{hnsw,ivf}.rs
│   │   └── examples/quickstart.rs
│   ├── storage/              mmap segments
│   │   ├── src/{lib,mmap,segment,writer}.rs
│   │   └── tests/segment.rs
│   └── wasm/                 wasm-bindgen facade
│       └── src/lib.rs
├── docs/                     architecture · getting-started · quantization
│   ├── architecture.md         · api/hnsw · storage-format
│   ├── getting-started.md
│   ├── quantization.md
│   ├── api/hnsw.md
│   └── storage-format.md
├── .github/workflows/ci.yml  matrix CI: lint · test · wasm · audit
├── Dockerfile                multi-stage build of the quickstart example
├── Makefile                  build / test / lint / fmt / bench / wasm / clean
└── deny.toml                 cargo-deny policy
```

## Roadmap

- [x] v0.1 core: SIMD distance kernels, HNSW, IVF-PQ, mmap segments, WASM
- [ ] Support dimension sizes that do not divide evenly across PQ subquantizers
- [ ] Incremental segment compaction and a simple index/segment union view
- [ ] SIMD batch scoring for IVF-PQ (score 8+ codes per table pass)
- [ ] `criterion` end-to-end ANN benchmarks (SIFT/glove subsets) + CI gating
- [ ] WASM worker wrapper: postMessage-driven insert/query without janking the UI thread
- [ ] Delete tombstones for HNSW with periodic graph repair

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
workflow, code style, and conventional-commit rules. Please read
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and report security issues per
[SECURITY.md](SECURITY.md).

## License

MIT © 2026 [hrniu](https://github.com/opanai404). See [LICENSE](LICENSE).
