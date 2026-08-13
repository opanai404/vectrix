# Architecture

Vectrix is a dependency-free embedded vector search engine. This document
walks through the crate layout, the two index families, the SIMD dispatch,
and the storage format, and shows how a query flows through the system.

## Crate map

```
┌──────────────────────────────────────────────────────────────────────┐
│                          vectrix workspace                           │
├───────────────┬───────────────┬───────────────┬──────────┬───────────┤
│   core        │   quant       │   index       │ storage  │   wasm    │
│  0 deps       │  1 dep        │  2 deps       │ 1 dep    │ 2 deps    │
│  vector       │  PQ codebooks │  HNSW graph   │  mmap    │ bindgen   │
│  Metric       │  scalar i8    │  IVF-PQ lists │ segments│ JS export │
│  SIMD kernels │  k-means      │  query        │  writer  │           │
└───────────────┴───────────────┴───────────────┴──────────┴───────────┘
```

The dependency rule is strict and deliberate: `core` depends on nothing,
`quant` depends on `core`, `index` depends on `core` + `quant`,
`storage` depends on `core` (+ `libc` for the raw mmap syscalls), and
`wasm` depends on `index` (+ `wasm-bindgen` glue).

## Build path and query path

```
BUILD                                          QUERY
─────                                          ─────

 raw vectors ──► quant: train centroids ──► index: HNSW / IVF-PQ
                     and PQ codebooks            │
                         │                       │  query vector
                         ▼                       ▼
                     residual codes ──► storage: segment ──► core: distance
                                     (mmap, ready to          kernels (SIMD)
                                      mmap-read later)            │
                                                                  ▼
                                                       top-k SearchHit set
```

- **Build.** Vectors are fed into an index. The HNSW index wires a
  hierarchical graph; the IVF-PQ index trains centroids and a product
  quantizer, assigns each vector to a centroid, encodes the residual, and
  appends the code to the centroid's inverted list. Segments are written
  out by `vectrix-storage`.
- **Query.** A query vector enters `core` distance kernels. For HNSW the
  search descends the graph level by level, then runs a bounded beam
  (`ef`). For IVF-PQ it probes the `nprobe` closest centroids and scores
  encoded vectors with an asymmetric distance table. The best `k` hits are
  returned as `SearchHit { id, distance }`.

## SIMD dispatch (`vectrix-core::simd`)

The kernels are hand-written against `std::arch` and selected at runtime:

```
dot / squared_l2
    ├── x86_64   → AVX2 + FMA (detected via is_x86_feature_detected!) · 8 × f32
    ├── aarch64  → NEON (architectural baseline)                      · 4 × f32
    └── fallback → portable scalar with explicit tail loop
```

Long vectors are consumed in whole SIMD lanes and the remainder is folded
in with a scalar tail loop, so no lane is skipped and no alignment
assumption is made. The kernels are `unsafe fn` gated by
`#[target_feature]`; the public safe wrappers re-assert equal lengths in
debug builds.

## HNSW (`vectrix-index::hnsw`)

- Each node owns its vector and one neighbor list per graph level.
- Level assignment follows `floor(-ln(U) / ln(m))`, capped at
  `MAX_LEVEL = 32`, using the workspace's deterministic `XorShift64` PRNG.
- Insertion descends greedily from the top level, then at each level runs
  `ef_construction`-bounded beam search and connects via `select_diverse`,
  which keeps close candidates while favoring geometric spread.
- Links are pruned to `M` (level 0) / `2M` (upper levels).

## IVF-PQ (`vectrix-index::ivf`)

- Centroids are trained with k-means over the raw vectors; the product
  quantizer is trained on residuals (`v - centroid`).
- Search builds, per probed centroid, an asymmetric distance table for the
  query residual, then scores each encoded vector as the sum of table
  lookups — never decoding.
- `search_exact` scans stored originals and is used as the brute-force
  baseline and for re-ranking.

## Storage (`vectrix-storage`)

Segments are read-only, memory-mapped files with a validated 24-byte
header (magic, version, header length, dim, count, metric) followed by
fixed-width records. See [storage-format.md](storage-format.md) for the
byte layout. Query reads are direct pointer dereferences into the mapped
region; `madvise(MADV_WILLNEED)` is exposed for sequential scans.

## WASM (`vectrix-wasm`)

A thin `wasm-bindgen` facade (`new` / `add` / `search`) over the HNSW
index. `Vec<f32>` arguments map to `Float32Array` in JavaScript; search
results are returned as an array of `{ id, distance }` objects.
