# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08

Initial release. First public, feature-complete slice of the engine.

### Added

- `vectrix-core`: dependency-free primitives — dense `Vector` type, the
  `Metric` enum (L2 / cosine / dot), and hand-written SIMD distance kernels
  (AVX2 + FMA on x86_64, NEON on aarch64) with runtime dispatch and scalar
  fallback.
- `vectrix-quant`: product quantization (8-bit, `m` subquantizers, k-means
  codebook training, asymmetric distance computation) and an int8 scalar
  quantizer.
- `vectrix-index`:
  - HNSW index with random level assignment, bounded `ef` beam search, and
    diverse neighbor selection. Supports L2, cosine, and dot metrics.
  - IVF-PQ index with k-means centroid assignment, residual encoding, and
    `nprobe`-controlled asymmetric distance search, plus an exact
    brute-force scan for baselines and re-ranking.
- `vectrix-storage`: memory-mapped segment format — validated 24-byte
  header, fixed-width records, `msync`/`madvise` integration, and a
  streaming writer.
- `vectrix-wasm`: `wasm-bindgen` exports for building and searching an HNSW
  index from the browser.
- CI: fmt + clippy gate, three-OS test matrix, wasm typecheck, and
  `cargo-deny` audit.
- Benchmarks: Criterion micro-benchmarks for the distance kernels.

[0.1.0]: https://github.com/hrniu/vectrix/releases/tag/v0.1.0
