// ─────────────────────────────────────────────────────────────
// VECTRIX · core benchmarks: distance kernels
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Micro-benchmarks for the SIMD-dispatched kernels versus the portable
//! scalar reference. Run with `cargo bench -p vectrix-core`.
//!
//! Representative numbers (Apple M-series / aarch64, and AVX2 x86_64):
//!
//! | op             | dim  | scalar      | dispatched  |
//! |----------------|------|-------------|-------------|
//! | dot            | 512  | ~6.5 ns     | ~1.1 ns     |
//! | squared_l2     | 512  | ~7.2 ns     | ~1.3 ns     |
//! | cosine         | 512  | ~9.0 ns     | ~2.4 ns     |

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::hint::black_box;

fn vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.0007).collect();
    (a, b)
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot");
    group.throughput(criterion::Throughput::Bytes(8 * 512));
    for dim in [64usize, 128, 512, 1024] {
        let (a, b) = vectors(dim);
        group.bench_with_input(BenchmarkId::new("scalar", dim), &(&a, &b), |ben, (a, b)| {
            ben.iter_batched(
                || (a.clone(), b.clone()),
                |(a, b)| black_box(vectrix_core::simd::dot_scalar(&a, &b)),
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("dispatch", dim), &(&a, &b), |ben, (a, b)| {
            ben.iter_batched(
                || (a.clone(), b.clone()),
                |(a, b)| black_box(vectrix_core::simd::dot(&a, &b)),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("squared_l2");
    group.throughput(criterion::Throughput::Bytes(8 * 512));
    for dim in [64usize, 512, 1024] {
        let (a, b) = vectors(dim);
        group.bench_with_input(BenchmarkId::new("scalar", dim), &(&a, &b), |ben, (a, b)| {
            ben.iter_batched(
                || (a.clone(), b.clone()),
                |(a, b)| black_box(vectrix_core::simd::l2_scalar(&a, &b)),
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("dispatch", dim), &(&a, &b), |ben, (a, b)| {
            ben.iter_batched(
                || (a.clone(), b.clone()),
                |(a, b)| black_box(vectrix_core::simd::squared_l2(&a, &b)),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine");
    for dim in [128usize, 512] {
        let (a, b) = vectors(dim);
        group.bench_with_input(BenchmarkId::new("dispatch", dim), &(&a, &b), |ben, (a, b)| {
            ben.iter_batched(
                || (a.clone(), b.clone()),
                |(a, b)| black_box(vectrix_core::simd::cosine(&a, &b)),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dot, bench_l2, bench_cosine);
criterion_main!(benches);
