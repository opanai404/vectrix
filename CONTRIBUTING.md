# Contributing to Vectrix

Thanks for considering a contribution to Vectrix. The project aims to be a
small, careful, dependency-light vector search engine — contributions that
keep that spirit are very welcome.

## Code of conduct

This project adheres to the Contributor Covenant (see
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)). By participating you agree to
uphold its terms.

## Development setup

Requirements:

- Rust 1.90 or newer (see `rust-toolchain.toml`; `rustup` will pick it up
  automatically).
- `cargo-fmt` and `clippy` components: `rustup component add rustfmt clippy`
- For the wasm target: `rustup target add wasm32-unknown-unknown`
- Optional: `cargo-deny` for the dependency audit (`make deny`).

Fork, clone, and verify the baseline:

```sh
cargo build --workspace --all-targets
cargo test --workspace --all-features
```

## Workflow

1. Open an issue first for anything non-trivial so we can agree on the
   shape of the change before code is written.
2. Create a branch off `main`. Use a name that describes the change:
   `feat/pq-gpu-training`, `fix/storage-header-validate`.
3. Implement the change with tests. For bug fixes, add a regression test
   that fails on `main` and passes on your branch.
4. Run the full quality gate locally:
   ```sh
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   cargo test --workspace --all-features
   ```
5. Open a pull request against `main`. Reference the issue it closes and
   summarize the approach, any tradeoffs, and the benchmark impact (we care
   a lot about the micro-benchmarks in `crates/core/benches/`).

## Code style

- `rustfmt` with the repository `rustfmt.toml` (100-column limit).
- `clippy -D warnings` must pass.
- Prefer zero new runtime dependencies. If a change genuinely needs one,
  discuss it in the issue first — the "dependency-free core" property is a
  deliberate part of the project identity.
- Keep unsafe contained: SIMD kernels live in `vectrix-core::simd`; new
  `unsafe` must be justified in a `// SAFETY:` comment.
- Every public item gets a doc comment. Match the existing tone: concise,
  technical, no marketing fluff.

## Conventional commits

We use conventional commit messages, which drive the changelog:

```
feat(hnsw): support leveled delete tombstones
fix(storage): validate segment magic before parsing header
perf(core): widen AVX2 dot loop to two accumulators
docs(quant): explain residual encoding for IVF-PQ
test(core): add property test for cosine bounds
```

Types: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `chore`, `ci`.
Scope is the crate or module being touched.

## Testing expectations

- New index code needs both a correctness test (small, deterministic data)
  and a recall assertion against a brute-force baseline.
- Property tests live under `tests/` and use `proptest`; keep the search
  space small so the suite stays fast.
- Benchmarks: if you touch the distance kernels, run
  `cargo bench -p vectrix-core` and report before/after numbers in the PR.

## Licensing

By contributing you agree that your contribution is licensed under the MIT
license as described in [LICENSE](LICENSE).
