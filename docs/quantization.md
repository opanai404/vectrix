# Quantization

Vectrix ships two quantizers in `vectrix-quant`, both of which exist to
shrink memory footprint and, for the IVF-PQ index, to make similarity
search over millions of vectors feasible on a single machine.

## Product quantization (PQ)

A product quantizer splits a `d`-dimensional vector into `m` contiguous
subvectors of `d/m` dimensions and quantizes each subvector independently
against its own codebook of `k = 2^bits` centroids. A vector therefore
compresses to `m` bytes (with `bits = 8`), down from `4d` bytes of `f32`.

```
v = [  v0 │ v1 │ ... │ v_{m-1} ]        d = 32, m = 8, d/m = 4
       │    │              │
       ▼    ▼              ▼
    [c0j] [c1j] ...    [c_{m-1}j]      one centroid per subvector
       │    │              │
       └────┴──────┬───────┘
               code = [j0 j1 ... j7]    8 bytes
```

Training runs one Lloyd iteration pass per subvector over the training
residuals (see below), seeded by the index's deterministic `XorShift64`
PRNG so that repeated builds are reproducible.

### Residual encoding (IVF-PQ)

IVF-PQ does not encode raw vectors. Each vector is assigned to its nearest
centroid `c`, and the quantizer is trained and applied to the *residual*

```
r = v - c
```

A query, for a probed centroid `c`, is likewise reduced to the residual
`q - c` before the asymmetric distance table is built.

### Asymmetric distance computation (ADC)

Instead of decoding a code and computing an exact distance, search computes
a distance table `T` of shape `m × k` once per (query, centroid) pair,

```
T[i][j] = || q_i - cb_i[j] ||²
```

and scores each encoded vector as a sum of `m` table lookups:

```
score(code) = Σ_i  T[i][code[i]]
```

This avoids any per-candidate arithmetic — the hot loop is pure gather
loads and an accumulate — and is why IVF-PQ search stays fast even though
the stored codes are byte-sized.

## Scalar quantization (SQ8)

For settings where PQ's reconstruction is too coarse, `ScalarQuantizer`
maps each `f32` to a signed `i8` using a symmetric scale:

```
scale = 127 / max(|min|, |max|)
q     = clamp(round(v · scale), -127, 127)
v'    = q / scale
```

`fit` derives the scale from a sample of the corpus. SQ8 cuts memory 4x
with a much lower reconstruction error than 8-bit PQ on uniform data, at
the cost of no dimension-level adaptivity. It is intended for use as a
pre-filter stage or as a low-footprint exact-ish index.

## Error characteristics

All numbers here are micro-benchmarks on synthetic data and will vary:

| Setup                                | Memory/vector | Mean relative error |
|--------------------------------------|---------------|---------------------|
| `f32` raw                            | 4d bytes      | 0                    |
| SQ8 (`i8`)                           | d bytes       | ~0.5–1%             |
| PQ8, m = d/4                         | d/4 bytes     | ~1–2%               |
| PQ8, m = d/8                         | d/8 bytes     | ~2–5%               |

The standard tradeoff holds: fewer subquantizers (`m`) means smaller codes
and worse reconstruction; increase `m` for accuracy, decrease it for
footprint.
