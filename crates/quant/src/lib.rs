// ─────────────────────────────────────────────────────────────
// VECTRIX · quant: PQ and scalar quantization
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Quantizers that shrink vectors before storage and enable fast
//! asymmetric-distance search:
//!
//! - [`pq::ProductQuantizer`] — the workhorse behind the IVF-PQ index.
//!   Compresses a `d`-dim vector to `m` bytes and supports table-driven
//!   approximate scoring without decoding.
//! - [`scalar::ScalarQuantizer`] — a symmetric int8 (`SQ8`) quantizer for
//!   low-footprint pre-filtering and near-exact storage.
//!
//! See `docs/quantization.md` for the math and the memory/accuracy
//! tradeoffs.

pub mod kmeans;
pub mod pq;
pub mod scalar;

pub use pq::{ProductQuantizer, ProductQuantizerError};
pub use scalar::ScalarQuantizer;
