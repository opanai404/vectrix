// ─────────────────────────────────────────────────────────────
// VECTRIX · index: HNSW and IVF-PQ
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! The index layer. Two approximate nearest-neighbor structures share a
//! common [`SearchHit`] result type and the [`distance::Metric`] scoring
//! convention (smaller is better):
//!
//! - [`hnsw::HnswIndex`] — a hierarchical navigable small-world graph.
//!   Best for datasets that fit in RAM and dynamic inserts.
//! - [`ivf::IvfPqIndex`] — inverted-file product quantization. Best for
//!   static corpora where footprint dominates: every vector costs a few
//!   bytes on disk and search uses table-driven approximate scoring.
//!
//! [`graph`] holds the graph primitives shared by both (neighbor
//! selection heuristics and the visited-set pool); [`query`] defines the
//! shared result type.

pub mod graph;
pub mod hnsw;
pub mod ivf;
pub mod query;

pub use graph::{Neighbor, VisitedPool, select_diverse};
pub use hnsw::{HnswIndex, HnswParams};
pub use ivf::{IvfPqIndex, IvfPqParams};
pub use query::SearchHit;

/// Re-exported error type so index users never need `vectrix-core` for
/// error handling.
pub type Error = vectrix_core::Error;
/// Re-exported result alias.
pub type Result<T> = vectrix_core::Result<T>;
