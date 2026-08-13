// ─────────────────────────────────────────────────────────────
// VECTRIX · storage: memory-mapped segments
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Read-only, memory-mapped persistence for vector batches.
//!
//! A [`segment::Segment`] is an mmap of a validated file: a 24-byte
//! header followed by fixed-width `(id, f32…)` records. Queries read
//! straight out of the mapping with zero copies and no per-record
//! allocation. [`writer::SegmentWriter`] produces those files; the raw
//! [`mmap::Mmap`] wrapper is the only place in the workspace that talks
//! to the OS `mmap`/`munmap`/`msync`/`madvise` syscalls.
//!
//! The on-disk format is specified in `docs/storage-format.md`.
//!
//! **Unix-only.** The mapping layer uses POSIX `mmap`/`munmap`/`msync`/
//! `madvise` via `libc`, so this crate is excluded from the Windows CI leg
//! (the rest of the workspace still tests there).

pub mod mmap;
pub mod segment;
pub mod writer;

pub use segment::{Segment, SegmentHeader, HEADER_BYTES, SEGMENT_MAGIC, SEGMENT_VERSION};
pub use writer::SegmentWriter;
