// ─────────────────────────────────────────────────────────────
// VECTRIX · core: vectors, metrics, and distance kernels
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Core primitives shared by every Vectrix index: the dense [`Vector`]
//! type, the [`distance::Metric`] enum that drives all scoring, and the
//! hand-written SIMD kernels in [`simd`].
//!
//! This crate has **zero dependencies**. The SIMD kernels are written
//! against `std::arch` and selected at runtime, so the same binary runs
//! on AVX2-capable x86_64, on aarch64 (NEON), and on anything else via
//! the portable scalar fallback.

pub mod distance;
pub mod heap;
pub mod rng;
pub mod simd;
pub mod vector;

use std::fmt;

/// Error type used across the workspace.
#[derive(Debug)]
pub enum Error {
    /// A caller-supplied argument is invalid (wrong dimension, bad length).
    InvalidArgument(String),
    /// An I/O error while opening, mapping, or writing storage.
    Io(std::io::Error),
    /// The on-disk format is inconsistent or truncated.
    Corrupt(String),
    /// A read or index operation ran past the available data.
    NotEnoughData { have: usize, need: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Error::Io(e) => write!(f, "i/o error: {e}"),
            Error::Corrupt(msg) => write!(f, "corrupt data: {msg}"),
            Error::NotEnoughData { have, need } => {
                write!(f, "need {need} elements, have {have}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Convenience alias for fallible operations across the workspace.
pub type Result<T> = std::result::Result<T, Error>;
