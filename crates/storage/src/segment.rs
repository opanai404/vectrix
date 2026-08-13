// ─────────────────────────────────────────────────────────────
// VECTRIX · storage: segment reader
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! The read side of the segment format.
//!
//! `Segment::open` maps a file, validates the header, and exposes
//! `read_vector` access, which decodes each record into an owned `Vector`.
//! All integers are little-endian; floats
//! are IEEE-754 binary32. The byte layout is pinned in
//! `docs/storage-format.md`.

use std::fs::File;
use std::path::Path;

use vectrix_core::distance::Metric;
use vectrix_core::vector::Vector;
use vectrix_core::{Error, Result};

use crate::mmap::{Advice, Mmap, MmapMode};

/// Magic bytes `"VXXC"` in little-endian.
pub const SEGMENT_MAGIC: u32 = 0x5658_5843;
/// Format version written by this crate.
pub const SEGMENT_VERSION: u32 = 1;
/// Fixed header size in bytes.
pub const HEADER_BYTES: usize = 24;

/// Parsed segment header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentHeader {
    /// Component count of each stored vector.
    pub dim: u32,
    /// Number of records.
    pub count: u32,
    /// `Metric as u8` (0 = L2, 1 = Cosine, 2 = Dot).
    pub metric: u8,
    /// Format version.
    pub version: u32,
}

impl SegmentHeader {
    /// Encodes the header into its canonical 24-byte layout.
    pub fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut b = [0u8; HEADER_BYTES];
        b[0..4].copy_from_slice(&SEGMENT_MAGIC.to_le_bytes());
        b[4..8].copy_from_slice(&self.version.to_le_bytes());
        b[8..12].copy_from_slice(&(HEADER_BYTES as u32).to_le_bytes());
        b[12..16].copy_from_slice(&self.dim.to_le_bytes());
        b[16..20].copy_from_slice(&self.count.to_le_bytes());
        b[20..24].copy_from_slice(&u32::from(self.metric).to_le_bytes());
        b
    }

    /// Decodes and validates the header, rejecting wrong magic, unknown
    /// versions, and bogus header lengths.
    pub fn decode(bytes: &[u8]) -> Result<SegmentHeader> {
        if bytes.len() < HEADER_BYTES {
            return Err(Error::NotEnoughData {
                have: bytes.len(),
                need: HEADER_BYTES,
            });
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != SEGMENT_MAGIC {
            return Err(Error::Corrupt(format!(
                "bad magic 0x{magic:08x} (expected VXXC)"
            )));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let header_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != SEGMENT_VERSION {
            return Err(Error::Corrupt(format!(
                "unsupported segment version {version}"
            )));
        }
        if header_len as usize != HEADER_BYTES {
            return Err(Error::Corrupt(format!(
                "header_len {header_len} != {HEADER_BYTES}"
            )));
        }
        let dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let metric = u8::try_from(u32::from_le_bytes(bytes[20..24].try_into().unwrap()))
            .unwrap_or(0);
        Ok(SegmentHeader {
            dim,
            count,
            metric,
            version,
        })
    }

    /// Bytes per record: `u32 id` + `dim × f32`.
    pub fn record_bytes(&self) -> usize {
        4 + self.dim as usize * 4
    }
}

/// A read-only, memory-mapped segment of vectors.
#[derive(Debug)]
pub struct Segment {
    mmap: Mmap,
    header: SegmentHeader,
}

impl Segment {
    /// Opens and validates a segment file.
    pub fn open(path: impl AsRef<Path>) -> Result<Segment> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let mmap = Mmap::map(&file, MmapMode::Read, len)?;
        let header = SegmentHeader::decode(mmap.as_slice())?;

        let expected = HEADER_BYTES + header.count as usize * header.record_bytes();
        if mmap.len() < expected {
            return Err(Error::Corrupt(format!(
                "truncated segment: file is {} bytes, expected {expected}",
                mmap.len()
            )));
        }
        Ok(Segment { mmap, header })
    }

    pub fn header(&self) -> &SegmentHeader {
        &self.header
    }

    pub fn count(&self) -> u32 {
        self.header.count
    }

    pub fn dim(&self) -> u32 {
        self.header.dim
    }

    /// The metric the segment was written under, if it is a known code.
    pub fn metric(&self) -> Option<Metric> {
        Metric::from_u8(self.header.metric)
    }

    /// Advises the kernel to warm the page cache (sequential scans).
    pub fn advise_willneed(&self) -> Result<()> {
        self.mmap.advise(Advice::WillNeed).map_err(Into::into)
    }

    /// Advises the kernel this segment is accessed randomly.
    pub fn advise_random(&self) -> Result<()> {
        self.mmap.advise(Advice::Random).map_err(Into::into)
    }

    /// Decodes record `id` as a vector. Out-of-range ids error with
    /// [`Error::NotEnoughData`].
    pub fn read_vector(&self, id: u32) -> Result<Vector> {
        if id >= self.header.count {
            return Err(Error::NotEnoughData {
                have: self.header.count as usize,
                need: id as usize + 1,
            });
        }
        let off = HEADER_BYTES + id as usize * self.header.record_bytes() + 4;
        let slice = &self.mmap[off..off + self.header.dim as usize * 4];
        let mut data = Vec::with_capacity(self.header.dim as usize);
        for chunk in slice.chunks_exact(4) {
            data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Vector::new(data))
    }

    /// Iterates `(id, vector)` pairs over the whole segment.
    pub fn iter(&self) -> impl Iterator<Item = Result<(u32, Vector)>> + '_ {
        (0..self.header.count).map(|id| self.read_vector(id).map(|v| (id, v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrips_through_bytes() {
        let h = SegmentHeader {
            dim: 4,
            count: 12,
            metric: 1,
            version: SEGMENT_VERSION,
        };
        let decoded = SegmentHeader::decode(&h.encode()).unwrap();
        assert_eq!(decoded, h);
        assert_eq!(decoded.record_bytes(), 4 + 4 * 4);
    }

    #[test]
    fn decode_rejects_wrong_magic() {
        let mut bytes = SegmentHeader {
            dim: 4,
            count: 0,
            metric: 0,
            version: SEGMENT_VERSION,
        }
        .encode();
        bytes[0] = 0xFF;
        let err = SegmentHeader::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)));
    }

    #[test]
    fn decode_rejects_unknown_version() {
        let h = SegmentHeader {
            dim: 4,
            count: 0,
            metric: 0,
            version: 99,
        };
        let err = SegmentHeader::decode(&h.encode()).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)));
    }
}
