// ─────────────────────────────────────────────────────────────
// VECTRIX · storage: segment writer
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! The write side of the segment format. Records accumulate in memory and
//! are written in a single pass on [`SegmentWriter::finish`], followed by
//! an `fsync` so a crash cannot leave a half-visible segment.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use vectrix_core::distance::Metric;
use vectrix_core::{Error, Result};

use crate::segment::{SEGMENT_VERSION, SegmentHeader};

/// Accumulates records and writes a complete segment on finish.
#[derive(Debug)]
pub struct SegmentWriter {
    path: PathBuf,
    dim: usize,
    metric: Metric,
    data: Vec<u8>,
    count: u32,
}

impl SegmentWriter {
    /// Prepares a writer for a segment of `dim`-dimensional vectors under
    /// `metric`.
    pub fn new(path: impl AsRef<Path>, dim: usize, metric: Metric) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            dim,
            metric,
            data: Vec::new(),
            count: 0,
        }
    }

    /// Appends one record. The vector must have exactly `dim` components.
    pub fn push(&mut self, id: u32, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::InvalidArgument(format!(
                "vector has {} components, segment dim is {}",
                vector.len(),
                self.dim
            )));
        }
        self.data.extend_from_slice(&id.to_le_bytes());
        for x in vector {
            self.data.extend_from_slice(&x.to_le_bytes());
        }
        self.count += 1;
        Ok(())
    }

    /// Number of records accumulated so far.
    pub fn len(&self) -> usize {
        self.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Writes the header and all records, flushes, and fsyncs. Consumes
    /// the writer. Leaves no partial segment behind on failure of the
    /// create — callers should write to a temp path and rename.
    pub fn finish(mut self) -> Result<()> {
        let header = SegmentHeader {
            dim: self.dim as u32,
            count: self.count,
            metric: self.metric as u8,
            version: SEGMENT_VERSION,
        };

        let file = File::create(&self.path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);

        out.write_all(&header.encode())?;
        out.write_all(&self.data)?;
        out.flush()?;
        out.get_ref().sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_length_vector() {
        let mut w = SegmentWriter::new("x.seg", 3, Metric::L2);
        let err = w.push(0, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn len_tracks_pushes() {
        let mut w = SegmentWriter::new("x.seg", 2, Metric::L2);
        assert!(w.is_empty());
        w.push(0, &[1.0, 2.0]).unwrap();
        w.push(1, &[3.0, 4.0]).unwrap();
        assert_eq!(w.len(), 2);
    }
}
