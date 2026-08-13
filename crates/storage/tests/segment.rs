// ─────────────────────────────────────────────────────────────
// VECTRIX · storage integration tests: segment roundtrip
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Writer → file → mmap-reader roundtrip, plus corruption handling.

use std::path::PathBuf;

use vectrix_core::distance::Metric;
use vectrix_core::vector::Vector;
use vectrix_core::Error;
use vectrix_storage::segment::Segment;
use vectrix_storage::writer::SegmentWriter;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("vectrix-storage-test");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{name}-{}", std::process::id()))
}

#[test]
fn writer_then_reader_roundtrips() {
    let path = scratch("roundtrip.seg");
    let mut writer = SegmentWriter::new(&path, 4, Metric::Cosine);
    writer.push(10, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    writer.push(20, &[5.0, 6.0, 7.0, 8.0]).unwrap();
    writer.push(30, &[9.0, 10.0, 11.0, 12.0]).unwrap();
    writer.finish().unwrap();

    let segment = Segment::open(&path).unwrap();
    assert_eq!(segment.count(), 3);
    assert_eq!(segment.dim(), 4);
    assert_eq!(segment.metric(), Some(Metric::Cosine));

    let v = segment.read_vector(1).unwrap();
    assert_eq!(v, Vector::new(vec![5.0, 6.0, 7.0, 8.0]));

    let collected: Vec<(u32, Vector)> = segment
        .iter()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(collected.len(), 3);
    assert_eq!(collected[0].0, 10);
    assert_eq!(collected[2].1, Vector::new(vec![9.0, 10.0, 11.0, 12.0]));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn read_vector_out_of_bounds_errors() {
    let path = scratch("oob.seg");
    let mut writer = SegmentWriter::new(&path, 2, Metric::L2);
    writer.push(0, &[0.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let segment = Segment::open(&path).unwrap();
    let err = segment.read_vector(7).unwrap_err();
    assert!(matches!(
        err,
        Error::NotEnoughData { have: 1, need: 8 }
    ));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn open_rejects_corrupt_magic() {
    let path = scratch("corrupt.seg");
    // Write a valid header, then clobber the magic.
    let mut writer = SegmentWriter::new(&path, 2, Metric::L2);
    writer.push(0, &[1.0, 1.0]).unwrap();
    writer.finish().unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] = 0x00;
    bytes[1] = 0x00;
    std::fs::write(&path, &bytes).unwrap();

    let err = Segment::open(&path).unwrap_err();
    assert!(matches!(err, Error::Corrupt(_)));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn open_rejects_truncated_file() {
    let path = scratch("truncated.seg");
    let mut writer = SegmentWriter::new(&path, 8, Metric::L2);
    writer.push(0, &[0.0; 8]).unwrap();
    writer.finish().unwrap();

    // Truncate the data area by 8 bytes.
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 8]).unwrap();

    let err = Segment::open(&path).unwrap_err();
    assert!(matches!(err, Error::Corrupt(_)));
    let _ = std::fs::remove_file(&path);
}
