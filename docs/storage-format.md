# Segment storage format

`vectrix-storage` persists a batch of fixed-dimension vectors as a
read-only, memory-mapped segment file. The format is intentionally simple:
a single validated header followed by tightly packed records. All
multi-byte integers are **little-endian**; floats are IEEE-754 binary32.

## Byte layout

```
 offset   size    field
──────────────────────────────────────────────
 0         4      magic         = 0x5658_5843 ("VXXC")
 4         4      version       = 1
 8         4      header_len    = 24
12         4      dim           = d
16         4      count         = n
20         4      metric        = 0 (L2) | 1 (Cosine) | 2 (Dot)
24         ·      record[0]
 ·         ·      record[1]
 ·         ·      ...
24 + n·r   ·      (end)
```

Each record is fixed width, `r = 4 + 4·d` bytes:

```
 id (u32 LE)  │  v[0] (f32 LE)  │  v[1] (f32 LE)  │  ...  │  v[d-1] (f32 LE)
```

The file length must be exactly `24 + n·r`. `Segment::open` validates the
magic, the version, `header_len`, and the total length before any record
is touched; anything else is rejected as `Error::Corrupt`.

## Reading

`Segment::open` memory-maps the file read-only (`MAP_PRIVATE`) and parses
the header from the first 24 bytes. `read_vector(id)` dereferences the
mapped region at offset `24 + id·r + 4` and decodes `d` floats; ids outside
`[0, n)` return `Error::NotEnoughData`. Because the mapping is read-only,
a malicious or truncated file cannot cause writes.

`madvise(MADV_WILLNEED)` is exposed (`Segment::advise_willneed`) to warm
the page cache before a sequential scan; `MADV_RANDOM` is exposed for
random-id workloads so the kernel stops prefetching.

## Writing

`SegmentWriter` accumulates records in a `Vec<u8>`, then `finish` writes
the header followed by the records in a single pass and `fsync`s the file.
The writer never overwrites in place; a build is either fully written or
leaves no partial segment behind (write to a temp path and rename).

## Versioning

Bumping the record layout requires a new `version`. A reader opens a
segment only when `version` matches its own; the `metric` field is opaque
to storage and interpreted by the caller. There is no cross-version
migration in 0.1.x — rebuild from source vectors.

## Guarantees

- Read path: zero copies, no allocation per `read_vector`.
- Alignment: the base offset `24` is 8-byte aligned, and records are
  `4·(d+1)` bytes, so every record starts on a 4-byte boundary.
- Endianness: fixed at little-endian; wasm32 and all tier-1 targets agree.
