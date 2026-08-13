# Security Policy

Vectrix processes untrusted data — serialized index segments, query vectors,
and, via the WASM build, data coming from the browser — so we take reports
of memory-safety and correctness issues seriously.

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Instead:

- Email `security@hrniu.dev` with the subject prefix `[vectrix]`.
- Alternatively, open a GitHub Security Advisory via the repository's
  **Security → Report a vulnerability** flow.

Include, if possible:

- The affected version(s) and platform(s) (Linux x86_64, macOS aarch64,
  wasm32, ...).
- A minimal reproduction: data shape, index configuration, and the
  operation that triggers the issue.
- Your assessment of impact (e.g. out-of-bounds read, write past a mapped
  region, panic in `unsafe` code reached via crafted input).

You will receive an acknowledgment within 72 hours. We coordinate a fix and
a release, and we credit reporters unless they prefer to stay anonymous.

## Supported versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Security fixes     |
| < 0.1   | End of life        |

Only the latest minor release receives backported security fixes.

## Security-relevant design notes

- `vectrix-storage` validates the segment header (magic, version, declared
  length) before reading records, and every mmap is read-only for the
  query path.
- `vectrix-core::simd` keeps SIMD unsafe strictly inside the kernel
  functions; the public wrappers re-assert length equality in debug builds.
- The WASM target runs with the same validation; it does not add an
  `unsafe` surface of its own.
