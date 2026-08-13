# ─────────────────────────────────────────────────────────────
# VECTRIX · container for the quickstart example
# SPDX-License-Identifier: MIT
# ─────────────────────────────────────────────────────────────

# Multi-stage build: compile the example once, ship only the binary.
FROM rust:1.90-slim AS builder

WORKDIR /build
COPY . .

RUN cargo build --release -p vectrix-index --example quickstart

FROM debian:bookworm-slim

COPY --from=builder /build/target/release/examples/quickstart /usr/local/bin/vectrix-demo

ENTRYPOINT ["/usr/local/bin/vectrix-demo"]
