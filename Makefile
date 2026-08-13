# ─────────────────────────────────────────────────────────────
# VECTRIX · developer makefile
# SPDX-License-Identifier: MIT
# ─────────────────────────────────────────────────────────────

CARGO   ?= cargo
TARGET   = wasm32-unknown-unknown
DENY    ?= cargo deny

.PHONY: all build test lint fmt bench wasm clean docs check deny

all: build

## Build the whole workspace with all targets and features.
build:
	$(CARGO) build --workspace --all-targets --all-features

## Run the full test suite (unit + integration + doc tests).
test:
	$(CARGO) test --workspace --all-features

## Rustfmt check + clippy with warnings denied.
lint:
	cargo fmt --all -- --check
	$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings

## Apply rustfmt to the workspace.
fmt:
	cargo fmt --all

## Criterion micro-benchmarks (distance kernels).
bench:
	$(CARGO) bench -p vectrix-core

## Build the wasm-bindgen target.
wasm:
	$(CARGO) build -p vectrix-wasm --target $(TARGET) --release

## Run the quickstart example.
run-example:
	$(CARGO) run --release -p vectrix-index --example quickstart

## cargo-deny dependency/advisory audit.
deny:
	$(DENY) check --workspace

## Build rustdoc and open it.
docs:
	$(CARGO) doc --workspace --no-deps --open

## Remove build artifacts.
clean:
	$(CARGO) clean
	rm -rf target/wasm32-unknown-unknown wasm/pkg

## Check hygiene: any lingering placeholder markers anywhere in the tree.
check-placeholders:
	! grep -rInE 'TODO: implement|FIXME: implement|lorem ipsum|XXX placeholder' --include='*.rs' --include='*.md' --include='*.toml' --include='*.yml' --include='*.yaml' .
