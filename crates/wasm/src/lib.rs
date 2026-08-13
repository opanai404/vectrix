// ─────────────────────────────────────────────────────────────
// VECTRIX · wasm: browser bindings
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! `wasm-bindgen` facade over the HNSW index for browser use.
//!
//! ```js
//! import init, { Vectrix } from "./vectrix_wasm.js";
//! await init();
//!
//! const index = new Vectrix(1 /* cosine */);
//! index.add(new Float32Array([0.1, 0.2, 0.3]));
//! index.add(new Float32Array([0.9, 0.8, 0.7]));
//!
//! const hits = index.search(new Float32Array([0.12, 0.21, 0.3]), 2);
//! // hits: [{ id: 0, distance: 0.0013 }, { id: 1, distance: 0.0945 }]
//! ```
//!
//! `Vec<f32>` arguments map to `Float32Array`; search results are an array
//! of `{ id, distance }` objects. All scoring runs on the WASM SIMD
//! kernels built into `vectrix-core`.

use wasm_bindgen::prelude::*;

use vectrix_core::distance::Metric;
use vectrix_index::hnsw::HnswIndex;

/// Metric selector used by the JS constructor.
/// `0` = squared L2, `1` = cosine, `2` = dot.
#[wasm_bindgen]
pub fn metric_from_u8(v: u8) -> u8 {
    match Metric::from_u8(v) {
        Some(_) => v,
        None => 0,
    }
}

/// A JavaScript-facing HNSW index.
#[wasm_bindgen]
pub struct Vectrix {
    index: HnswIndex,
}

#[wasm_bindgen]
impl Vectrix {
    /// Creates an empty index. `metric` selects the scoring metric
    /// (`0` = L2, `1` = cosine, `2` = dot); unknown values fall back to L2.
    #[wasm_bindgen(constructor)]
    pub fn new(metric: u8) -> Vectrix {
        let metric = match metric {
            1 => Metric::Cosine,
            2 => Metric::Dot,
            _ => Metric::L2,
        };
        Vectrix {
            index: HnswIndex::with_defaults(metric),
        }
    }

    /// Appends a vector and returns its id.
    pub fn add(&mut self, vector: Vec<f32>) -> u32 {
        self.index.insert(vector.into())
    }

    /// Returns the `k` nearest neighbors of `query` as an array of
    /// `{ id, distance }` objects, ascending by distance.
    pub fn search(&self, query: Vec<f32>, k: usize) -> Result<js_sys::Array, JsValue> {
        let hits = self.index.search(&query, k);
        let out = js_sys::Array::new_with_length(hits.len() as u32);
        for (i, hit) in hits.iter().enumerate() {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &JsValue::from_str("id"), &JsValue::from_f64(hit.id as f64))
                .map_err(|e| JsValue::from(e))?;
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("distance"),
                &JsValue::from_f64(hit.distance as f64),
            )
            .map_err(|e| JsValue::from(e))?;
            out.set(i as u32, JsValue::from(obj));
        }
        Ok(out)
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}
