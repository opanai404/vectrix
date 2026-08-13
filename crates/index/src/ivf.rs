// ─────────────────────────────────────────────────────────────
// VECTRIX · index: IVF-PQ index
// SPDX-License-Identifier: MIT
// ─────────────────────────────────────────────────────────────

//! Inverted-file index over product-quantized residuals.
//!
//! Vectors are assigned to the nearest of `nlist` k-means centroids; each
//! vector is then stored as an `m`-byte product-quantized *residual*
//! (`v - centroid`) in the centroid's inverted list. Search probes the
//! `nprobe` closest centroids and scores encoded vectors with asymmetric
//! distance tables — no decoding, no arithmetic per candidate beyond a
//! gather-accumulate. `search_exact` provides the brute-force baseline and
//! can re-rank approximate hits.
//!
//! See `docs/quantization.md` for the ADC math.

use vectrix_core::distance::Metric;
use vectrix_core::heap::TopK;
use vectrix_core::rng::XorShift64;
use vectrix_core::vector::Vector;
use vectrix_quant::kmeans::kmeans;
use vectrix_quant::{ProductQuantizer, ProductQuantizerError};

use crate::query::SearchHit;

/// Configuration for the IVF-PQ index.
#[derive(Clone, Copy, Debug)]
pub struct IvfPqParams {
    /// Number of inverted lists (centroids).
    pub nlist: usize,
    /// Number of lists probed per search by default.
    pub nprobe: usize,
    /// Product-quantizer subquantizers.
    pub m: usize,
    /// Code bits per subquantizer (1..=8; 8 is standard).
    pub k_bits: u8,
    /// Lloyd iterations for both centroid and codebook training.
    pub k_means_iters: usize,
    /// Seed for reproducible training.
    pub seed: u64,
}

impl Default for IvfPqParams {
    fn default() -> Self {
        Self {
            nlist: 1024,
            nprobe: 16,
            m: 32,
            k_bits: 8,
            k_means_iters: 24,
            seed: 0x1C0FFEE,
        }
    }
}

/// The IVF-PQ index. Built eagerly from a corpus; immutable thereafter.
#[derive(Debug)]
pub struct IvfPqIndex {
    params: IvfPqParams,
    metric: Metric,
    dim: usize,
    centroids: Vec<Vector>,
    pq: ProductQuantizer,
    /// Per centroid: stored vector ids.
    lists: Vec<Vec<u32>>,
    /// Per centroid: packed `m`-byte codes, one per entry in `lists`.
    codes: Vec<Vec<u8>>,
    /// Original vectors, kept for exact scoring and re-ranking.
    vectors: Vec<Vector>,
}

impl IvfPqIndex {
    /// Builds the index over `vectors`, training centroids and the product
    /// quantizer on the residuals. Requires `vectors.len() >= nlist`.
    pub fn build(
        metric: Metric,
        params: IvfPqParams,
        vectors: Vec<Vector>,
    ) -> Result<Self, IvfPqBuildError> {
        let n = vectors.len();
        if n == 0 {
            return Err(IvfPqBuildError::EmptyCorpus);
        }
        if n < params.nlist {
            return Err(IvfPqBuildError::TooFewVectors {
                corpus: n,
                nlist: params.nlist,
            });
        }
        let dim = vectors[0].dim();
        let mut rng = XorShift64::new(params.seed);

        // 1. Train centroids on the raw vectors.
        let mut data = Vec::with_capacity(n * dim);
        for v in &vectors {
            debug_assert_eq!(v.dim(), dim);
            data.extend_from_slice(v.as_slice());
        }
        let flat = kmeans(&data, n, dim, params.nlist, params.k_means_iters, &mut rng);
        let centroids: Vec<Vector> = flat
            .chunks_exact(dim)
            .map(|c| Vector::new(c.to_vec()))
            .collect();

        // 2. Assign vectors, compute residuals, train the PQ on them.
        let mut assignments = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);
        for v in &vectors {
            let ci = argmin_by(metric, v.as_slice(), &centroids);
            assignments.push(ci);
            let residual: Vec<f32> = v
                .as_slice()
                .iter()
                .zip(centroids[ci].as_slice())
                .map(|(x, c)| x - c)
                .collect();
            residuals.push(Vector::new(residual));
        }

        let mut pq = ProductQuantizer::new(dim, params.m, params.k_bits)?;
        pq.train(&residuals, params.k_means_iters, params.seed ^ 0x9E37_79B9)?;

        // 3. Populate inverted lists.
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); params.nlist];
        let mut codes: Vec<Vec<u8>> = vec![Vec::new(); params.nlist];
        for (i, r) in residuals.iter().enumerate() {
            let ci = assignments[i];
            lists[ci].push(i as u32);
            codes[ci].extend_from_slice(&pq.encode(r.as_slice()));
        }

        Ok(Self {
            params,
            metric,
            dim,
            centroids,
            pq,
            lists,
            codes,
            vectors,
        })
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Approximate search over the `nprobe` closest centroids.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchHit> {
        self.search_nprobe(query, k, self.params.nprobe)
    }

    /// Approximate search with an explicit number of probed centroids.
    pub fn search_nprobe(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchHit> {
        if self.vectors.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut topk = TopK::new(k);
        let mut scored: Vec<(f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (self.metric.distance(query, c.as_slice()), i))
            .collect();
        scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let code_len = self.pq.code_len();
        for &(_, ci) in scored.iter().take(nprobe.min(self.params.nlist)) {
            // Query residual relative to this centroid, then ADC table.
            let residual: Vec<f32> = query
                .iter()
                .zip(self.centroids[ci].as_slice())
                .map(|(q, c)| q - c)
                .collect();
            let table = self.pq.dist_table(&residual);
            let list = &self.lists[ci];
            let codes = &self.codes[ci];
            for (off, &id) in list.iter().enumerate() {
                let code = &codes[off * code_len..(off + 1) * code_len];
                let d = self.pq.approx_distance(&table, code);
                topk.push(d, id);
            }
        }

        topk.into_items()
            .into_iter()
            .map(|(distance, id)| SearchHit { id, distance })
            .collect()
    }

    /// Exact brute-force scan over the stored originals. Used as a
    /// baseline and for final re-ranking of approximate hits.
    pub fn search_exact(&self, query: &[f32], k: usize) -> Vec<SearchHit> {
        if self.vectors.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut topk = TopK::new(k);
        for (i, v) in self.vectors.iter().enumerate() {
            topk.push(self.metric.distance(query, v.as_slice()), i as u32);
        }
        topk.into_items()
            .into_iter()
            .map(|(distance, id)| SearchHit { id, distance })
            .collect()
    }

    /// Approximate search followed by an exact re-rank of the top
    /// candidates, keeping `k` hits.
    pub fn search_rerank(&self, query: &[f32], k: usize, nprobe: usize, pool: usize) -> Vec<SearchHit> {
        let pool = pool.max(k);
        let approx = self.search_nprobe(query, pool, nprobe);
        let mut topk = TopK::new(k);
        for hit in approx {
            if let Some(v) = self.vectors.get(hit.id as usize) {
                topk.push(self.metric.distance(query, v.as_slice()), hit.id);
            }
        }
        topk.into_items()
            .into_iter()
            .map(|(distance, id)| SearchHit { id, distance })
            .collect()
    }
}

/// Errors produced while building an [`IvfPqIndex`].
#[derive(Debug)]
pub enum IvfPqBuildError {
    /// No vectors were supplied.
    EmptyCorpus,
    /// Fewer vectors than inverted lists.
    TooFewVectors { corpus: usize, nlist: usize },
    /// The product quantizer rejected the configuration.
    Quantizer(ProductQuantizerError),
}

impl std::fmt::Display for IvfPqBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IvfPqBuildError::EmptyCorpus => write!(f, "cannot build from an empty corpus"),
            IvfPqBuildError::TooFewVectors { corpus, nlist } => {
                write!(f, "corpus has {corpus} vectors but nlist is {nlist}")
            }
            IvfPqBuildError::Quantizer(e) => write!(f, "quantizer error: {e}"),
        }
    }
}

impl std::error::Error for IvfPqBuildError {}

impl From<ProductQuantizerError> for IvfPqBuildError {
    fn from(e: ProductQuantizerError) -> Self {
        IvfPqBuildError::Quantizer(e)
    }
}

/// Index of the centroid with the smallest distance to `query`.
fn argmin_by(metric: Metric, query: &[f32], centroids: &[Vector]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = metric.distance(query, c.as_slice());
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_corpus() {
        let err = IvfPqIndex::build(Metric::L2, IvfPqParams::default(), Vec::new()).unwrap_err();
        assert!(matches!(err, IvfPqBuildError::EmptyCorpus));
    }

    #[test]
    fn build_populates_all_lists() {
        let params = IvfPqParams {
            nlist: 4,
            nprobe: 2,
            m: 4,
            k_bits: 8,
            k_means_iters: 8,
            seed: 3,
        };
        let vectors: Vec<Vector> = (0..32)
            .map(|i| Vector::new(vec![i as f32 / 32.0, (32 - i) as f32 / 32.0]))
            .collect();
        let idx = IvfPqIndex::build(Metric::L2, params, vectors).unwrap();
        assert_eq!(idx.len(), 32);
        let total: usize = idx.lists.iter().map(Vec::len).sum();
        assert_eq!(total, 32);
    }
}
