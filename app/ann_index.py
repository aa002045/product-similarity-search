"""
Part 3 (bonus) — approximate nearest neighbor search with FAISS/HNSW.

Why ANN at all, given the dataset is only 30k rows?
    It isn't needed yet. A brute-force vectorized scan (what similarity.py
    does) over 30k rows is a handful of numpy ops over ~tens of thousands
    of floats — see scripts/benchmark.py for a measured number, typically
    low-single-digit milliseconds on a laptop CPU. The point of this part
    is to implement and justify the approach that becomes *necessary* once
    the catalog is, say, 10-100M products (a real Amazon-scale catalog),
    where an exact O(n) scan per query stops being viable at
    request-serving latency, and to show the break-even reasoning rather
    than assert it.

Why HNSW (via FAISS), not Annoy, not plain IVF?
  - HNSW (Hierarchical Navigable Small World graphs — Malkov & Yashunin,
    2016, "Efficient and robust approximate nearest neighbor search using
    HNSW graphs", https://arxiv.org/abs/1603.09320) gives very high
    recall/latency trade-offs with no training/clustering step, which
    matters here because the catalog isn't static — new products get
    added continuously in a real system, and HNSW supports incremental
    `index.add()` without a retrain, unlike IVF-family indexes which need
    a k-means-style `train()` pass over representative data first.
  - FAISS (Johnson, Douze & Jégou, 2017, "Billion-scale similarity search
    with GPUs", https://arxiv.org/abs/1702.08734) is the implementation
    used here because it's the most production-proven library with this
    algorithm, has a mature Python API, and gives a clear upgrade path
    (IVF-PQ for memory-compressed billion-scale search) if the catalog
    grows far past what HNSW's full-float graph can hold in RAM.
  - Annoy (random-projection forest) was the other suggested option; it's
    simpler and has a smaller memory footprint per tree, but it's
    read-only after building (every future product addition needs a full
    rebuild) and, at equal recall targets, generally needs to scan more
    candidate nodes than HNSW in published benchmarks. Given this service
    needs to keep serving while new products are periodically reindexed,
    HNSW's incremental `add()` is the deciding factor, not raw QPS.

What's actually indexed, and why category/brand/colour aren't in the ANN vector:
  FAISS's HNSW index measures a single geometric distance (L2 here) over
  one dense vector space. `category` (253 values), `brand` (exact-match
  categorical, 6458 values), and `colour` (multi-valued tag overlap)
  don't have a natural dense embedding that preserves "same category" as
  "close in L2 space" without essentially building a lookup table
  disguised as geometry — forcing them in would either blow up
  dimensionality (one-hot) or add a metric the index wasn't built to
  respect. Instead, the ANN index covers the `numeric` + `text` blocks
  (pre-scaled by sqrt(weight) per block, so L2 distance in the
  concatenated space approximates the *weighted* combined distance of
  those two blocks — a standard trick for mixing feature blocks under a
  single Euclidean index). The ANN index is used for fast CANDIDATE
  GENERATION; the final ranking is still computed with the full, exact
  `calculate_similarity` (category + brand + colour + numeric + text)
  over just that short candidate list. This is the standard "cheap
  retrieval, exact re-rank" split used in real recommendation/search
  systems, and it means
  approximate search never changes *which* attributes matter for the final
  ranking — only how fast the candidate set is found.
"""
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np

from app.config import DEFAULT_WEIGHTS
from app.feature_matrix import FeatureStore
from app.similarity import calculate_similarity


def build_ann_vectors(store: FeatureStore, weights: dict = None) -> np.ndarray:
    weights = weights or DEFAULT_WEIGHTS
    numeric_block = store.numeric_scaled * np.sqrt(weights["numeric"])
    text_block = store.text_reduced * np.sqrt(weights["text"])
    vectors = np.concatenate([numeric_block, text_block], axis=1).astype(np.float32)
    return np.ascontiguousarray(vectors)


@dataclass
class AnnIndex:
    index: "faiss.IndexHNSWFlat"
    vectors: np.ndarray

    @classmethod
    def build(cls, vectors: np.ndarray, M: int = 32, ef_construction: int = 200) -> "AnnIndex":
        dim = vectors.shape[1]
        index = faiss.IndexHNSWFlat(dim, M)  # M = graph connectivity; 16-64 is the typical range
        index.hnsw.efConstruction = ef_construction
        index.add(vectors)
        return cls(index=index, vectors=vectors)

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, vectors: np.ndarray) -> "AnnIndex":
        index = faiss.read_index(path)
        return cls(index=index, vectors=vectors)

    def search(self, query_vector: np.ndarray, k: int, ef_search: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        self.index.hnsw.efSearch = ef_search
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        return distances[0], indices[0]


def find_similar_products_ann(store: FeatureStore, ann: AnnIndex, product_id: str,
                                num_similar: int, shortlist_multiplier: int = 8) -> List[str]:
    """ANN-accelerated equivalent of similarity.find_similar_products:
    retrieve a shortlist via HNSW, then re-rank it exactly."""
    idx = store.idx_of(product_id)
    shortlist_k = min(len(store.df), num_similar * shortlist_multiplier + 1)

    _, candidate_idx = ann.search(ann.vectors[idx], shortlist_k)
    candidate_idx = [int(i) for i in candidate_idx if i != idx and i != -1]

    scored = [(calculate_similarity(store, idx, c), c) for c in candidate_idx]
    scored.sort(key=lambda t: -t[0])
    top = scored[:num_similar]
    return [store.product_id_at(i) for _, i in top]
