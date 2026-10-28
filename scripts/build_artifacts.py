#!/usr/bin/env python3
"""
Offline precompute step: build the feature store + FAISS index once, and
persist both to disk (cache/). Run this whenever the underlying dataset
changes; the FastAPI service and Part-1 functions can then either rebuild
from the raw file (simple, ~seconds, what main.py does today) or load these
artifacts directly (what you'd want once the dataset is large enough that
"just recompute TF-IDF on every pod startup" stops being free).

This is the "caching strategy" referenced in the assignment applied at the
catalog level (as opposed to app/cache.py's per-request result cache, and
app/image_features.py's per-image embedding cache — three different cache
layers for three different things that go stale on different schedules).
"""
import pickle
import sys
import time

sys.path.insert(0, ".")  # allow `python scripts/build_artifacts.py` from repo root

from app.ann_index import AnnIndex, build_ann_vectors
from app.config import FAISS_INDEX_PATH, FEATURE_MATRIX_PATH
from app.feature_matrix import build_feature_store


def main():
    t0 = time.time()
    store = build_feature_store()
    print(f"Built feature store for {len(store.df)} products in {time.time() - t0:.2f}s")

    with open(FEATURE_MATRIX_PATH.replace(".npz", ".pkl"), "wb") as f:
        pickle.dump(store, f)
    print(f"Saved feature store -> {FEATURE_MATRIX_PATH.replace('.npz', '.pkl')}")

    t0 = time.time()
    vectors = build_ann_vectors(store)
    ann = AnnIndex.build(vectors)
    ann.save(FAISS_INDEX_PATH)
    print(f"Built + saved FAISS HNSW index ({vectors.shape}) in {time.time() - t0:.2f}s -> {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
