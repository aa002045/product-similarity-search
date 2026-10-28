#!/usr/bin/env python3
"""
Measures, rather than asserts, the Part-3 trade-off:
  - latency of brute-force (similarity.find_similar_products) vs
    ANN-accelerated (ann_index.find_similar_products_ann) lookups
  - recall@k of the ANN path against the brute-force path's output,
    treated as ground truth

Run: python scripts/build_artifacts.py && python scripts/benchmark.py
"""
import random
import sys
import time

sys.path.insert(0, ".")

from app.ann_index import AnnIndex, build_ann_vectors, find_similar_products_ann
from app.feature_matrix import build_feature_store
from app.similarity import find_similar_products, set_store

N_QUERIES = 200
K = 10


def main():
    store = build_feature_store()
    set_store(store)

    vectors = build_ann_vectors(store)
    ann = AnnIndex.build(vectors)

    random.seed(42)
    sample_ids = random.sample(list(store.uniq_ids), N_QUERIES)

    t0 = time.perf_counter()
    brute_results = {pid: find_similar_products(pid, K) for pid in sample_ids}
    brute_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    ann_results = {pid: find_similar_products_ann(store, ann, pid, K) for pid in sample_ids}
    ann_elapsed = time.perf_counter() - t0

    recalls = []
    for pid in sample_ids:
        truth = set(brute_results[pid])
        got = set(ann_results[pid])
        recalls.append(len(truth & got) / max(len(truth), 1))
    avg_recall = sum(recalls) / len(recalls)

    print(f"Dataset size:              {len(store.df)} products")
    print(f"Queries:                   {N_QUERIES}, k={K}")
    print(f"Brute-force total:         {brute_elapsed*1000:.1f} ms  ({brute_elapsed/N_QUERIES*1000:.2f} ms/query)")
    print(f"ANN (HNSW) total:          {ann_elapsed*1000:.1f} ms  ({ann_elapsed/N_QUERIES*1000:.2f} ms/query)")
    print(f"Recall@{K} (ANN vs exact):  {avg_recall:.3f}")


if __name__ == "__main__":
    main()
