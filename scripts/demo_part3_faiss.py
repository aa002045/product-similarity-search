"""
Part 3 (bonus) demo — run this file directly to SEE the FAISS/HNSW
approximate nearest neighbor index actually being built and queried,
compared side-by-side against the exact brute-force search from Part 1.

Run from VS Code: open this file, click the "Run" (play) button in the
top-right corner, or right-click in the editor -> "Run Python File in
Terminal". Takes about 10-20 seconds (building the index is the slow part;
each individual query after that is near-instant).
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault(
    "PRODUCT_DATA_PATH",
    os.path.join(
        os.path.dirname(__file__), "..", "data",
        "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson",
    ),
)

from app.ann_index import AnnIndex, build_ann_vectors, find_similar_products_ann  # noqa: E402
from app.feature_matrix import build_feature_store  # noqa: E402
from app.similarity import find_similar_products, set_store  # noqa: E402


def describe(store, product_id: str) -> str:
    idx = store.idx_of(product_id)
    row = store.df.iloc[idx]
    brand = row["brand_clean"] or "-"
    category = row["category_clean"] or "-"
    name = row["product_name"][:55]
    return f"[{brand}][{category}] {name}"


def main():
    print("1. Loading dataset + building feature store (same as Part 1)...")
    store = build_feature_store()
    set_store(store)
    print(f"   Loaded {len(store.df)} products.\n")

    print("2. Building the FAISS HNSW index (this is the actual Part 3 step)...")
    t0 = time.time()
    vectors = build_ann_vectors(store)
    ann = AnnIndex.build(vectors)
    print(f"   Built index over {vectors.shape[0]} vectors, {vectors.shape[1]} dimensions, "
          f"in {time.time() - t0:.2f}s.\n")

    # --- Side-by-side: exact (Part 1) vs ANN (Part 3) on a real query ---
    query_id = "26d41bdc1495de290bc8e6062d927729"
    k = 5
    print(f"3. Comparing exact vs. ANN search for one real query (top {k}):")
    print("   QUERY:", describe(store, query_id))

    exact_results = find_similar_products(query_id, k)
    ann_results = find_similar_products_ann(store, ann, query_id, k)

    print(f"\n   EXACT (brute-force, Part 1):")
    for pid in exact_results:
        print("    -", describe(store, pid))

    print(f"\n   ANN (FAISS/HNSW, Part 3):")
    for pid in ann_results:
        print("    -", describe(store, pid))

    overlap = len(set(exact_results) & set(ann_results))
    print(f"\n   Overlap: {overlap}/{k} results match between exact and ANN search.")

    # --- Timing + recall over many queries (the numbers in the README) ---
    print("\n4. Running the full benchmark (200 queries, k=10) — this is where the")
    print("   README's numbers come from:")
    random.seed(42)
    sample_ids = random.sample(list(store.uniq_ids), 200)
    K = 10

    t0 = time.perf_counter()
    brute_results = {pid: find_similar_products(pid, K) for pid in sample_ids}
    brute_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    ann_results_all = {pid: find_similar_products_ann(store, ann, pid, K) for pid in sample_ids}
    ann_elapsed = time.perf_counter() - t0

    recalls = [
        len(set(brute_results[pid]) & set(ann_results_all[pid])) / K
        for pid in sample_ids
    ]
    avg_recall = sum(recalls) / len(recalls)

    print(f"   Brute-force:  {brute_elapsed*1000:7.1f} ms total  ({brute_elapsed/200*1000:.2f} ms/query)")
    print(f"   ANN (HNSW):   {ann_elapsed*1000:7.1f} ms total  ({ann_elapsed/200*1000:.2f} ms/query)")
    print(f"   Recall@{K}:    {avg_recall:.3f}  (fraction of exact top-{K} that ANN also found)")


if __name__ == "__main__":
    main()
