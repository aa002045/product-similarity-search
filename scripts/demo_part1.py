"""
Deliverable 1 demo — run this file directly to see find_similar_products()
work against a few real products, with human-readable output (name,
brand, category, price) instead of just a list of IDs.

Run from VS Code: open this file, then click the "Run" (play) button in
the top-right corner, or right-click anywhere in the editor and choose
"Run Python File in Terminal".
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Point at the dataset copy that lives inside this project (no env var
# needed when running this way).
os.environ.setdefault(
    "PRODUCT_DATA_PATH",
    os.path.join(
        os.path.dirname(__file__), "..", "data",
        "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson",
    ),
)

from app.similarity import find_similar_products, get_store  # noqa: E402


def describe(store, product_id: str) -> str:
    idx = store.idx_of(product_id)
    row = store.df.iloc[idx]
    brand = row["brand_clean"] or "-"
    category = row["category_clean"] or "-"
    price = row["sales_price"]
    name = row["product_name"][:65]
    return f"[{brand}][{category}] {name} | price={price}"


def main():
    print("Loading dataset and building the feature store (TF-IDF, scaling)...")
    store = get_store()
    print(f"Loaded {len(store.df)} products.\n")

    # A few real product_ids from the dataset, picked to show different
    # categories (fashion top, saree, polo).
    sample_ids = [
        "26d41bdc1495de290bc8e6062d927729",
        store.uniq_ids[100],
        store.uniq_ids[5000],
    ]

    for pid in sample_ids:
        print("=" * 90)
        print("QUERY PRODUCT:")
        print(" ", describe(store, pid))
        print(f"\nTop 5 similar products to {pid}:")
        for rank, result_id in enumerate(find_similar_products(pid, 5), start=1):
            print(f"  {rank}. {describe(store, result_id)}")
        print()


if __name__ == "__main__":
    main()
