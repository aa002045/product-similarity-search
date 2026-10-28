"""
Sanity tests against the real dataset (not mocked) — at 30k rows and a few
hundred ms to build, mocking buys nothing here and risks testing a fake
schema instead of the one the real file actually has.
"""
import sys

sys.path.insert(0, ".")

import pytest

from app.feature_matrix import build_feature_store
from app.similarity import find_similar_products, set_store


@pytest.fixture(scope="module")
def store():
    s = build_feature_store()
    set_store(s)
    return s


def test_returns_requested_count(store):
    pid = store.uniq_ids[0]
    result = find_similar_products(pid, 5)
    assert len(result) == 5


def test_excludes_query_product(store):
    pid = store.uniq_ids[10]
    result = find_similar_products(pid, 10)
    assert pid not in result


def test_results_are_valid_product_ids(store):
    pid = store.uniq_ids[20]
    result = find_similar_products(pid, 5)
    valid_ids = set(store.uniq_ids)
    assert all(r in valid_ids for r in result)


def test_unknown_product_raises(store):
    with pytest.raises(KeyError):
        find_similar_products("not-a-real-id", 5)


def test_num_similar_capped_by_catalog_size(store):
    # Requesting more than exist (minus the query itself) should not crash.
    result = find_similar_products(store.uniq_ids[0], len(store.df) + 1000)
    assert len(result) == len(store.df) - 1


def test_same_brand_products_rank_above_unrelated(store):
    """Weak but meaningful check: two products that share an (uncommon,
    informative) brand should be more similar to each other under
    calculate_similarity than two arbitrary products with different
    brands, text, and price — this is the whole point of the brand block."""
    from app.similarity import calculate_similarity

    brand_counts = {}
    for b in store.brand:
        if b:
            brand_counts[b] = brand_counts.get(b, 0) + 1
    same_brand = next(b for b, c in brand_counts.items() if c >= 2)
    idxs = [i for i, b in enumerate(store.brand) if b == same_brand][:2]

    diff_brand_idx = next(i for i, b in enumerate(store.brand) if b and b != same_brand)

    sim_same = calculate_similarity(store, idxs[0], idxs[1])
    sim_diff = calculate_similarity(store, idxs[0], diff_brand_idx)
    assert sim_same >= sim_diff
