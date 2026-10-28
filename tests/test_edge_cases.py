"""
Edge cases against the real dataset, one level up from
test_preprocessing_edge_cases.py (which tests the parsers in isolation).
These exercise the scoring and API layers under boundary conditions that
are easy to get wrong: self-comparison, a product missing nearly every
attribute, num_similar at the edges of the valid range, and determinism.
"""
import sys

sys.path.insert(0, ".")

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.feature_matrix import build_feature_store
from app.main import app
from app.similarity import calculate_similarity, find_similar_products, set_store


@pytest.fixture(scope="module")
def store():
    s = build_feature_store()
    set_store(s)
    return s


# --- self-similarity ---------------------------------------------------------

def test_product_is_maximally_similar_to_itself(store):
    """calculate_similarity(x, x) must be the maximum attainable score —
    if this ever drifts below 1.0, some component is miscomputing a
    perfect match as imperfect (e.g. a float-precision or normalization
    bug in the cosine/Jaccard terms)."""
    idx = 0
    score = calculate_similarity(store, idx, idx)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_self_similarity_is_maximal_for_several_products(store):
    rng = np.random.default_rng(0)
    for idx in rng.integers(0, len(store.df), size=20):
        score = calculate_similarity(store, int(idx), int(idx))
        assert score == pytest.approx(1.0, abs=1e-6), f"idx={idx} self-similarity={score}"


# --- category block (added after spot-checking found cross-category matches) --

def test_category_block_excluded_when_either_side_missing(store):
    """Mirrors the brand/colour fallback tests: if either product has no
    labeled category (15% of rows don't), that block must be dropped, not
    scored as a mismatch."""
    no_category = next(i for i in range(len(store.df)) if not store.category[i])
    has_category = next(i for i in range(len(store.df)) if store.category[i])

    from app.similarity import _vectorized_base_scores
    from app.config import DEFAULT_WEIGHTS

    # If the category block were wrongly scored as a 0.0 mismatch instead
    # of excluded, this pair's score would be pulled down by a full
    # `weights["category"]` penalty it shouldn't take.
    scores = _vectorized_base_scores(store, no_category, DEFAULT_WEIGHTS)
    assert not math.isnan(scores[has_category])
    assert 0.0 <= scores[has_category] <= 1.0


def test_same_category_products_score_higher_than_different_category(store):
    """Direct regression test for the spot-check finding in README Section
    0b: two products sharing a specific, informative category should be
    MORE similar under calculate_similarity than two products that differ
    in category — this is the entire point of having added the block."""
    category_counts: dict = {}
    for c in store.category:
        if c:
            category_counts[c] = category_counts.get(c, 0) + 1
    # pick an uncommon-but-not-unique category so the pair is specific, not
    # the near-universal "ClothingAccessories" fallback
    common_category = max(
        (c for c, n in category_counts.items() if n >= 2),
        key=lambda c: category_counts[c] if category_counts[c] < 100 else 0,
    )
    same_cat_idxs = [i for i, c in enumerate(store.category) if c == common_category][:2]
    diff_cat_idx = next(
        i for i, c in enumerate(store.category) if c and c != common_category
    )

    sim_same_category = calculate_similarity(store, same_cat_idxs[0], same_cat_idxs[1])
    sim_diff_category = calculate_similarity(store, same_cat_idxs[0], diff_cat_idx)
    assert sim_same_category >= sim_diff_category


# --- the worst-case real row: missing almost everything ----------------------

def test_product_missing_brand_colour_price_weight_still_returns_results(store):
    """Found via build_feature_store() exploration: uniq_id
    410c62298852e68f34c35560f2311e5a has no brand, no colour tag, no
    derivable price, no parseable weight — only rating + text + an image
    URL survive. The fallback-weight-redistribution logic must not divide
    by zero or crash when almost every block is unavailable."""
    pid = "410c62298852e68f34c35560f2311e5a"
    idx = store.idx_of(pid)
    assert not store.brand[idx]
    assert not store.colour_sets[idx]
    assert math.isnan(store.df.iloc[idx]["price"])
    assert math.isnan(store.df.iloc[idx]["weight_g"])

    result = find_similar_products(pid, 5)
    assert len(result) == 5
    assert pid not in result


def test_calculate_similarity_between_two_fully_missing_products(store):
    """Two products that are BOTH missing brand and colour: those blocks
    must be excluded (available_weight calc), not scored as a false
    match — this is the exact bug class that was found and fixed in
    _parse_colour_set (NaN silently becoming a fake matching tag)."""
    missing_both = [
        i for i in range(len(store.df))
        if not store.brand[i] and not store.colour_sets[i]
    ][:2]
    assert len(missing_both) == 2, "expected at least 2 such rows in this dataset"

    score = calculate_similarity(store, missing_both[0], missing_both[1])
    assert 0.0 <= score <= 1.0
    assert not math.isnan(score)


# --- num_similar boundaries ---------------------------------------------------

def test_num_similar_equals_one(store):
    pid = store.uniq_ids[5]
    result = find_similar_products(pid, 1)
    assert len(result) == 1


def test_num_similar_equals_entire_catalog_minus_one(store):
    pid = store.uniq_ids[5]
    n = len(store.df) - 1
    result = find_similar_products(pid, n)
    assert len(result) == n
    assert len(set(result)) == n  # no duplicates
    assert pid not in result


def test_num_similar_far_exceeds_catalog_size_does_not_crash(store):
    pid = store.uniq_ids[5]
    result = find_similar_products(pid, len(store.df) * 10)
    assert len(result) == len(store.df) - 1  # capped, not padded/duplicated


# --- determinism ---------------------------------------------------------------

def test_find_similar_products_is_deterministic(store):
    pid = store.uniq_ids[100]
    r1 = find_similar_products(pid, 10)
    r2 = find_similar_products(pid, 10)
    assert r1 == r2  # same order every time, not just same set


# --- API layer edge cases -------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_num_similar_zero_rejected(client):
    r = client.get("/find_similar_products", params={"product_id": "x", "num_similar": 0})
    assert r.status_code == 422


def test_api_num_similar_negative_rejected(client):
    r = client.get("/find_similar_products", params={"product_id": "x", "num_similar": -1})
    assert r.status_code == 422


def test_api_num_similar_at_upper_bound_accepted(client):
    from app.similarity import get_store
    pid = get_store().uniq_ids[0]
    r = client.get("/find_similar_products", params={"product_id": pid, "num_similar": 100})
    assert r.status_code == 200
    assert len(r.json()) == 100


def test_api_num_similar_over_upper_bound_rejected(client):
    r = client.get("/find_similar_products", params={"product_id": "x", "num_similar": 101})
    assert r.status_code == 422


def test_api_empty_product_id_is_404_not_500(client):
    r = client.get("/find_similar_products", params={"product_id": "", "num_similar": 5})
    assert r.status_code == 404


def test_api_unicode_and_special_characters_in_product_id_do_not_crash(client):
    """Not a real product id — confirms the lookup fails cleanly (404) for
    adversarial-looking input rather than leaking a 500 or a stack trace,
    since product_id flows straight into a dict lookup with no eval/SQL
    involved."""
    for pid in ["'; DROP TABLE products; --", "日本語", "a" * 10000, "../../etc/passwd"]:
        r = client.get("/find_similar_products", params={"product_id": pid, "num_similar": 5})
        assert r.status_code == 404
        assert "detail" in r.json()


def test_api_missing_required_params_is_422(client):
    r = client.get("/find_similar_products", params={"product_id": "x"})
    assert r.status_code == 422
    r = client.get("/find_similar_products", params={"num_similar": 5})
    assert r.status_code == 422
