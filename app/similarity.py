"""
Part 1 — the required `find_similar_products` function, plus
`calculate_similarity`, matching the assignment's stub names.

Scoring model
-------------
Each candidate product gets a score in five independently-computed blocks,
then combined as a WEIGHTED AVERAGE OVER ONLY THE BLOCKS THAT ARE ACTUALLY
AVAILABLE for that pair (not a fixed weighted sum) — this is the "fallback
mechanism when certain features are unavailable" the assignment asks for.
If, say, neither product has a `colour` tag, that 8%-weight block is
dropped and its weight is redistributed proportionally across the blocks
that ARE available, rather than scoring it 0 (which would wrongly penalize
two products just for both lacking a tag) or 1 (which would wrongly reward
missingness).

Blocks:
  numeric  — price, sales_price, weight_g, rating (standardized, imputed).
             Euclidean distance -> similarity, because these are magnitude-
             meaningful physical quantities (two cheap products being
             "close to the origin" together is coincidental, not similar —
             cosine would conflate that; Euclidean on standardized features
             does not).
  category — exact match (1.0 / 0.0) on the dataset's child product
             category (e.g. "WomensKurtasKurtis", "MensT_Shirts" — see
             preprocessing.py). NOT one of the assignment's required
             attributes; added after manually spot-checking real
             `find_similar_products` output and finding cross-category
             matches (a men's polo shirt matched against a bow tie and a
             women's swim skirt) that none of the required attributes
             would have caught, since nothing before this forced "same
             kind of item" to matter. Gets the single largest weight
             deliberately: being the wrong KIND of product is a more
             fundamental mismatch than being the right kind from the
             wrong brand.
  brand    — exact match (1.0 / 0.0). Brand has 6458 distinct values; a
             one-hot/embedding representation would dilute this strong,
             sparse signal, so it's scored directly instead (see
             preprocessing.py docstring for the full reasoning).
  colour   — Jaccard overlap of colour-tag sets, because `colour` is
             multi-valued ("Black|White") — a single-label equality check
             would miss a product tagged "Black|White" matching one tagged
             just "Black".
  text     — cosine similarity of SVD-reduced TF-IDF vectors over
             product_name + meta_keywords + brand (see text_features.py).

Image similarity is layered on top as an optional re-ranking stage — see
`find_similar_products(..., use_images=True)` — rather than being a 5th
block computed for all 30k products on every request (see image_features.py
for why: it would mean a CNN forward pass over the whole catalog per
query).

Two-stage retrieval (candidate generation -> re-ranking)
----------------------------------------------------------
`find_similar_products` always does a fast, fully-vectorized pass over the
whole catalog using the numeric+category+brand+colour+text blocks (cheap:
dense numpy ops, no network I/O). When `use_images=True`, it takes the top
`shortlist_multiplier * num_similar` candidates from that pass and ONLY for
those does it fetch/compute image embeddings and re-blend them in. This
mirrors how production recommendation systems are actually structured
(cheap candidate generation, expensive re-ranking on a shortlist) and keeps
image inference off the hot path for every request.
"""
from functools import lru_cache
from typing import List, Optional

import numpy as np

from app.config import DEFAULT_WEIGHTS
from app.feature_matrix import FeatureStore, build_feature_store

# Kept at module scope to mirror the assignment's sample stub
# (`df = pd.read_json(...)` at import time) — loaded lazily via
# `get_store()` so importing this module doesn't immediately pay the
# dataset-load + TF-IDF-fit cost (useful for tests that only need a few
# rows, and for the FastAPI app which controls the loading moment itself).
_STORE: Optional[FeatureStore] = None


def get_store() -> FeatureStore:
    global _STORE
    if _STORE is None:
        _STORE = build_feature_store()
    return _STORE


def set_store(store: FeatureStore) -> None:
    """Test / service-startup hook to inject a prebuilt store instead of
    rebuilding it (e.g. load the cached artifacts written by
    scripts/build_artifacts.py)."""
    global _STORE
    _STORE = store


def _jaccard(a: set, b: set) -> Optional[float]:
    if not a or not b:
        return None
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else None


def calculate_similarity(store: FeatureStore, idx_a: int, idx_b: int,
                          weights: Optional[dict] = None) -> float:
    """Pairwise similarity between two products (by row index), in [0, 1].

    Exposed as a standalone function (per the assignment's stub) in
    addition to the vectorized path used internally by
    `find_similar_products` — this one is O(1) per call and is what you'd
    reach for in a test or a notebook; the vectorized version below is what
    actually powers the service, since looping this function 30,000 times
    in Python would be the main latency cost of the whole feature.
    """
    weights = weights or DEFAULT_WEIGHTS
    components, available_weight = {}, 0.0

    dist = float(np.linalg.norm(store.numeric_scaled[idx_a] - store.numeric_scaled[idx_b]))
    components["numeric"] = 1.0 / (1.0 + dist)
    available_weight += weights["numeric"]

    brand_a, brand_b = store.brand[idx_a], store.brand[idx_b]
    if brand_a and brand_b:
        components["brand"] = 1.0 if brand_a == brand_b else 0.0
        available_weight += weights["brand"]

    cat_a, cat_b = store.category[idx_a], store.category[idx_b]
    if cat_a and cat_b:
        components["category"] = 1.0 if cat_a == cat_b else 0.0
        available_weight += weights["category"]

    jac = _jaccard(store.colour_sets[idx_a], store.colour_sets[idx_b])
    if jac is not None:
        components["colour"] = jac
        available_weight += weights["colour"]

    cos = float(np.dot(store.text_reduced[idx_a], store.text_reduced[idx_b]))
    components["text"] = (cos + 1.0) / 2.0  # SVD cosine can be negative; map to [0,1]
    available_weight += weights["text"]

    if available_weight == 0:
        return 0.0
    return sum(weights[k] * v for k, v in components.items()) / available_weight


def _vectorized_base_scores(store: FeatureStore, idx: int, weights: dict) -> np.ndarray:
    n = len(store.df)

    dists = np.linalg.norm(store.numeric_scaled - store.numeric_scaled[idx], axis=1)
    numeric_sim = 1.0 / (1.0 + dists)

    brand_a = store.brand[idx]
    if brand_a:
        brand_match = (store.brand == brand_a).astype(float)
        brand_available = (store.brand != "") & (brand_a != "")
    else:
        brand_match = np.zeros(n)
        brand_available = np.zeros(n, dtype=bool)

    category_a = store.category[idx]
    if category_a:
        category_match = (store.category == category_a).astype(float)
        category_available = (store.category != "") & (category_a != "")
    else:
        category_match = np.zeros(n)
        category_available = np.zeros(n, dtype=bool)

    set_a = store.colour_sets[idx]
    colour_sim = np.zeros(n)
    colour_available = np.zeros(n, dtype=bool)
    if set_a:
        for i, s in enumerate(store.colour_sets):
            if s:
                union = len(set_a | s)
                colour_sim[i] = (len(set_a & s) / union) if union else 0.0
                colour_available[i] = True

    cos = store.text_reduced @ store.text_reduced[idx]
    text_sim = (cos + 1.0) / 2.0

    total_score = weights["numeric"] * numeric_sim
    total_weight = np.full(n, weights["numeric"])

    total_score += np.where(brand_available, weights["brand"] * brand_match, 0.0)
    total_weight += np.where(brand_available, weights["brand"], 0.0)

    total_score += np.where(category_available, weights["category"] * category_match, 0.0)
    total_weight += np.where(category_available, weights["category"], 0.0)

    total_score += np.where(colour_available, weights["colour"] * colour_sim, 0.0)
    total_weight += np.where(colour_available, weights["colour"], 0.0)

    total_score += weights["text"] * text_sim
    total_weight += weights["text"]

    return total_score / total_weight


def find_similar_products(product_id: str, num_similar: int,
                           use_images: bool = False,
                           shortlist_multiplier: int = 5) -> List[str]:
    """Required Part 1 entry point.

    Returns up to `num_similar` product IDs most similar to `product_id`,
    excluding the product itself, most-similar first.
    """
    store = get_store()
    idx = store.idx_of(product_id)
    weights = DEFAULT_WEIGHTS

    scores = _vectorized_base_scores(store, idx, weights)
    scores[idx] = -np.inf  # never recommend the product itself

    if use_images:
        from app.image_features import get_shared_embedder

        shortlist_size = min(len(store.df) - 1, max(num_similar * shortlist_multiplier, num_similar))
        shortlist = np.argpartition(-scores, shortlist_size)[:shortlist_size]

        embedder = get_shared_embedder()
        if embedder.available():
            query_emb = embedder.get_embedding(store.image_urls[idx])
            if query_emb is not None:
                img_w = weights["image"]
                rescale = 1.0 - img_w  # the base score already summed to 1.0 of non-image weight
                for i in shortlist:
                    cand_emb = embedder.get_embedding(store.image_urls[i])
                    if cand_emb is None:
                        continue  # fallback: leave this candidate's score as the non-image score
                    img_sim = float(np.dot(query_emb, cand_emb))  # both L2-normalized -> cosine
                    img_sim = (img_sim + 1.0) / 2.0
                    scores[i] = rescale * scores[i] + img_w * img_sim

    # Tie-break deterministically: higher score first, then closer
    # sales_price, then higher rating, then uniq_id for total determinism.
    #
    # Full tie-break is only applied to a bounded candidate buffer, not all
    # 30k rows: np.argpartition narrows to the top `buffer_size` by score in
    # O(n) first, and only that small buffer goes through the Python-level
    # sort key — sorting all 30k rows with a per-element Python lambda would
    # cost ~n log n Python-level calls for no benefit, since anything past
    # the buffer can never place in the top `num_similar` anyway.
    buffer_size = min(len(scores) - 1, max(num_similar * 10, 50))
    candidate_idx = np.argpartition(-scores, buffer_size)[:buffer_size]

    price_gap = np.abs(store.sales_price[candidate_idx] - store.sales_price[idx])
    price_gap = np.nan_to_num(price_gap, nan=np.inf)
    rating = np.nan_to_num(store.rating[candidate_idx], nan=-np.inf)
    cand_scores = scores[candidate_idx]

    order = sorted(
        range(len(candidate_idx)),
        key=lambda j: (-cand_scores[j], price_gap[j], -rating[j], store.uniq_ids[candidate_idx[j]]),
    )
    result_idx = [candidate_idx[j] for j in order if candidate_idx[j] != idx][:num_similar]
    return [store.product_id_at(i) for i in result_idx]
