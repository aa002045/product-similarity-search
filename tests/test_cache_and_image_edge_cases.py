"""
Edge cases for the two things most likely to silently misbehave rather
than loudly crash: the result cache (a correctness bug here just serves
stale/wrong data, no exception) and image fetching (the whole point of
its fallback design is to swallow failures, so it needs a direct test to
confirm it's swallowing the RIGHT failures and not hiding a real bug).
"""
import sys

sys.path.insert(0, ".")

from app.cache import make_cached_lookup
from app.image_features import ImageEmbedder


# --- result cache ------------------------------------------------------------

def test_cached_lookup_returns_same_result_without_recomputing():
    calls = []

    def fn(product_id, num_similar):
        calls.append((product_id, num_similar))
        return [f"{product_id}-{i}" for i in range(num_similar)]

    cached, invalidate = make_cached_lookup(fn)

    r1 = cached("p1", 3)
    r2 = cached("p1", 3)
    assert r1 == r2 == ["p1-0", "p1-1", "p1-2"]
    assert len(calls) == 1  # second call was a cache hit, fn not called again


def test_cached_lookup_distinguishes_different_num_similar():
    calls = []

    def fn(product_id, num_similar):
        calls.append((product_id, num_similar))
        return [f"{product_id}-{i}" for i in range(num_similar)]

    cached, _ = make_cached_lookup(fn)
    cached("p1", 3)
    cached("p1", 5)  # different num_similar must NOT hit the (p1, 3) cache entry
    assert len(calls) == 2


def test_invalidate_all_forces_recompute():
    calls = []

    def fn(product_id, num_similar):
        calls.append(1)
        return ["x"]

    cached, invalidate = make_cached_lookup(fn)
    cached("p1", 1)
    cached("p1", 1)
    assert len(calls) == 1

    invalidate()
    cached("p1", 1)
    assert len(calls) == 2  # cache was actually cleared, not a no-op


def test_cache_returns_independent_lists_not_shared_mutable_state():
    """Each caller gets its own list — mutating a returned result must not
    corrupt what a later cache hit returns."""
    cached, _ = make_cached_lookup(lambda pid, n: ["a", "b"])
    r1 = cached("p", 2)
    r1.append("mutated")
    r2 = cached("p", 2)
    assert r2 == ["a", "b"], "mutating one caller's result leaked into the cache"


# --- image fetch fallback -----------------------------------------------------

def test_get_embedding_returns_none_for_none_url():
    embedder = ImageEmbedder()
    assert embedder.get_embedding(None) is None


def test_get_embedding_returns_none_for_malformed_url_without_raising():
    embedder = ImageEmbedder()
    for bad_url in ["not-a-url", "ftp://", "", "http://this-domain-should-not-exist-xyz123.invalid/img.jpg"]:
        result = embedder.get_embedding(bad_url)  # must not raise
        assert result is None


def test_get_embedding_caches_failures_do_not_poison_cache_as_success():
    """A failed fetch must return None and must NOT be written into the
    on-disk embedding cache as if it were a valid (all-zero or similar)
    embedding — otherwise a transient network blip permanently poisons
    that product's similarity scoring."""
    embedder = ImageEmbedder()
    bad_url = "http://this-domain-should-not-exist-xyz123.invalid/img.jpg"
    embedder.get_embedding(bad_url, persist=False)
    key = embedder._url_key(bad_url)
    assert key not in embedder._cache
