"""
Request-level result cache.

Design decision: a simple bounded LRU over `(product_id, num_similar)`
rather than a TTL cache or an external store like Redis. The reasoning:

- Product catalog data here is refreshed by an offline batch job (see
  scripts/build_artifacts.py), not by live writes — so "freshness" isn't a
  per-request concern; it's a per-reindex concern. A fixed-size LRU that
  gets wiped once on reindex (`invalidate_all()`) is both correct and far
  simpler than reasoning about per-entry TTLs.
- A handful of popular products dominate real traffic (power-law access,
  same shape as the brand-frequency long tail we already saw in the data).
  An LRU of a few thousand entries will have a very high hit rate for the
  actual request distribution without needing to cache the whole catalog.
- No external cache (Redis/Memcached) because the Kubernetes deployment
  here (see k8s/) is a small, single-service demo — adding a stateful
  dependency would be real operational cost for no measured benefit at
  this scale. The trade-off: this cache is per-pod, not shared across
  replicas, so a cold pod after a rolling deploy starts with 0% hit rate
  and a multi-replica deployment has N independent caches. That's an
  acceptable trade-off here; it would NOT be once there are enough
  replicas/QPS that the redundant compute (or the cold-start latency
  spikes) actually shows up on a dashboard — that's the point at which
  this should move to a shared cache.
"""
from functools import lru_cache
from typing import List, Tuple


def make_cached_lookup(fn, maxsize: int = 4096):
    """Wrap a `(product_id, num_similar) -> List[str]` function with an
    LRU cache, returning (cached_fn, invalidate_all)."""

    @lru_cache(maxsize=maxsize)
    def cached(product_id: str, num_similar: int) -> Tuple[str, ...]:
        return tuple(fn(product_id, num_similar))

    def lookup(product_id: str, num_similar: int) -> List[str]:
        return list(cached(product_id, num_similar))

    return lookup, cached.cache_clear
