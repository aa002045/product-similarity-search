"""
Part 2 — FastAPI microservice.

Design decisions beyond the assignment's sample stub:
  - Feature store is built ONCE at process startup (`lifespan`), not per
    request and not at import time — import-time loading would make even
    `pytest --collect-only` pay the full dataset-load + TF-IDF-fit cost,
    and per-request loading would make every call pay it.
  - `num_similar` and `product_id` are validated explicitly: unknown
    product_id -> 404, non-positive num_similar -> 422/400, anything else
    unexpected -> 500 with the exception logged server-side but not leaked
    to the client. The sample stub's `raise e` would return a raw Python
    traceback as the HTTP response body, which both leaks internals and
    gives the client the wrong status code (FastAPI would turn an
    unhandled exception into a 500 anyway, but silently, with no control
    over the error shape returned to callers).
  - `/healthz` and `/readyz` are added (not asked for explicitly, but
    required in practice for the Kubernetes liveness/readiness probes
    referenced in k8s/deployment.yaml — a pod with no readiness probe will
    get traffic routed to it before the feature store finishes loading).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from typing import List

from app.cache import make_cached_lookup
from app.feature_matrix import build_feature_store
from app.similarity import find_similar_products as _find_similar_products
from app.similarity import set_store

logger = logging.getLogger("uvicorn.error")

_state = {"ready": False, "cached_lookup": None, "invalidate": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading feature store...")
    store = build_feature_store()
    set_store(store)
    cached_lookup, invalidate = make_cached_lookup(_find_similar_products)
    _state["cached_lookup"] = cached_lookup
    _state["invalidate"] = invalidate
    _state["ready"] = True
    logger.info("Feature store ready: %d products", len(store.df))
    yield
    _state["ready"] = False


app = FastAPI(title="Product Similarity Service", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    """Liveness: process is up and serving HTTP at all."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness: feature store is loaded and the service can actually
    answer a /find_similar_products call."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="feature store not ready")
    return {"status": "ready"}


@app.get("/find_similar_products", response_model=List[str])
def get_similar_products(
    product_id: str = Query(..., description="uniq_id of the source product"),
    num_similar: int = Query(..., gt=0, le=100, description="number of similar products to return"),
):
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="service is starting up, try again shortly")

    try:
        return _state["cached_lookup"](product_id, num_similar)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown product_id: {product_id!r}")
    except Exception:
        logger.exception("find_similar_products failed for product_id=%s", product_id)
        raise HTTPException(status_code=500, detail="internal error computing similar products")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
