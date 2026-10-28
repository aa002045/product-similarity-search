"""
Optional multimodal extension: visual similarity via a pretrained CNN.

Design decisions:

1. ResNet18, not a bigger model. EfficientNet/ResNet50 would likely extract
   marginally richer features, but ResNet18's 512-dim penultimate-layer
   embedding is already far more than this problem needs (these are
   catalog photos — mostly a garment on a plain/white background, not
   natural scenes), and it is the cheapest common pretrained backbone to
   run on CPU inside a container with no GPU. Given the service runs in a
   generic Kubernetes pod (no GPU guaranteed per the assignment's Docker/K8s
   requirement), inference cost per image is a real constraint, not just a
   nicety.

2. Lazy + cached, NOT precomputed for all 30k products up front. Embedding
   every product image means downloading ~30k-60k images from Amazon's CDN
   and running CNN inference on all of them before the service can answer
   a single request — a huge, slow, bandwidth-heavy batch job for a signal
   that's only 5% of the final similarity weight (see config.py). Instead:
   embeddings are computed on first access and cached to disk keyed by a
   hash of the image URL. This directly implements the assignment's
   "caching strategies for frequently accessed products" ask — the popular
   products end up cached after their first lookup; cold long-tail products
   pay the cost once.

3. Explicit fallback. Network fetch or decode can fail (dead URL, timeout,
   corrupt image). `get_embedding` returns None rather than raising, and the
   scoring layer (similarity.py) treats a None embedding as "this component
   is unavailable for this product" and redistributes its weight across the
   remaining components rather than crashing or silently treating it as
   maximally dissimilar.

This module is optional at runtime: if `torch`/`torchvision` aren't
installed, `ImageEmbedder.available()` returns False and callers skip the
image block entirely without error.
"""
import hashlib
import io
import logging
import os
import urllib.request
from functools import lru_cache
from typing import Optional

import numpy as np

from app.config import IMAGE_EMBEDDING_CACHE_PATH

logger = logging.getLogger(__name__)

_EMBED_DIM = 512
_FETCH_TIMEOUT_S = 4
_USER_AGENT = "Mozilla/5.0 (product-similarity-assignment)"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


class ImageEmbedder:
    """Thin wrapper around a frozen pretrained ResNet18 feature extractor,
    with an on-disk embedding cache (npz, keyed by url hash)."""

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._cache: dict[str, np.ndarray] = {}
        self._load_cache()

    @staticmethod
    def available() -> bool:
        return _torch_available()

    def _load_cache(self):
        if os.path.exists(IMAGE_EMBEDDING_CACHE_PATH):
            try:
                data = np.load(IMAGE_EMBEDDING_CACHE_PATH, allow_pickle=False)
                self._cache = {k: data[k] for k in data.files}
                logger.info("Loaded %d cached image embeddings", len(self._cache))
            except Exception:
                logger.exception("Failed to load image embedding cache; starting fresh")
                self._cache = {}

    def _save_cache(self):
        if self._cache:
            np.savez_compressed(IMAGE_EMBEDDING_CACHE_PATH, **self._cache)

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()  # drop the classifier head, keep the 512-d pooled features
        model.eval()
        self._model = model
        self._preprocess = weights.transforms()

    @staticmethod
    def _url_key(url: str) -> str:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()

    def _fetch_image_bytes(self, url: str) -> Optional[bytes]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_S) as resp:
                return resp.read()
        except Exception as e:
            logger.warning("Image fetch failed for %s: %s", url, e)
            return None

    def get_embedding(self, url: Optional[str], persist: bool = True) -> Optional[np.ndarray]:
        if not url or not self.available():
            return None

        key = self._url_key(url)
        if key in self._cache:
            return self._cache[key]

        raw = self._fetch_image_bytes(url)
        if raw is None:
            return None

        try:
            import torch
            from PIL import Image

            img = Image.open(io.BytesIO(raw)).convert("RGB")
            self._ensure_model()
            tensor = self._preprocess(img).unsqueeze(0)
            with torch.no_grad():
                emb = self._model(tensor).squeeze(0).numpy().astype(np.float32)
        except Exception as e:
            logger.warning("Image embedding failed for %s: %s", url, e)
            return None

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm  # L2-normalize so cosine similarity == dot product

        self._cache[key] = emb
        if persist:
            self._save_cache()
        return emb


@lru_cache(maxsize=1)
def get_shared_embedder() -> ImageEmbedder:
    """Process-wide singleton so the model is loaded once and the on-disk
    cache is shared across requests within a worker."""
    return ImageEmbedder()
