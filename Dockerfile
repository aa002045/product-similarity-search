# --- Design notes -----------------------------------------------------
# - python:3.11-slim, not full python:3.11: faiss-cpu + fastapi + pandas
#   don't need a full Debian toolchain at runtime, only at wheel-build
#   time, and nearly all of these ship prebuilt manylinux wheels — slim
#   keeps the final image smaller without giving anything up.
# - libgomp1 is installed explicitly: faiss's C++ core uses OpenMP, and
#   the slim base doesn't include libgomp by default — without it the
#   `import faiss` at container startup fails with a missing .so, which
#   is a confusing failure to debug from a k8s CrashLoopBackOff log vs.
#   catching it here at build time.
# - torch/torchvision (the optional image-similarity path) are NOT
#   installed in this image — they're commented out in requirements.txt.
#   That's a deliberate image-size trade-off: torch's CPU wheel alone is
#   several hundred MB, for a feature that's 5% of the final similarity
#   weight and off by default. Uncomment it in requirements.txt and
#   rebuild if you want /find_similar_products to use visual similarity.
# - The dataset is baked into the image (COPY data/) for this assignment
#   so `docker run` works with zero extra setup. In a real deployment
#   you would NOT do this — product catalogs change far more often than
#   application code, and a multi-GB-scale catalog shouldn't ride along
#   with every image push. The realistic version: mount a PVC/volume, or
#   pull the latest catalog snapshot from object storage (S3/GCS) in an
#   initContainer, and point PRODUCT_DATA_PATH at that mounted path
#   instead (see k8s/deployment.yaml for exactly this override).
# ------------------------------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

ENV PRODUCT_DATA_PATH=/app/data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson
ENV PRODUCT_CACHE_DIR=/app/cache
RUN mkdir -p /app/cache

EXPOSE 8000

# A plain, unauthenticated health endpoint is fine to curl from inside the
# container at build/runtime-check time; it's what the k8s probes also hit.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
