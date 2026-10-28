# Product Similarity Search

Given a product's `uniq_id`, return the `num_similar` most similar products
from the Amazon Fashion catalog (30,000 rows, Feb-Apr 2020 crawl).

## Quickstart

```bash
pip install -r requirements.txt
PRODUCT_DATA_PATH=/path/to/the.ldjson python -m pytest tests/ -v
PRODUCT_DATA_PATH=/path/to/the.ldjson uvicorn app.main:app --reload
curl "http://localhost:8000/find_similar_products?product_id=<uniq_id>&num_similar=5"
```

```bash
docker build -t product-similarity .
docker run -p 8000:8000 product-similarity
```
The dataset is not included in this repo. Place marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson in the data/ folder before running.

## 0. Testing, and two real bugs it found

68 tests across 4 files, all exercising the real dataset (not mocks —
see `tests/test_similarity.py`'s docstring for why):

- `tests/test_similarity.py` — basic correctness (count, exclusion of
  self, validity, determinism-adjacent checks)
- `tests/test_preprocessing_edge_cases.py` — unit-level boundary cases for
  every parser: sentinel weight values, unit variants (`"200g"` vs
  `"1.5 kg"`), malformed/garbled text, **a 100% (or >100%) discount**, NaN
  representations
- `tests/test_edge_cases.py` — integration-level: self-similarity must be
  the maximum score, a real product missing almost every attribute,
  `num_similar` at 1 / at `len(catalog)-1` / far beyond the catalog size,
  determinism of repeated calls, and the API layer's error handling
  (negative/zero/over-limit `num_similar`, empty/unicode/adversarial-
  looking `product_id`, missing params)
- `tests/test_cache_and_image_edge_cases.py` — the result cache actually
  caches *and* actually invalidates, cache entries aren't shared mutable
  state, and image fetch failures return `None` without being written
  into the embedding cache as if they were valid

Writing these surfaced two real, non-cosmetic bugs, both now fixed in
`app/preprocessing.py`:

1. **A divide-by-zero / negative-price landmine.** `price` was derived as
   `sales_price / (1 - discount_frac)`. A discount of exactly 100% (or a
   garbled value like `"150%"`) makes that `inf` or negative. The real
   dataset's max discount happens to be 99%, so this never fired here —
   but it's a *shared* `StandardScaler` fit across the whole numeric
   block, so one future bad row would have corrupted every product's
   numeric similarity, not just its own. Fixed: any discount outside
   `[0%, 100%)` is now treated as missing, same as any other unusable
   value. Regression test: `test_price_derivation_never_produces_inf_or_negative`.

2. **`float('nan')` is truthy in Python.** `_parse_colour_set` checked
   `if not raw: return set()` to detect a missing colour — but pandas
   represents an *absent* dict key (the `colour` field is missing from
   ~80% of the raw JSON records, not merely empty) as `float('nan')`
   when the DataFrame is built, and `bool(float('nan'))` is `True`. So
   `not raw` was `False` for every missing colour, and execution fell
   through to `str(raw).split("|")`, turning NaN into the literal string
   `"nan"` — treated as a real colour tag. The result: **every pair of
   the ~24,000 colour-less products (80% of the catalog) scored a false
   *perfect* Jaccard colour match** instead of correctly excluding the
   colour block entirely. This wasn't just a correctness bug — it was
   also a measured **~3.7x performance regression** (brute-force search
   dropped from ~9.1ms/query to ~2.5ms/query after the fix), because the
   per-query colour-overlap loop was doing real (bogus) work for 80% of
   the catalog instead of skipping it. The same NaN-truthiness gap
   existed in `_first_image_url` for the large/medium/small URL columns
   and was fixed the same way. Regression tests:
   `test_colour_nan_is_not_confused_with_a_real_tag`,
   `test_calculate_similarity_between_two_fully_missing_products`.

Run `pytest tests/ -v` to see all 68 pass.

## 0b. Does the output actually look right? (not just "does it run")

Passing tests proves the code doesn't crash and behaves correctly on the
cases it was told to check — it does not prove the *recommendations* are
good. So beyond the automated tests, a handful of real query products were
spot-checked by eye (brand, category, price shown alongside each result):

```
QUERY:  [Shiya] Shiya Tassar Silk Saree with Blouse Piece | price=1199.0
SIMILAR: 4 of 5 results are other Shiya sarees, same price point.    -- good
QUERY:  [Jerzees] Jerzees 537 Mens Easy Care Polo | price=3819.0
SIMILAR: a handmade wood bow tie, a women's swim skirt, a web belt.  -- bad
```

The second case exposed a real gap: at that point, nothing in the scoring
model represented "is this even the same KIND of product" — `brand`,
`colour`, `price/weight/rating`, and `text` (which only *implicitly*
captures category through keyword overlap) all failed to stop a men's
polo shirt from matching accessories, because their price and some text
tokens happened to align. `category` is not in the assignment's required
attribute list, but the dataset carries one: `parent___child_category__all`
is a dict like `{"ClothingAccessories": "#19,259", "WomensKurtasKurtis":
"#1793"}`, present on 85% of rows, with 253 distinct child-category values
(far lower cardinality than brand, far more decisive for this failure
mode). Added as a 6th scoring block (`app/preprocessing.py`'s
`_extract_category`, `app/similarity.py`), exact-match like brand, and
given the **single largest weight** (0.27) — deliberately higher than
brand, because the wrong kind of product is a more fundamental mismatch
than the right kind from the wrong brand.

Re-running the same spot-check after adding it: the Shiya saree case now
visibly reinforces through `category=WomensSarees` on every match. The
Jerzees polo case is **still weak** — and that's an honest, diagnosed
limitation, not a leftover bug: that specific product's raw record has no
`parent___child_category__all` value at all (category coverage is 85%,
not 100%), so the category block is correctly excluded as unavailable for
that query (same fallback-weight-redistribution mechanism as every other
missing attribute), and the result falls back to exactly the same
text/price-driven behavior as before. The fix only helps where the
underlying data has the signal — which is most of the catalog, but not
all of it, and that boundary is worth stating rather than glossing over.

**This change cost something, measurably**: adding `category` as the
heaviest-weighted block, while deliberately keeping it *out* of the Part-3
FAISS vector (same reasoning as brand/colour — see Section 3), made the
ANN candidate-generation gap bigger, not smaller: recall@10 of the ANN
path against the (now category-aware) exact ground truth dropped from
0.609 to **0.554**. That's exactly the trade-off Section 3 already
predicted in the abstract ("excluding a high-weight exact-match attribute
from the ANN vector space costs candidate coverage") — this is that
prediction actually landing, with a number, once the weighting changed.
It sharpens the Section 3 conclusion: a coarse category pre-filter before
the ANN search (not implemented here) would be the next thing to build if
this were going past a 30k-row prototype.

## 1. What the real data actually looks like (and why that matters)

The assignment's attribute list (`brand, color, price, image, sales_price,
weight, rating`) doesn't map 1:1 onto the dataset's real column names, and
several of those columns are noisy or sparse. This was profiled directly
(30,000 rows) before writing any similarity code, because designing against
assumed clean data and discovering the mismatch later is how you end up
re-deriving every decision below:

| Assignment term | Real field(s) | Notes |
|---|---|---|
| `brand` | `brand` | 73% populated, **6,458 distinct values**, long-tailed (top brand "Max" = 504 items, most brands have 1) |
| `color` | `colour` | only **20% populated** (6,029 / 30,000), and **multi-valued**: `"Black|White"`, `"Black|Blue|Red"` |
| `price` | *(no raw field)* | derived from `sales_price / (1 - discount_percentage)`; `discount_percentage` is only 49% populated, so this derived `price` is frequently unavailable too — treated as missing, never silently backfilled from `sales_price` |
| `sales_price` | `sales_price` | string like `"200.00"`, 90% populated |
| `weight` | `weight` | **79% of rows (23,745) use the sentinel `"999999999"`** for "unknown" — if parsed as a real number, the single most common "value" becomes the most extreme one and dominates any distance metric. Real values are free text: `"200 g"`, `"1.5 kg"`, a handful of literal garbage (marketing copy that leaked into the field from a scraping error) |
| `rating` | `rating` | 100% populated, range 1.0-5.0, no zero/default values — the cleanest numeric field in the dataset |
| `image` | `image_urls__small/medium/large` | pipe-separated lists of URLs per size tier; `large` missing for 4% of rows |

Full missingness report: `app.preprocessing.missing_value_report(df)`.

## 2. Design decisions and trade-offs

### Missing data: impute + flag, never drop, never silently zero-score

`price` is missing on the majority of rows and `weight` is missing on ~79%
of them. Dropping those rows would discard most of the catalog. Instead,
for each numeric column: add a binary `<col>_was_missing` flag, impute the
value with the column median, then standard-scale everything (flags
included). The flag matters because without it, two products that are
*both* missing `weight` look identical on that dimension purely because
they were imputed to the same median — overstating their similarity as an
artifact of the imputation strategy rather than a real signal. This is a
mitigation, not a complete fix (a linear distance metric only partially
captures what the flag is trying to say) — worth stating plainly rather
than overclaiming it.

Same principle at the scoring layer, for a different reason: if two
products are both missing a `colour` tag, or one is missing `brand`, that
block is **excluded from the score and its weight is redistributed
proportionally across the blocks that ARE available** for that pair — not
scored as 0 (penalizes missingness) or 1 (rewards it). This is the
assignment's "fallback mechanism when certain features are unavailable."

### Why brand and colour are *not* in the same vector as price/weight/rating

Brand has 6,458 distinct values. One-hot encoding that into the same dense
vector as 4 numeric columns would mean ~6,458 of ~6,462 dimensions are
brand, and Euclidean/cosine similarity over that vector would be almost
entirely a (diluted, blurry) brand comparison — defeating the point of
having a numeric block at all. Instead, brand similarity is computed
directly as an **exact-match indicator**, and colour (multi-valued) as
**Jaccard overlap of tag sets** — both combined into the final score as
their own weighted blocks (see `app/similarity.py`). This keeps a sparse,
specific signal ("same brand") from being averaged away, and correctly
handles `colour` being a *set* rather than a single label (a product
tagged `"Black|White"` should partially match one tagged just `"Black"` —
equality would miss that entirely).

### Why Euclidean for numeric, cosine for text

Numeric attributes (price, weight, rating) are magnitude-meaningful
physical quantities — two products both being "close to zero" in
standardized price isn't inherently a similarity signal the way it would
be for, say, a direction-only embedding. Euclidean distance on standardized
features respects that; cosine similarity would conflate "cheap and cheap"
with "similar," which isn't the intended semantics. TF-IDF text vectors are
the opposite case: document length varies and cosine similarity's
normalization is exactly the standard fix for that — it's why cosine is
the default for any bag-of-words/TF-IDF representation.

### Text similarity: TF-IDF, not a transformer embedding, by default

The dataset's "text" is short, keyword-dense product titles (`"LA' Facon
Cotton Kalamkari Handblock Saree Blouse Fabric 100 cms Black Base Dancers
(Cotton)"`) — the kind of text where lexical overlap competes well with
semantic embeddings, because the semantics mostly *are* the keywords
(fabric, garment type, pattern). TF-IDF + bigrams, reduced via TruncatedSVD
(not PCA — PCA requires dense, mean-centered input, which would destroy the
sparsity of a 30k × 20k TF-IDF matrix for no benefit; TruncatedSVD operates
directly on the sparse matrix and is the standard LSA pairing with TF-IDF),
is deterministic, has zero model-download dependency, and fits on the
whole catalog in under a second. A sentence-transformer embedding would
generalize better to paraphrases ("tee" vs. "t-shirt") at the cost of a
~90MB model download and non-trivial CPU time per row — `app/text_features.py`
includes a working `embed_with_sentence_transformer` as a drop-in upgrade,
unused by default, so that trade-off is a one-line change, not a redesign.

### Multimodal (image) similarity: a re-ranking stage, not a 5th parallel block

Computing a CNN embedding for every product's image on every request would
mean a forward pass over the whole catalog per query — the dataset has
~30-60k image URLs total, and fetching + embedding all of them up front is
a slow, bandwidth-heavy batch job for a signal that's intentionally only
5% of the final weight (see `DEFAULT_WEIGHTS` in `app/config.py`). Instead:
`find_similar_products` always runs the cheap numeric+brand+colour+text
pass over the full catalog first (pure numpy, no network I/O), takes the
top `shortlist_multiplier × num_similar` candidates, and **only for those**
fetches/embeds images (ResNet18, chosen over a bigger backbone because
these are plain-background catalog photos, not natural scenes, and the
service has no guaranteed GPU) and re-blends the score. This is the same
"cheap candidate generation, expensive re-ranking" split used in real
recommendation systems. Image embeddings are cached to disk keyed by a
hash of the URL — first lookup of a popular product pays the download+CNN
cost once, every subsequent lookup (by anyone) is a cache hit. If a fetch
fails (dead URL, timeout — verified against this dataset's real URLs) the
component is dropped for that candidate and its weight redistributed, same
as any other missing attribute.

### Caching: three layers, three different invalidation triggers

1. **Per-request result cache** (`app/cache.py`) — bounded LRU over
   `(product_id, num_similar)`. Catalog data changes via an offline batch
   job, not live writes, so there's no per-entry freshness concern — the
   whole cache is just cleared on reindex. No external cache (Redis) is
   used: this is a small single-service deployment, and a per-pod LRU is
   the honest trade-off until replica count/QPS is high enough that the
   redundant compute across pods actually shows up on a dashboard.
2. **Per-image embedding cache** (`app/image_features.py`) — invalidated
   only if the source image URL changes (new hash).
3. **Catalog-level feature artifacts** (`scripts/build_artifacts.py`) —
   the full feature store + FAISS index, rebuilt whenever the dataset
   itself is refreshed.

### Tie-breaking

Equal scores are broken by: closer `sales_price` → higher `rating` →
`uniq_id` (for total determinism in tests). The assignment allows any
tie-breaker; these were picked because they're both already-loaded numeric
signals, not because they're uniquely correct.

## 3. Part 3 (bonus): approximate nearest neighbor search

**Is ANN actually needed at 30k rows? No — measured, not assumed:**

```
$ python scripts/build_artifacts.py && python scripts/benchmark.py
Dataset size:              30000 products
Queries:                   200, k=10
Brute-force total:         778.2 ms  (3.89 ms/query)
ANN (HNSW) total:          64.6 ms   (0.32 ms/query)
Recall@10 (ANN vs exact):  0.554  (at shortlist_multiplier=8)
```

Brute-force is ~3.9ms/query — already far below any reasonable latency
budget at this catalog size. ANN is implemented and benchmarked anyway
because the point of this section is to demonstrate the approach that
*becomes* necessary at real Amazon scale (10-100M+ products), where O(n)
scans per query stop being viable, and to show the actual break-even
reasoning with numbers rather than assert it.

**Why HNSW via FAISS, not Annoy:**
- **HNSW** — Malkov & Yashunin, 2016, *"Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World
  graphs"* ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)) —
  supports incremental `index.add()` with no retraining step, which matters
  because a real product catalog has continuous additions, unlike IVF-style
  indexes that need a k-means-style `train()` pass over representative
  data first.
- **FAISS** — Johnson, Douze & Jégou, 2017, *"Billion-scale similarity
  search with GPUs"* ([arXiv:1702.08734](https://arxiv.org/abs/1702.08734))
  — the most production-proven implementation of HNSW, with a clear upgrade
  path (IVF-PQ for memory-compressed search) if the catalog outgrows what
  HNSW's full-float graph can hold in RAM.
- **Annoy** (random-projection forest) is simpler and lower-memory per
  tree, but is read-only after building — every catalog update needs a
  full rebuild — and needs more candidate scans than HNSW at equal recall
  in published benchmarks. Given this service needs to keep serving while
  periodically reindexed, HNSW's incremental add was the deciding factor.

**Why recall@10 is only ~0.6, and what that number is actually telling you:**
The FAISS index only covers the `numeric` + `text` blocks (each pre-scaled
by `sqrt(weight)` before concatenation, so L2 distance in the combined
space approximates the weighted combination of those two blocks — a
standard trick for mixing feature blocks under one Euclidean index).
`brand` and `colour` are deliberately excluded from that vector (see
Section 2's reasoning) and are only applied during the exact re-rank of the
ANN shortlist. That means when a product's true top-10 match is driven
mostly by an exact brand match rather than numeric/text closeness, it can
be absent from the ANN candidate shortlist entirely — no amount of
re-ranking recovers a candidate that was never retrieved. This was
measured, not just reasoned about:

```
shortlist_multiplier=  8 -> recall@10 = 0.558
shortlist_multiplier= 20 -> recall@10 = 0.698
shortlist_multiplier= 50 -> recall@10 = 0.756
```

Recall climbs with a larger shortlist (confirming it's a candidate-coverage
gap, not an HNSW tuning issue — `efSearch` was also checked and isn't the
limiting factor at this scale), but never reaches 1.0, because the gap is
structural: some true top-10 matches simply aren't near neighbors in
numeric+text space at all. **The honest conclusion:** at this dataset's
actual scale, use the exact brute-force path (Part 1) — it's both faster
in absolute terms for 30k rows in this prototype (no graph-construction
overhead). The ANN path's value only materializes at a scale where
brute-force stops fitting the latency budget, at which point you'd also
want to fix this recall gap — e.g. a coarse brand-bucket pre-filter before
the ANN search, or indexing brand as a learned embedding instead of a
one-hot. That is exactly the kind of result a take-home should surface
honestly rather than paper over.

## 4. Part 2: the microservice

- `GET /find_similar_products?product_id=...&num_similar=...` → `List[str]`
- `GET /healthz` — liveness (process up)
- `GET /readyz` — readiness (feature store loaded) — used by the k8s
  readiness probe so a pod doesn't receive traffic before it can actually
  answer a request
- Errors: unknown `product_id` → `404`; `num_similar <= 0` or `> 100` →
  `422` (FastAPI's automatic query validation); anything unexpected → `500`
  with the exception logged server-side, not leaked to the client (the
  assignment's sample `raise e` would return a raw traceback as the
  response body)

## 5. Docker / Kubernetes

`Dockerfile` bakes the dataset into the image for a zero-setup `docker run`
— deliberately **not** how you'd do this in production (see the comments
in `Dockerfile`): a real catalog changes far more often than application
code and shouldn't ride along with every image push; mount a volume or
pull from object storage in an initContainer instead, and point
`PRODUCT_DATA_PATH` there (`k8s/deployment.yaml` already demonstrates the
override).

`k8s/deployment.yaml`: 2 replicas, readiness probe on `/readyz` (not
`/healthz` — using the liveness check for readiness would route traffic to
a pod that's up but still building its TF-IDF matrix), resource
requests/limits set because this service does real CPU work (TF-IDF fit +
SVD on startup, optional CNN inference).

`image_features.py`'s torch/torchvision dependency is commented out of
`requirements.txt` and **not installed in the Docker image** by default —
several hundred MB for a CPU-only torch wheel, for a feature that's 5% of
the final score and off unless `use_images=True` is explicitly requested.

## 6. What's deliberately out of scope

- No automatic brand canonicalization (e.g. "Nike" vs "NIKE INC") — the
  6,458-value brand column wasn't audited for near-duplicates; exact-match
  scoring means near-duplicate brand strings are silently treated as
  different.
- No learned re-ranking (the weights in `config.py` are hand-set, not fit
  against click/purchase data — this dataset has no interaction signal to
  fit against anyway).
- Image embeddings aren't precomputed for the full catalog — see Section 2
  for why that's a deliberate lazy-cache design, not an oversight.
- `category` (Section 0b) is only available for 85% of rows; the other
  15% get no benefit from it and can still produce cross-category matches
  driven purely by price/text, same as before it was added — not fixed,
  because the underlying data simply doesn't carry a category label for
  those rows, and inventing one would be fabricating a value, not cleaning
  one.
- No ground-truth evaluation set (no labeled "these two products ARE
  similar" pairs exist for this dataset) — quality was assessed by manual
  spot-checking real output (Section 0b), not a quantitative metric like
  precision@k against human judgments. That's a real limitation of this
  submission, not a hidden one: a proper evaluation would need either
  human-labeled similarity pairs or a proxy signal (e.g. co-purchase data,
  which this dataset doesn't include either) to score against.
