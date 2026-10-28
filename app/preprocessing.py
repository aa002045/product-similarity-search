"""
Cleaning + feature engineering.

This is the module with the most "reasoning" in it, so each function has a
short note on *why*, not just what. Every design decision below was made by
actually profiling the 30k-row dataset first (see the exploration notes in
README.md) rather than guessing from the field names in the assignment.

Key mismatches between the assignment's attribute list and the real schema,
and how they're resolved:

- "color"  -> dataset field is `colour`, and it is only populated for ~20%
  of rows (6029 / 30000), and it's MULTI-VALUED, e.g. "Black|White|Red".
  We keep it as a *set* of colour tokens and compare products with Jaccard
  overlap rather than treating it as a single categorical value.

- "price"  -> there is no raw list-price field. There IS `sales_price`
  (the price actually charged) and `discount_percentage` (e.g. "40%") for
  ~49% of rows. We derive an approximate original list price as
  sales_price / (1 - discount/100) when discount is known, and treat it as
  missing otherwise. This is an explicit assumption, not a measured value —
  called out here and in the README so a reviewer doesn't mistake it for
  ground truth.

- "weight" -> stored as free text ("200 g", "1.5 kg", "99.8 g") with
  "999999999" used as a sentinel for "unknown" in ~79% of rows. Parsed to
  grams; sentinel and unparseable text (a handful of rows have garbage —
  e.g. marketing copy — in the weight field from a scraping error) become
  NaN, not 999999999. Treating the sentinel as a real number would make the
  single most common "value" in the column also the most extreme one,
  which would silently dominate any distance computation.

- "image"  -> dataset gives pipe-separated URL lists in `image_urls__small/
  medium/large`. We take the first URL from the largest available size
  (large > medium > small) as "the" product image for embedding.

- brand    -> 6458 distinct values, long-tailed (top brand "Max" has 504
  items, most brands have 1). This is too high-cardinality to one-hot
  encode into a dense vector without either exploding dimensionality or
  losing the tail. We don't embed brand into the numeric block at all —
  instead brand similarity is computed directly as an exact-match
  indicator (same brand == strong signal, different brand == 0), combined
  at the scoring stage (see similarity.py). That keeps a sparse, specific
  signal from being diluted by averaging over a huge one-hot space.

- category -> NOT in the assignment's attribute list, added after manually
  spot-checking real `find_similar_products` output and finding cross-
  category matches (a men's polo shirt matched against a bow tie and a
  women's swim skirt; a baby girl's top matched against boys' socks and
  men's trousers) — driven by price/text happening to align with nothing
  stopping the category mismatch. `parent___child_category__all` is a
  dict like {"ClothingAccessories": "#19,259", "WomensKurtasKurtis":
  "#1793"} — present on 85% of rows, 253 distinct child-category values
  (far lower cardinality than brand, and far more decisive for "is this
  even the same kind of product"). We extract the non-generic key
  ("ClothingAccessories" is the near-universal parent and carries no
  discriminative signal — present on 98% of rows that have the dict at
  all) and score it as an exact-match block, same mechanics as brand.
"""
import re
from typing import Optional, Set

import numpy as np
import pandas as pd

from app.config import NUMERIC_COLUMNS

WEIGHT_SENTINEL = "999999999"
_WEIGHT_RE = re.compile(r"^\s*([\d.]+)\s*(kg|kilograms|g|grams)\b", re.IGNORECASE)
_KG_UNITS = {"kg", "kilograms"}


def _parse_weight_grams(raw: Optional[str]) -> float:
    if raw is None or raw == WEIGHT_SENTINEL:
        return np.nan
    m = _WEIGHT_RE.match(str(raw))
    if not m:
        return np.nan  # e.g. scraped marketing copy that leaked into this field
    value, unit = float(m.group(1)), m.group(2).lower()
    return value * 1000.0 if unit in _KG_UNITS else value


def _parse_money(raw) -> float:
    if raw is None or raw == "":
        return np.nan
    try:
        return float(str(raw).replace(",", "").strip())
    except ValueError:
        return np.nan


def _parse_percent(raw) -> float:
    if raw is None or raw == "":
        return np.nan
    try:
        frac = float(str(raw).strip().rstrip("%")) / 100.0
    except ValueError:
        return np.nan
    # A discount of 100% (or a negative/garbled value, e.g. a scrape error
    # yielding "150%") makes `price = sales_price / (1 - frac)` divide by
    # zero or negative, producing inf/a negative "price". That single bad
    # row would then poison the shared StandardScaler fit across the WHOLE
    # numeric block in feature_matrix.py — one row's bad data corrupting
    # every product's similarity score, not just its own. Not observed in
    # the current dataset (max real value is 99%) but not something to
    # leave latent: treat any discount outside (0, 100) as unusable data,
    # same as any other missing value.
    if not (0.0 <= frac < 1.0):
        return np.nan
    return frac


def _parse_colour_set(raw) -> Set[str]:
    # `not raw` alone is NOT sufficient here: pandas represents a missing
    # key (the `colour` field is absent from ~80% of the raw JSON records,
    # not merely empty) as float('nan') when building the DataFrame, and
    # `bool(float('nan'))` is True in Python — `not nan` is False, not
    # True. Without this explicit check, every missing colour silently
    # became the literal string "nan" -> the one-element set {"nan"},
    # which made every pair of colour-less products register a *perfect*
    # Jaccard match instead of correctly excluding the colour block
    # entirely. That's not a cosmetic bug: it inflated colour_sim for the
    # majority of the catalog (colour is missing on ~80% of rows).
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or raw == "":
        return set()
    return {tok.strip().lower() for tok in str(raw).split("|") if tok.strip()}


def _first_image_url(row) -> Optional[str]:
    # The source JSON's own key naming is inconsistent — the small tier is
    # named "image_urls__small", but medium/large are just "medium"/"large"
    # (verified against the real file, not assumed). Listed largest-first
    # since a bigger image is the better input for the CNN embedding.
    for col in ("large", "medium", "image_urls__small"):
        val = row.get(col)
        # Same `float('nan')` truthiness trap as colour (see
        # _parse_colour_set): these columns are also absent-key-as-NaN for
        # ~1% of rows. `if val:` alone would pass a NaN float through and
        # stringify it into the literal "nan", which get_embedding() would
        # then try (and fail) to fetch as a real URL on every cache-miss
        # lookup — harmless (the fallback catches the failure) but
        # wasteful and semantically wrong: these products DO have no
        # image, and should report that as such from the start.
        if isinstance(val, float) and np.isnan(val):
            continue
        if val:
            first = str(val).split("|")[0].strip()
            if first:
                return first
    return None


_GENERIC_CATEGORY_KEY = "clothingaccessories"  # present on ~98% of rows that have the dict at all — not discriminative


def _extract_category(raw) -> Optional[str]:
    """`parent___child_category__all` is a dict of {category_name: rank},
    e.g. {"ClothingAccessories": "#19,259", "WomensKurtasKurtis": "#1793"}.
    We want the specific (child) category, not the near-universal parent.

    isinstance(raw, dict) is the correct missing-value check here (not
    `not raw`/truthiness) — pandas stores an absent dict-valued column as
    float('nan') for that row, same representation that caused the colour
    and image-url bugs above; a dict column needs the same explicit
    type check rather than relying on truthiness.
    """
    if not isinstance(raw, dict) or not raw:
        return None
    specific = [k for k in raw if k.strip().lower() != _GENERIC_CATEGORY_KEY]
    chosen = specific[0] if specific else next(iter(raw))
    return chosen.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with typed, analysis-ready columns added.

    Original raw columns are kept (prefixed implicitly by just leaving them
    as-is); new derived columns use clean, final names so downstream code
    never has to re-parse strings.
    """
    out = df.copy()

    out["sales_price"] = out.get("sales_price").map(_parse_money) if "sales_price" in out else np.nan
    out["discount_frac"] = out.get("discount_percentage").map(_parse_percent) if "discount_percentage" in out else np.nan

    # Derived original/list price. NaN when we have no discount to back it out —
    # we deliberately do NOT fall back to sales_price here, because that would
    # make "price" and "sales_price" identical for 51% of rows and silently
    # double-count the same signal in the numeric similarity block.
    out["price"] = out["sales_price"] / (1.0 - out["discount_frac"])
    out.loc[out["discount_frac"].isna(), "price"] = np.nan

    out["weight_g"] = out.get("weight").map(_parse_weight_grams) if "weight" in out else np.nan

    out["rating"] = pd.to_numeric(out.get("rating"), errors="coerce") if "rating" in out else np.nan

    out["brand_clean"] = out.get("brand").fillna("").str.strip()
    out["brand_clean"] = out["brand_clean"].replace("", np.nan)

    out["category_clean"] = (
        out.get("parent___child_category__all").map(_extract_category)
        if "parent___child_category__all" in out else None
    )

    out["colour_set"] = out.get("colour").map(_parse_colour_set) if "colour" in out else [set()] * len(out)

    out["image_url"] = out.apply(_first_image_url, axis=1)
    out["has_image"] = out["image_url"].notna()

    out["text_blob"] = (
        out.get("product_name", "").fillna("") + " " +
        out.get("meta_keywords", "").fillna("") + " " +
        out.get("brand_clean", "").fillna("")
    ).str.strip()

    return out


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Quick diagnostic used during development / in the README — not used
    at request-serving time."""
    cols = NUMERIC_COLUMNS + ["brand_clean", "colour_set", "image_url"]
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        if c == "colour_set":
            missing = (df[c].map(len) == 0).sum()
        else:
            missing = df[c].isna().sum()
        rows.append({"column": c, "missing": int(missing), "missing_pct": round(100 * missing / len(df), 1)})
    return pd.DataFrame(rows)
