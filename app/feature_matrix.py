"""
Builds the in-memory "feature store" shared by both the exact (Part 1) and
approximate (Part 3) similarity engines, so there's exactly one place that
turns raw rows into vectors — the two engines must agree on what a product
"looks like" or their results would silently diverge.

Missing-value strategy for the numeric block (price, sales_price, weight_g,
rating):
    1. Add a binary `<col>_was_missing` indicator BEFORE imputing.
    2. Impute the actual value with the column median.
    3. StandardScale everything (indicators included) to zero mean / unit
       variance so no single raw-unit column (grams vs. rupees vs. a 1-5
       rating) dominates a Euclidean distance just because its native scale
       is bigger.
Why not drop rows with missing values instead? ~70-80% of rows are missing
`price` or `weight_g` (see README) — dropping them would throw away most of
the catalog. Why not impute and stop there (no indicator)? Because then two
products that are BOTH missing weight would look identical on that
dimension purely by construction, which overstates their similarity. The
indicator flags that "this closeness is partly an artifact of imputation"
to the (small) degree a linear model like Euclidean distance can represent
that — it's a mitigation, not a full fix, and that limitation is worth
being upfront about.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.config import NUMERIC_COLUMNS
from app.data_loader import load_raw_dataframe
from app.preprocessing import clean_dataframe
from app.text_features import fit_tfidf, reduce_dimensions


@dataclass
class FeatureStore:
    df: pd.DataFrame
    id_to_idx: Dict[str, int]
    uniq_ids: np.ndarray                # (n,) str array, same order as every other array here
    numeric_scaled: np.ndarray          # (n, d_numeric) standardized, imputed + missing-flags
    brand: np.ndarray                   # (n,) object array, "" for missing
    category: np.ndarray                # (n,) object array, "" for missing — see preprocessing.py
    colour_sets: List[Set[str]]
    text_reduced: np.ndarray            # (n, d_text) dense, SVD-reduced TF-IDF, L2-normalized
    image_urls: np.ndarray              # (n,) object array, None for missing
    sales_price: np.ndarray             # raw, used for tie-breaking
    rating: np.ndarray                  # raw, used for tie-breaking
    combined_ann_vectors: Optional[np.ndarray] = field(default=None)

    def idx_of(self, product_id: str) -> int:
        if product_id not in self.id_to_idx:
            raise KeyError(f"Unknown product_id: {product_id!r}")
        return self.id_to_idx[product_id]

    def product_id_at(self, idx: int) -> str:
        return self.uniq_ids[idx]


def _build_numeric_block(df: pd.DataFrame) -> np.ndarray:
    raw = df[NUMERIC_COLUMNS].astype(float)
    missing_flags = raw.isna().astype(float)
    missing_flags.columns = [f"{c}_was_missing" for c in NUMERIC_COLUMNS]

    imputed = raw.fillna(raw.median(numeric_only=True))

    combined = pd.concat([imputed, missing_flags], axis=1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined.values)
    return scaled


def build_feature_store(data_path: Optional[str] = None, text_svd_dims: int = 100) -> FeatureStore:
    raw_df = load_raw_dataframe(data_path)
    df = clean_dataframe(raw_df).reset_index(drop=True)

    id_to_idx = {pid: i for i, pid in enumerate(df["uniq_id"].tolist())}

    numeric_scaled = _build_numeric_block(df)

    brand = df["brand_clean"].fillna("").values.astype(object)
    category = df["category_clean"].fillna("").values.astype(object)
    colour_sets = df["colour_set"].tolist()

    _, tfidf_matrix = fit_tfidf(df["text_blob"].tolist())
    _, text_reduced = reduce_dimensions(tfidf_matrix, n_components=text_svd_dims)
    # L2-normalize rows so a dot product == cosine similarity downstream.
    norms = np.linalg.norm(text_reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    text_reduced = text_reduced / norms

    image_urls = df["image_url"].where(df["image_url"].notna(), None).values

    store = FeatureStore(
        df=df,
        id_to_idx=id_to_idx,
        uniq_ids=df["uniq_id"].values.astype(str),
        numeric_scaled=numeric_scaled,
        brand=brand,
        category=category,
        colour_sets=colour_sets,
        text_reduced=text_reduced,
        image_urls=image_urls,
        sales_price=df["sales_price"].values,
        rating=df["rating"].values,
    )
    return store
