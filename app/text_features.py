"""
Text similarity.

Design decision: default to TF-IDF, not a transformer embedding, and here's
the actual trade-off (the assignment explicitly allows either):

- TF-IDF is deterministic, has zero external model-download dependency, is
  ~instant to fit on 30k short product titles, and is easy for a reviewer to
  audit (you can literally point at which shared words drove a match).
  Product titles in this dataset are short, keyword-dense strings
  ("LA' Facon Cotton Kalamkari Handblock Saree Blouse Fabric...") — this is
  exactly the regime where lexical overlap (TF-IDF) competes well with
  semantic embeddings, because the "semantics" mostly ARE the keywords
  (fabric, garment type, pattern, gender).
- A sentence-transformer embedding (e.g. all-MiniLM-L6-v2) would generalize
  better to paraphrases ("tee" vs "t-shirt") and is the better choice if
  titles get noisier/longer or you need cross-lingual matching — but it
  costs a ~90MB model download, non-trivial CPU/GPU time over 30k rows, and
  a harder-to-explain similarity score for the write-up.

The code is written so swapping in a transformer later is a one-function
change (see `embed_with_sentence_transformer`, unused by default) — that's
the actual deliverable of "pluggable", not a config flag nobody exercises.
"""
from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import RANDOM_SEED


def fit_tfidf(texts, max_features: int = 20000) -> tuple:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),       # unigrams + bigrams: "cotton saree" > "cotton" alone
        min_df=2,                 # drop hapax terms — mostly scrape noise / typos
        stop_words="english",
        sublinear_tf=True,        # log-scaled term frequency, standard for short docs
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def reduce_dimensions(tfidf_matrix: sparse.spmatrix, n_components: int = 100):
    """TruncatedSVD, not PCA.

    PCA requires a dense, mean-centered matrix; TF-IDF matrices are sparse
    and centering would destroy that sparsity (30k x 20k dense floats is a
    needless multi-GB array). TruncatedSVD operates directly on the sparse
    matrix and is the standard pairing with TF-IDF (this is literally LSA —
    Latent Semantic Analysis — when applied to a term-document matrix).

    This reduction matters for Part 3 (ANN): FAISS indexes dense vectors, so
    the 20k-dim sparse TF-IDF space needs to be collapsed before it can be
    concatenated with the numeric block and indexed.
    """
    n_components = min(n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    reduced = svd.fit_transform(tfidf_matrix)
    return svd, reduced


def embed_with_sentence_transformer(texts, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """Optional upgrade path. Not called by default — see module docstring."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer(model_name)
    return model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
