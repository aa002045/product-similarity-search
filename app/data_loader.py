"""
Data loading.

Design decision: the assignment's sample stub uses `pd.read_json(..., lines=True)`.
That works, but pandas' JSON parser loads the *entire* 30k-line file into one
Python-level pass and then infers dtypes column-by-column over mixed-type
nested dicts (e.g. `product_details__k_v_pairs`), which is slower and more
memory-hungry than it needs to be for data we are going to re-type ourselves
anyway in preprocessing.py.

Trade-off taken here: parse line-by-line with the stdlib `json` module into a
list of dicts, then build the DataFrame once. For 30k rows this is a few
hundred milliseconds either way, so at *this* scale it barely matters — the
real reason to do it this way is that it degrades gracefully: a single
malformed line doesn't blow up the whole load (we just skip + count it), and
it avoids pandas guessing a dtype for a column that is sometimes a dict,
sometimes a string, sometimes missing (seen in this dataset:
`sales_rank_in_parent_category`, `product_details__k_v_pairs`, etc.).
"""
import json
import logging
from typing import Optional

import pandas as pd

from app.config import DATA_PATH

logger = logging.getLogger(__name__)


def load_raw_dataframe(path: Optional[str] = None) -> pd.DataFrame:
    path = path or DATA_PATH
    records = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        logger.warning("Skipped %d malformed lines while loading %s", skipped, path)

    df = pd.DataFrame.from_records(records)

    if "uniq_id" not in df.columns:
        raise ValueError("Expected a 'uniq_id' column — got: %s" % list(df.columns))

    # uniq_id is our product_id. Dropping exact duplicates defensively;
    # a scraped marketing dataset like this one can contain re-crawls of the
    # same product on different dates.
    before = len(df)
    df = df.drop_duplicates(subset="uniq_id", keep="last").reset_index(drop=True)
    if before != len(df):
        logger.info("Dropped %d duplicate uniq_id rows", before - len(df))

    return df
