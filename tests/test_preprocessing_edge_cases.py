"""
Unit-level edge cases for the parsing functions in app/preprocessing.py.

These run against synthetic inputs (not the full dataset) specifically to
exercise boundary/malformed values that may be rare or absent in the
current 30k-row file but are real possibilities in the field format:
NaN-as-missing-key, 100%+ discounts, unit variations, garbled text that
leaked into a numeric field from a scraping error, etc.

Two real bugs were found and fixed by writing exactly this kind of test
(see git history / comments in preprocessing.py):
  1. `_parse_colour_set` used `if not raw`, which does NOT catch
     `float('nan')` (nan is truthy in Python) — missing colours were
     silently becoming the fake tag {"nan"} instead of an empty set,
     which made every pair of colour-less products (~80% of the catalog)
     register a false perfect Jaccard match.
  2. `_first_image_url` had the same NaN-truthiness gap for the
     large/medium/small URL columns.
"""
import sys

sys.path.insert(0, ".")

import math

import numpy as np
import pytest

from app.preprocessing import (
    _extract_category,
    _first_image_url,
    _parse_colour_set,
    _parse_money,
    _parse_percent,
    _parse_weight_grams,
)


# --- weight ---------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("999999999", None),        # the sentinel IS the missing-value marker
    ("200 g", 200.0),
    ("200g", 200.0),             # no space between number and unit
    ("1.5 kg", 1500.0),
    ("1.5kilograms", 1500.0),
    ("99.8 g", 99.8),
    ("0 g", 0.0),                # zero is a valid (if odd) weight, not missing
    (None, None),
    ("", None),
    ("The Slim N Lift slimming shirt even smoothest out your back", None),  # real scrape garbage seen in this dataset
    ("-50 g", None),             # malformed/negative — regex requires [\d.] only, no sign
])
def test_parse_weight_grams(raw, expected):
    result = _parse_weight_grams(raw)
    if expected is None:
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected)


# --- money ------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("200.00", 200.0),
    ("1,999.00", 1999.0),       # thousands separator
    (None, None),
    ("", None),
    (float("nan"), None),       # pandas' absent-key representation
    ("not a number", None),
])
def test_parse_money(raw, expected):
    result = _parse_money(raw)
    if expected is None:
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected)


# --- discount percent --------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("40%", 0.40),
    ("0%", 0.0),                 # boundary: a real (if unusual) 0% discount
    ("99%", 0.99),
    ("100%", None),              # boundary that would divide-by-zero in price = sales_price/(1-frac)
    ("150%", None),              # garbled/invalid, must not silently become -2.0 or similar
    ("-10%", None),              # negative discount is nonsensical, treat as unusable
    (None, None),
    ("", None),
    (float("nan"), None),
])
def test_parse_percent_guards_against_divide_by_zero(raw, expected):
    result = _parse_percent(raw)
    if expected is None:
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected)


def test_price_derivation_never_produces_inf_or_negative():
    """Direct regression test for the found bug: price = sales_price / (1
    - discount_frac) must never see discount_frac >= 1.0 reach it."""
    sales_price = 100.0
    for discount_raw in ["100%", "150%", "-5%", None, ""]:
        frac = _parse_percent(discount_raw)
        price = sales_price / (1.0 - frac) if not math.isnan(frac) else float("nan")
        assert math.isnan(price), f"discount={discount_raw!r} produced price={price}, expected NaN"

    # And the boundary that SHOULD work:
    frac = _parse_percent("50%")
    price = sales_price / (1.0 - frac)
    assert price == pytest.approx(200.0)


# --- colour -------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Black|White", {"black", "white"}),
    ("Black | White ", {"black", "white"}),   # stray whitespace around separator
    ("Black", {"black"}),
    ("", set()),
    (None, set()),
    (float("nan"), set()),       # the actual bug: pandas' absent-key representation
    ("Black|Black", {"black"}),  # duplicate tags collapse via set semantics
])
def test_parse_colour_set(raw, expected):
    assert _parse_colour_set(raw) == expected


def test_colour_nan_is_not_confused_with_a_real_tag():
    """float('nan') must NOT produce {'nan'} — regression test for the bug
    found while edge-case testing this module."""
    result = _parse_colour_set(float("nan"))
    assert result == set()
    assert "nan" not in result


# --- image URL selection -------------------------------------------------------

@pytest.mark.parametrize("row,expected", [
    ({"large": "http://a.com/1.jpg|http://a.com/2.jpg", "medium": "http://b.com/1.jpg"}, "http://a.com/1.jpg"),
    ({"large": float("nan"), "medium": "http://b.com/1.jpg"}, "http://b.com/1.jpg"),  # NaN large falls through
    ({"large": float("nan"), "medium": float("nan"), "image_urls__small": "http://c.com/1.jpg"}, "http://c.com/1.jpg"),
    ({"large": float("nan"), "medium": float("nan"), "image_urls__small": float("nan")}, None),  # all missing
    ({}, None),  # columns absent entirely from the row
])
def test_first_image_url(row, expected):
    assert _first_image_url(row) == expected


# --- category extraction -------------------------------------------------------
# Added after manually spot-checking real find_similar_products output and
# finding cross-category matches (see README.md Section 0b) — `category`
# isn't in the assignment's attribute list but is a real field in the data.

@pytest.mark.parametrize("raw,expected", [
    ({"ClothingAccessories": "#19,259", "WomensKurtasKurtis": "#1793"}, "WomensKurtasKurtis"),
    ({"WomensKurtasKurtis": "#1793", "ClothingAccessories": "#19,259"}, "WomensKurtasKurtis"),  # key order must not matter
    ({"ClothingAccessories": "#19,259"}, "ClothingAccessories"),  # only the generic key present -> use it, don't discard all info
    ({}, None),
    (None, None),
    (float("nan"), None),  # same NaN-as-missing-dict representation as colour/image columns
    ("not a dict", None),  # malformed/unexpected type must not raise
])
def test_extract_category(raw, expected):
    assert _extract_category(raw) == expected


def test_category_nan_does_not_raise_or_become_a_fake_category():
    """Direct regression-shaped test for the same bug class as colour: a
    missing category dict must not be mistaken for a real one."""
    result = _extract_category(float("nan"))
    assert result is None
