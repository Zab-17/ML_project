"""Unit tests for the Zillow row rules, feature helpers, price bands and cluster prices.

Run: python3 -m pytest test_zillow.py
"""
# AI was used to help write this code, but the resulting code was reviewed and edited by a human.

import numpy as np
import pandas as pd

from band_models import band_edges, price_bands
from zillow_features import MILES_PER_DEGREE, as_yes_no, census_tracts, code_text, miles_from_centre
from zillow_models import cluster_prices
from zillow_rows import kept_homes

GOOD_HOME = {"taxvaluedollarcnt": 500_000, "assessmentyear": 2016, "propertylandusetypeid": 261,
             "calculatedfinishedsquarefeet": 1500, "bedroomcnt": 3, "bathroomcnt": 2}


def homes_with(changes):
    return pd.DataFrame([GOOD_HOME] + [{**GOOD_HOME, **change} for change in changes])


def test_each_bad_home_is_counted_once_by_its_first_failing_rule():
    raw = homes_with([{"taxvaluedollarcnt": np.nan}, {"assessmentyear": 2015}, {"taxvaluedollarcnt": 9_000},
                      {"propertylandusetypeid": 47}, {"calculatedfinishedsquarefeet": np.nan},
                      {"bedroomcnt": 0}, {"assessmentyear": 2015, "bedroomcnt": 0}])
    kept, dropped = kept_homes(raw)
    assert len(kept) == 1
    assert list(dropped.values())[1:] == [1, 2, 1, 1, 1, 1, 1]


def test_census_tract_drops_the_block_digits():
    tracts = census_tracts(pd.Series([60590524.262052, 61110076.011008, np.nan]))
    assert tracts.tolist() == ["6059052426", "6111007601", "not recorded"]


def test_code_text_removes_float_endings_and_names_blanks():
    assert code_text(pd.Series([12447.0, np.nan, 5.0])).tolist() == ["12447", "not recorded", "5"]


def test_one_degree_north_is_69_miles():
    rows = pd.DataFrame({"latitude": [34_000_000, 35_000_000], "longitude": [-118_000_000, -118_000_000]})
    miles = miles_from_centre(rows, fit=np.array([True, False]))
    assert miles["miles north"].tolist() == [0.0, MILES_PER_DEGREE]
    assert miles["miles east"].tolist() == [0.0, 0.0]


def test_two_valued_category_becomes_one_yes_no_column_for_the_rarer_value():
    labels = pd.Series(["no", "no", "yes", "no"])
    columns = as_yes_no({"pool": labels}, fit=np.array([True] * 4))
    assert columns.columns.tolist() == ["pool = yes"]
    assert columns["pool = yes"].tolist() == [0, 0, 1, 0]


def test_price_bands_split_training_homes_into_equal_fifths():
    values = np.arange(100, dtype=float)
    bands = price_bands(values, band_edges(values))
    assert np.bincount(bands)[1:].tolist() == [20, 20, 20, 20, 20]


def test_empty_cluster_gets_the_overall_median():
    prices = cluster_prices(np.array([0, 0, 2]), np.array([1.0, 3.0, 10.0]), cluster_count=3)
    assert prices.tolist() == [2.0, 3.0, 10.0]


def test_block_search_finds_the_same_neighbours_as_the_one_pass_search():
    import scipy.sparse as sp
    from slide_models import nearest_rows
    from zillow_neighbours import BlockSearch

    table = sp.random(3_000, 40, density=0.3, format="csr", random_state=0)
    train, query = table[:2_500], table[2_500:]
    for distance in ("euclidean", "cosine"):
        one_pass = nearest_rows(train, query, 10, distance)
        in_blocks = BlockSearch(training_block=700, query_batch=128).nearest_rows(train, query, 10, distance)
        assert (one_pass == in_blocks).mean() > 0.99, distance   # float rounding may swap near-equal neighbours
