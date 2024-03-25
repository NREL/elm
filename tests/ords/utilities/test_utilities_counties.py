# -*- coding: utf-8 -*-
"""ELM Ordinance county utilities tests. """
from pathlib import Path

import pytest
import pandas as pd

from elm.ords.utilities.counties import (
    load_all_county_info,
    load_counties_from_fp,
    county_websites,
)
from elm.ords.utilities.exceptions import ELMOrdsValueError


def test_load_counties():
    """Test the `load_all_county_info` function."""

    county_info = load_all_county_info()
    assert not county_info.empty

    expected_cols = [
        "County",
        "State",
        "FIPS",
        "County Type",
        "Full Name",
        "Website",
    ]
    assert all(col in county_info for col in expected_cols)
    assert len(county_info) == len(county_info.groupby(["County", "State"]))

    # Spot checks:
    assert "Decatur" in set(county_info["County"])
    assert "Box Elder" in set(county_info["County"])
    assert "Colorado" in set(county_info["State"])
    assert "Rhode Island" in set(county_info["State"])


def test_county_websites():
    """Test the `county_websites` function"""

    websites = county_websites()
    assert len(websites) == len(load_all_county_info())
    assert isinstance(websites, dict)
    assert all(isinstance(key, tuple) for key in websites)
    assert all(len(key) == 2 for key in websites)

    # Spot checks:
    assert ("decatur", "indiana") in websites
    assert ("el paso", "colorado") in websites
    assert ("box elder", "utah") in websites


def test_load_counties_from_fp(tmp_path):
    """Test `load_counties_from_fp` function."""

    test_county_fp = tmp_path / "out.csv"
    input_counties = pd.DataFrame(
        {"County": ["decatur", "DNE County"], "State": ["INDIANA", "colorado"]}
    )
    input_counties.to_csv(test_county_fp)

    counties = load_counties_from_fp(test_county_fp)

    assert len(counties) == 1
    assert set(counties["County"]) == {"Decatur"}
    assert set(counties["State"]) == {"Indiana"}
    assert {type(val) for val in counties["FIPS"]} == {int}


def test_load_counties_from_fp_bad_input(tmp_path):
    """Test `load_counties_from_fp` function."""

    test_county_fp = tmp_path / "out.csv"
    pd.DataFrame().to_csv(test_county_fp)

    with pytest.raises(ELMOrdsValueError) as err:
        load_counties_from_fp(test_county_fp)

    expected_msg = (
        "The following required columns were not found in the county input:"
    )
    assert expected_msg in str(err)
    assert "County" in str(err)
    assert "State" in str(err)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
