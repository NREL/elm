# -*- coding: utf-8 -*-
"""ELM Ordinance county utilities tests. """
from pathlib import Path

import pytest

from elm.ords.utilities.counties import load_county_info, county_websites


def test_load_counties():
    """Test the `load_county_info` function."""

    county_info = load_county_info()
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
    assert len(websites) == len(load_county_info())
    assert type(websites) == dict
    assert all(type(key) == tuple for key in websites)
    assert all(len(key) == 2 for key in websites)

    # Spot checks:
    assert ("decatur", "indiana") in websites
    assert ("el paso", "colorado") in websites
    assert ("box elder", "utah") in websites


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
