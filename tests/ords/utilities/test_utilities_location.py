# -*- coding: utf-8 -*-
"""ELM Ordinance Location utility tests. """
from pathlib import Path

import pytest

from elm.ords.utilities.location import County


def test_basic_county_properties():
    """Tets basic properties for ``County`` class"""

    county = County("Box Elder", "Utah")

    assert repr(county) == "County(Box Elder, Utah, is_parish=False)"
    assert county.full_name == "Box Elder County, Utah"
    assert county.full_name == str(county)

    assert county == County("Box elder", "uTah")
    assert county != County("Box Elder", "Utah", is_parish=True)

    assert county == "Box Elder County, Utah"
    assert county == "Box elder, Utah"


def test_basic_parish_properties():
    """Tets basic properties for ``County`` class with ``is_parish=True``"""

    county = County("Box Elder", "Utah", is_parish=True)

    assert repr(county) == "County(Box Elder, Utah, is_parish=True)"
    assert county.full_name == "Box Elder Parish, Utah"
    assert county.full_name == str(county)

    assert county == County("Box elder", "uTah", is_parish=True)
    assert county != County("Box Elder", "Utah", is_parish=False)

    assert county == "Box Elder Parish, Utah"
    assert county == "Box elder, Utah"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
