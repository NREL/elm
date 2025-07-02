# -*- coding: utf-8 -*-
"""Miscellaneous ELM Ordinance tests"""
from pathlib import Path
from datetime import datetime

import pytest

from elm.ords.extraction.date import _parse_date


def test_parse_date_empty():
    """Test `_parse_date` for empty list"""
    assert _parse_date([]) == (None, None, None)


def test_parse_date_multiple_elements_day_biggest():
    """Test `_parse_date` with multiple elements where day is the biggest"""
    dates = [{"year": 2023, "month": 7, "day": 30},
             {"year": 2023, "month": 7, "day": 15}]
    assert _parse_date(dates) == (2023, 7, 30)


def test_parse_date_multiple_elements_month_biggest():
    """Test `_parse_date` with multiple elements where month is the biggest"""
    dates = [{"year": 2023, "month": 7, "day": 15},
             {"year": 2023, "month": 10, "day": 15}]
    assert _parse_date(dates) == (2023, 10, 15)


def test_parse_date_multiple_elements_year_biggest():
    """Test `_parse_date` with multiple elements where year is the biggest"""
    dates = [{"year": 2023, "month": 7, "day": 15},
             {"year": 2024, "month": 7, "day": 15}]
    assert _parse_date(dates) == (2024, 7, 15)


def test_parse_date_bad_element():
    """Test `_parse_date` with only one component"""
    dates = [{"year": 2023, "month": 10, "day": 15}]
    assert _parse_date(dates) == (2023, 10, 15)


def test_parse_date_one_bad_element():
    """Test `_parse_date` with only one component"""
    dates = [{"year": 2023, "month": 20, "day": 15}]
    assert _parse_date(dates) == (2023, None, 15)

    dates = [{"year": 1900, "month": 7, "day": 15}]
    assert _parse_date(dates) == (None, 7, 15)

    dates = [{"year": datetime.now().year + 1, "month": 7, "day": 15}]
    assert _parse_date(dates) == (None, 7, 15)

    dates = [{"year": 2023, "month": 7, "day": 32}]
    assert _parse_date(dates) == (2023, 7, None)


def test_parse_date_one_component():
    """Test `_parse_date` with only one component"""
    dates = [{"year": 2023}]
    assert _parse_date(dates) == (2023, None, None)


def test_parse_date_multiple_components_with_missing_elements():
    """Test `_parse_date` with multiple components with missing elements"""
    dates = [{"year": None, "month": 7, "day": 15},
             {"year": 2024, "month": None, "day": None}]
    assert _parse_date(dates) == (2024, None, None)

    dates = [{"year": 2024, "month": 7, "day": 15},
             {"year": 2024, "month": None, "day": 30}]
    assert _parse_date(dates) == (2024, 7, 15)

    dates = [{"year": 2024, "month": None, "day": 15},
             {"year": 2024, "month": None, "day": 30}]
    assert _parse_date(dates) == (2024, None, 30)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
