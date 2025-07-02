# -*- coding: utf-8 -*-
"""Miscellaneous ELM Ordinance tests"""
from pathlib import Path

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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
