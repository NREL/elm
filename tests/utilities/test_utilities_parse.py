# -*- coding: utf-8 -*-
"""Test ELM Ordinance retry utilities"""
from pathlib import Path

import pytest

from elm.utilities.parse import is_double_col, remove_blank_pages


def test_is_double_col():
    """Test the `is_double_col` heuristic function"""

    assert not is_double_col("Some Text")
    assert is_double_col("Some    Text")
    assert is_double_col(
        """
        Some double    here over
        column text    multiple lines.
        given          :)
        """
    )
    assert not is_double_col(
        """
        Some text  with odd   spacing
        and  multiple lines but  not
        double column!
        """
    )


def test_remove_blank_pages():
    """Test the `remove_blank_pages` function"""

    assert remove_blank_pages([]) == []
    assert remove_blank_pages([""]) == []
    assert remove_blank_pages(["Here", ""]) == ["Here"]
    assert remove_blank_pages(["We", "  "]) == ["We"]
    assert remove_blank_pages(["", " Go ", "  ", "Again"]) == [" Go ", "Again"]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
