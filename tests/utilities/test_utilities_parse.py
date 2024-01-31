# -*- coding: utf-8 -*-
"""Test ELM Ordinance retry utilities"""
from pathlib import Path

import pytest

from elm.utilities.parse import (
    is_double_col,
    remove_blank_pages,
    format_html_tables,
)

SAMPLE_TABLE_TEXT = """Some text.
<table>
    <tr>
        <th>Company</th>
        <th>Contact</th>
        <th>Country</th>
    </tr>
    <tr>
        <td>Alfreds Futterkiste</td>
        <td>Maria Anders</td>
        <td>Germany</td>
    </tr>
    <tr>
        <td>Centro comercial Moctezuma</td>
        <td>Francisco Chang</td>
        <td>Mexico</td>
    </tr>
</table>
"""
EXPECTED_TABLE_OUT = """Some text.
|    | Company                    | Contact         | Country   |
|---:|:---------------------------|:----------------|:----------|
|  0 | Alfreds Futterkiste        | Maria Anders    | Germany   |
|  1 | Centro comercial Moctezuma | Francisco Chang | Mexico    |
"""


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


def test_format_html_tables():
    assert format_html_tables("test") == "test"
    assert format_html_tables(SAMPLE_TABLE_TEXT) == EXPECTED_TABLE_OUT

    bad_table_text = SAMPLE_TABLE_TEXT + "\nBad table:\n<table></table>"
    assert format_html_tables(bad_table_text) == bad_table_text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
