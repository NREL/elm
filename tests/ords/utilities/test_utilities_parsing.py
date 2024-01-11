# -*- coding: utf-8 -*-
"""Test ELM Ordinance parsing utilities"""
from pathlib import Path

import pytest

from elm.ords.utilities.parsing import llm_response_as_json


@pytest.mark.parametrize(
    "in_str,expected",
    [
        (' {"a": 1} ', {"a": 1}),
        ('```json\n{"a": True, "b": False}```', {"a": True, "b": False}),
        ('{"a": True', {}),
    ],
)
def test_sync_retry(in_str, expected):
    """Test the `llm_response_as_json` decorator"""

    assert llm_response_as_json(in_str) == expected


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
