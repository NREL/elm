# -*- coding: utf-8 -*-
"""Test ELM Ordinance parsing utilities"""
from pathlib import Path

import pytest

from elm.ords.utilities.parsing import (
    llm_response_as_json,
    merge_overlapping_texts,
)


@pytest.mark.parametrize(
    "in_str,expected",
    [
        (' {"a": 1} ', {"a": 1}),
        ('```json\n{"a": True, "b": False}```', {"a": True, "b": False}),
        ('{"a": True', {}),
    ],
)
def test_sync_retry(in_str, expected):
    """Test the `llm_response_as_json` function"""

    assert llm_response_as_json(in_str) == expected


@pytest.mark.parametrize(
    "text_chunks,n,expected",
    [
        (
            [
                "Some text. Some overlap. More text. More text that "
                "shouldn't be touched. Some overlap.",
                "Some overlap. More text.",
                "Some non-overlapping text.",
            ],
            12,
            "Some text. Some overlap. More text. More text that "
            "shouldn't be touched. Some overlap. More text.\nSome "
            "non-overlapping text.",
        )
    ],
)
def test_merge_overlapping_texts(text_chunks, n, expected):
    """Test the `merge_overlapping_texts` function"""

    assert merge_overlapping_texts(text_chunks, n) == expected


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
