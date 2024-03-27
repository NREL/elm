# -*- coding: utf-8 -*-
"""ELM Web scraping utilities tests"""
from pathlib import Path

import pytest

from elm.web.document import HTMLDocument
from elm.web.utilities import (
    clean_search_query,
    compute_fn_from_url,
    write_url_doc_to_file,
)


@pytest.mark.parametrize(
    "query, expected_out",
    [
        ("", ""),
        ("1", "1"),
        (" a ", "a"),
        ('1.    "a test"', "a test"),
        (' 1. a test" ', '1. a test"'),
        (' 1. "a test ', "a test"),
        (' 1. "a test" and another ', 'a test" and another'),
        ("A normal query", "A normal query"),
    ],
)
def test_clean_search_query(query, expected_out):
    """Test cleaning google web search query"""

    assert clean_search_query(query) == expected_out


@pytest.mark.parametrize(
    "url, expected_out",
    [
        ("http://www.example.com/?=%20test", "examplecom20test"),
        ("https://www.example.com/?=%20test", "examplecom20test"),
        ("www.example.com/?=%20test", "examplecom20test"),
        ("example.com/?=%20test-again", "examplecom20testagain"),
        (
            "example.com/?=%20test" + "a" * 200,
            "examplecom20test"
            + "a" * 48
            + "e03013400c0bddd83e6d7d14ce28c3ec"
            + "d8afd02a789651f54d7f4273dc0528eb",
        ),
    ],
)
def test_compute_fn_from_url(url, expected_out):
    """Test computing filename from url"""

    out = compute_fn_from_url(url)
    assert out == expected_out
    assert len(out) <= 128


def test_compute_fn_from_url_make_unique():
    """Test computing filename from url and making it unique"""

    test_url = "http://www.example.com/?=%20test"
    test_1 = compute_fn_from_url(test_url)
    test_2 = compute_fn_from_url(test_url)
    test_3 = compute_fn_from_url(test_url, make_unique=True)
    test_4 = compute_fn_from_url(test_url, make_unique=True)

    assert test_1 == test_2
    assert test_1 != test_3
    assert test_1 != test_4
    assert test_2 != test_3
    assert test_2 != test_4
    assert test_3 != test_4

    for fn in [test_1, test_2, test_3, test_4]:
        assert "-" not in fn


def test_write_url_doc_to_file(tmp_path):
    """Test basic execution of `write_url_doc_to_file`"""

    doc = HTMLDocument(["test"])
    doc.metadata["source"] = "http://www.example.com/?=%20test"
    out_fp = write_url_doc_to_file(doc, doc.text, tmp_path)

    text_files = list(tmp_path.glob("*.txt"))
    assert len(text_files) == 1
    with open(text_files[0], "r") as fh:
        assert fh.read().startswith("test")

    assert out_fp.name == "examplecom20test.txt"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
