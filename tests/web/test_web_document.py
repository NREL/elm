# -*- coding: utf-8 -*-
"""ELM Web Document class tests"""
from pathlib import Path

import pytest
import pdftotext
import pandas as pd

from elm import TEST_DATA_DIR
from elm.web.document import PDFDocument, HTMLDocument


class TestSplitter:
    """Splitter class for testing purposes."""

    def split_text(self, text):
        """Split text on newlines."""
        return text.split("\n")


@pytest.mark.parametrize("doc_type", [PDFDocument, HTMLDocument])
def test_basic_document(doc_type):
    """Test basic properties of the `Document` class"""

    doc = doc_type([""])
    assert doc.text == ""
    assert doc.raw_pages == []
    assert doc.attrs == {}
    if doc_type is PDFDocument:
        assert doc.num_raw_pages_to_keep == 0
        assert doc._last_page_index == 0
    assert doc.WRITE_KWARGS
    assert doc.FILE_EXTENSION


def test_pdf_doc():
    """Test Document class for sample PDF file"""
    doc_path = Path(TEST_DATA_DIR) / "GPT-4.pdf"

    with open(doc_path, "rb") as fh:
        pdf = pdftotext.PDF(fh, physical=True)

    og_text = "\n".join(pdf)
    doc = PDFDocument(pdf)

    assert 0 < len(doc.text) < len(og_text)

    assert "9/13/23, 11:23 AM" in og_text
    assert "9/13/23, 11:23 AM" not in doc.text

    assert "\r\n" in og_text or "\n\n" in og_text
    assert "\r\n" not in doc.text and "\n\n" not in doc.text

    assert doc.num_raw_pages_to_keep == 7
    assert doc._last_page_index == -2

    # pylint: disable=unnecessary-comprehension
    all_pages = [page for page in pdf]
    expected_raw_pages = all_pages[:7] + all_pages[-2:]
    assert doc.raw_pages == expected_raw_pages

    doc = PDFDocument(pdf, percent_raw_pages_to_keep=1000, max_raw_pages=1000)
    assert doc.raw_pages == all_pages


def test_html_doc():
    """Test Document class for sample HTML file"""
    doc_path = Path(TEST_DATA_DIR) / "Whatcom.txt"

    with open(doc_path, "r", encoding="utf-8") as fh:
        og_text = fh.read()

    doc = HTMLDocument([og_text])

    assert 0 < len(doc.text) < len(og_text)
    for tag in ["<p class", "</td>", "<a href"]:
        assert tag in og_text
        assert tag not in doc.text

    expected_table_fp = Path(TEST_DATA_DIR) / "expected_whatcom_table.txt"
    with open(expected_table_fp, "r", encoding="utf-8") as fh:
        expected_table = fh.read()

    assert expected_table not in og_text
    assert expected_table in doc.text

    assert doc.raw_pages == [og_text]


def test_html_doc_with_splitter():
    """Test Document class for sample HTML file with a text splitter"""
    doc_path = Path(TEST_DATA_DIR) / "Whatcom.txt"

    with open(doc_path, "r", encoding="utf-8") as fh:
        og_text = fh.read()

    doc = HTMLDocument([og_text], text_splitter=TestSplitter())

    assert len(doc.raw_pages) == og_text.count("\n") + 1


def test_doc_repr():
    """Test document repr method"""

    a = PDFDocument([])
    expected_repr = "PDFDocument with 0 pages\nAttrs: None"
    assert repr(a) == expected_repr

    b = PDFDocument(["test", "another"])
    expected_repr = "PDFDocument with 2 pages\nAttrs: None"
    assert repr(b) == expected_repr

    c = PDFDocument(["a"] * 1543)
    c.attrs = {"A": "some text", "b": pd.DataFrame({"Test": ["a"] * 1953})}

    expected_repr = ("PDFDocument with 1,543 pages\nAttrs:\n  A:\tsome text\n"
                     "  b:\tDataFrame with 1,953 rows")
    assert repr(c) == expected_repr


@pytest.mark.parametrize("pages", ([], ["aaa"], ["\n\n", "\n", "\r3"]))
def test_doc_is_empty(pages):
    """Test that doc without any text returns ``True`` for `empty` property"""

    assert PDFDocument(pages).empty
    assert HTMLDocument(pages).empty


def test_html_string_is_empty_doc():
    """Test that doc with just html text is empty"""

    html_str = ('<!DOCTYPE html><html><head></head><body style="height: 100%; '
                'width: 100%; overflow: hidden; margin:0px; background-color: '
                'rgb(38, 38, 38);"><embed '
                'name="0AA1AAF79D1EC03DEFB5B45B5529FA56" '
                'style="position:absolute; left: 0; top: 0;" width="100%" '
                'height="100%" src="about:blank" type="application/pdf" '
                'internalid="0AA1AAF79D1EC03DEFB5B45B5529FA56"></body></html>')
    assert HTMLDocument([html_str]).empty


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
