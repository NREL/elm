# -*- coding: utf-8 -*-
"""ELM Web Document class tests"""
from pathlib import Path

import pytest
import pdftotext

from elm import TEST_DATA_DIR
from elm.web.document import PDFDocument, HTMLDocument


@pytest.mark.parametrize("doc_type", [PDFDocument, HTMLDocument])
def test_basic_document(doc_type):
    """Test basic properties of the `Document` class"""

    doc = doc_type([""])
    assert doc.text == ""
    assert doc.raw_pages == []
    assert doc.metadata == {}
    if doc_type is PDFDocument:
        assert doc.num_raw_pages_to_keep == 0
        assert doc._last_page_index == 0


def test_pdf_doc():
    """Test Document class for sample PDF file"""
    doc_path = Path(TEST_DATA_DIR) / "GPT-4.pdf"

    with open(doc_path, "rb") as fh:
        pdf = pdftotext.PDF(fh, physical=True)

    og_text = "\n".join(pdf)
    doc = PDFDocument(pdf)

    assert 0 < len(doc.text) < len(og_text)
    for phrase in ["\r\n", "9/13/23, 11:23 AM"]:
        assert phrase in og_text
        assert phrase not in doc.text

    assert doc.num_raw_pages_to_keep == 7
    assert doc._last_page_index == -2

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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
