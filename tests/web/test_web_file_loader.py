# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""ELM Web file loading tests"""
import asyncio
import filecmp
from pathlib import Path
from functools import partial
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import pytest
import pdftotext

from elm import TEST_DATA_DIR
from elm.web.file_loader import AsyncFileLoader
from elm.web.document import PDFDocument, HTMLDocument
import elm.web.html_pw

GPT4_DOC_PATH = Path(TEST_DATA_DIR) / "GPT-4.pdf"
WHATCOM_DOC_PATH = Path(TEST_DATA_DIR) / "Whatcom.txt"


class MockResponse:
    """Mock web response for tests."""

    def __init__(self, read_return):
        """Store the desired read response."""
        self.read_return = read_return

    async def read(self):
        """Return what class was initialized with."""
        return self.read_return


@asynccontextmanager
async def patched_get(session, url, *args, **kwargs):
    """Patched implementation for get that reads from disk."""
    if url == "gpt-4":
        with open(GPT4_DOC_PATH, "rb") as fh:
            content = fh.read()
    elif url == "Whatcom":
        with open(WHATCOM_DOC_PATH, "rb") as fh:
            content = fh.read()

    yield MockResponse(content)


async def patched_get_html(url, *args, **kwargs):
    """Patched implementation for html get that reads from disk."""
    with open(WHATCOM_DOC_PATH, "r", encoding="utf-8") as fh:
        content = fh.read()

    return content


@pytest.mark.asyncio
async def test_async_file_loader_basic_pdf(monkeypatch):
    """Test `AsyncFileLoader` for a basic PDF doc"""

    monkeypatch.setattr(
        aiohttp.ClientSession,
        "get",
        patched_get,
        raising=True,
    )

    loader = AsyncFileLoader()
    doc = await loader.fetch(url="gpt-4")

    with open(GPT4_DOC_PATH, "rb") as fh:
        pdf = pdftotext.PDF(fh, physical=True)

    truth = PDFDocument(pdf)

    assert doc.text == truth.text
    assert doc.metadata["source"] == "gpt-4"
    assert "cache_fn" not in doc.metadata


@pytest.mark.asyncio
async def test_async_file_loader_basic_html(monkeypatch):
    """Test `AsyncFileLoader` for a basic HTML doc"""

    monkeypatch.setattr(
        aiohttp.ClientSession,
        "get",
        patched_get,
        raising=True,
    )
    monkeypatch.setattr(
        elm.web.html_pw,
        "_load_html",
        patched_get_html,
        raising=True,
    )

    loader = AsyncFileLoader()
    doc = await loader.fetch(url="Whatcom")

    with open(WHATCOM_DOC_PATH, "r", encoding="utf-8") as fh:
        content = fh.read()

    truth = HTMLDocument([content])

    assert doc.text == truth.text
    assert doc.metadata["source"] == "Whatcom"
    assert "cache_fn" not in doc.metadata


@pytest.mark.asyncio
async def test_async_file_loader_fetch_all(monkeypatch, tmp_path):
    """Test `AsyncFileLoader.fetch_all` function for basic docs"""

    monkeypatch.setattr(
        aiohttp.ClientSession,
        "get",
        patched_get,
        raising=True,
    )
    monkeypatch.setattr(
        elm.web.html_pw,
        "_load_html",
        patched_get_html,
        raising=True,
    )

    def _write_file(doc, content):
        out_fn = doc.metadata["source"]
        out_fp = tmp_path / f"{out_fn}.{doc.FILE_EXTENSION}"
        with open(out_fp, **doc.WRITE_KWARGS) as fh:
            fh.write(content)
        if doc.metadata["source"] == "gpt-4":
            return None
        return out_fp

    async def _cache_file(doc, content):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            pool, partial(_write_file, doc, content)
        )
        return result

    assert not list(tmp_path.glob("*"))

    with ThreadPoolExecutor() as pool:
        loader = AsyncFileLoader(file_cache_coroutine=_cache_file)
        docs = await loader.fetch_all("gpt-4", "Whatcom")

    assert len(docs) == 2

    with open(GPT4_DOC_PATH, "rb") as fh:
        pdf = pdftotext.PDF(fh, physical=True)

    truth_pdf = PDFDocument(pdf)

    with open(WHATCOM_DOC_PATH, "r", encoding="utf-8") as fh:
        content = fh.read()

    truth_html = HTMLDocument([content])

    assert docs[0].text == truth_pdf.text
    assert docs[0].metadata["source"] == "gpt-4"
    assert "cache_fn" not in docs[0].metadata

    assert docs[1].text == truth_html.text
    assert docs[1].metadata["source"] == "Whatcom"
    assert docs[1].metadata["cache_fn"].stem == "Whatcom"

    assert len(list(tmp_path.glob("*"))) == 2
    assert filecmp.cmp(GPT4_DOC_PATH, list(tmp_path.glob("*.pdf"))[0])

    html_file = list(tmp_path.glob("*.txt"))[0]
    assert not filecmp.cmp(WHATCOM_DOC_PATH, html_file)
    with open(html_file, "r", encoding="utf-8") as fh:
        assert truth_html.text == fh.read()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
