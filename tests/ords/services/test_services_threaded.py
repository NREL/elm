# -*- coding: utf-8 -*-
"""Test ELM Ordinances TempFileCache Services"""
from pathlib import Path

import pytest

from elm.web.document import HTMLDocument
from elm.ords.services.threaded import TempFileCache, FileMover


@pytest.mark.asyncio
async def test_temp_file_cache_service():
    """Test base implementation of `TempFileCache` class"""

    doc = HTMLDocument(["test"])
    doc.metadata["source"] = "http://www.example.com/?=%20test"

    cache = TempFileCache()
    cache.acquire_resources()
    out_fp = await cache.process(doc, doc.text)
    assert out_fp.exists()
    assert out_fp.read_text().startswith("test")
    cache.release_resources()
    assert not out_fp.exists()


@pytest.mark.asyncio
async def test_file_move_service(tmp_path):
    """Test base implementation of `FileMover` class"""

    doc = HTMLDocument(["test"])
    doc.metadata["source"] = "http://www.example.com/?=%20test"

    cache = TempFileCache()
    cache.acquire_resources()
    out_fp = await cache.process(doc, doc.text)
    assert out_fp.exists()
    assert out_fp.read_text().startswith("test")
    doc.metadata["cache_fn"] = out_fp

    expected_moved_fp = tmp_path / out_fp.name
    assert not expected_moved_fp.exists()
    mover = FileMover(tmp_path)
    mover.acquire_resources()
    moved_fp = await mover.process(doc)
    assert expected_moved_fp == moved_fp
    assert not out_fp.exists()
    assert moved_fp.exists()
    assert moved_fp.read_text().startswith("test")

    cache.release_resources()
    mover.release_resources()
    assert moved_fp.exists()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
