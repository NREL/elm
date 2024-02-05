# -*- coding: utf-8 -*-
"""Test ELM Ordinances TempFileCache Services"""
import time
from pathlib import Path

import pytest

from elm.web.document import HTMLDocument
from elm.ords.services.temp_file_cache import TempFileCache


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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
