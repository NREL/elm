# -*- coding: utf-8 -*-
"""Test ELM Ordinances TempFileCache Services"""
import logging
import asyncio
from pathlib import Path

import pytest

from elm.web.document import HTMLDocument
from elm.ords.services.threaded import (TempFileCache, FileMover,
                                        ThreadedService)
from elm.ords.services.provider import RunningAsyncServices
from elm.ords.utilities.queued_logging import LocationFileLog, LogListener


logger = logging.getLogger("elm")


def _log_from_thread():
    """Call logger instance from a thread"""
    logger.debug("HELLO WORLD")


@pytest.mark.asyncio
async def test_temp_file_cache_service():
    """Test base implementation of `TempFileCache` class"""

    doc = HTMLDocument(["test"])
    doc.attrs["source"] = "http://www.example.com/?=%20test"

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
    doc.attrs["source"] = "http://www.example.com/?=%20test"

    cache = TempFileCache()
    cache.acquire_resources()
    out_fp = await cache.process(doc, doc.text)
    assert out_fp.exists()
    assert out_fp.read_text().startswith("test")
    doc.attrs["cache_fn"] = out_fp

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


@pytest.mark.asyncio
async def test_logging_within_service(tmp_path, assert_message_was_logged):
    """Test that logging within a threaded service doesn't crash the process"""

    class ThreadedLogging(ThreadedService):
        """Subclass for testing"""

        @property
        def can_process(self):
            return True

        async def process(self):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.pool, _log_from_thread)

    log_listener = LogListener(["elm"], level="DEBUG")
    services = [ThreadedLogging()]

    async with RunningAsyncServices(services), log_listener as ll:
        with LocationFileLog(
            ll, tmp_path, location="test_loc", level="DEBUG"
        ):
            await ThreadedLogging.call()

    assert_message_was_logged("HELLO WORLD")


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
