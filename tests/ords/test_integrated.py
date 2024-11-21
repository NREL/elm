# -*- coding: utf-8 -*-
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=unused-argument
"""ELM Ordinance integration tests"""
import time
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import aiohttp
import httpx
import pytest
import openai

import elm.web.html_pw
from elm import TEST_DATA_DIR
from elm.ords.services.usage import TimeBoundedUsageTracker, UsageTracker
from elm.ords.services.openai import OpenAIService, usage_from_response
from elm.ords.services.threaded import TempFileCache
from elm.ords.services.provider import RunningAsyncServices
from elm.ords.utilities.queued_logging import LocationFileLog, LogListener
from elm.web.google_search import PlaywrightGoogleLinkSearch
from elm.web.file_loader import AsyncFileLoader
from elm.web.document import HTMLDocument


WHATCOM_DOC_PATH = Path(TEST_DATA_DIR) / "Whatcom.txt"


class MockResponse:
    def __init__(self, read_return):
        self.read_return = read_return

    async def read(self):
        return self.read_return


@asynccontextmanager
async def patched_get(session, url, *args, **kwargs):
    if url == "Whatcom":
        with open(WHATCOM_DOC_PATH, "rb") as fh:
            content = fh.read()

    yield MockResponse(content)


async def patched_get_html(url, *args, **kwargs):
    with open(WHATCOM_DOC_PATH, "r", encoding="utf-8") as fh:
        content = fh.read()

    return content


@pytest.mark.asyncio
async def test_openai_query(sample_openai_response, monkeypatch):
    """Test querying OpenAI while tracking limits and usage"""

    start_time = None
    elapsed_times = []

    async def _test_response(*args, **kwargs):
        time_elapsed = time.monotonic() - start_time
        elapsed_times.append(time_elapsed)
        if time_elapsed < 3:
            response = httpx.Response(404)
            response.request = httpx.Request(method="test", url="test")
            raise openai.RateLimitError(
                "for testing", response=response, body=None
            )

        if kwargs.get("bad_request"):
            response = httpx.Response(404)
            response.request = httpx.Request(method="test", url="test")
            raise openai.BadRequestError(
                "for testing", response=response, body=None
            )
        return sample_openai_response()

    client = openai.AsyncOpenAI(api_key="dummy")
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        _test_response,
        raising=True,
    )
    rate_tracker = TimeBoundedUsageTracker(max_seconds=5)
    openai_service = OpenAIService(
        client, rate_limit=3, rate_tracker=rate_tracker
    )

    usage_tracker = UsageTracker("my_county", usage_from_response)
    async with RunningAsyncServices([openai_service]):
        start_time = time.monotonic()
        message = await OpenAIService.call(
            usage_tracker=usage_tracker, model="gpt-4"
        )
        message2 = await OpenAIService.call(model="gpt-4")

        assert openai_service.rate_tracker.total == 13
        assert message == "test_response"
        assert message2 == "test_response"
        assert len(elapsed_times) == 3
        assert elapsed_times[0] < 1
        assert elapsed_times[1] >= 4
        assert elapsed_times[2] >= 9

        assert usage_tracker == {
            "default": {
                "requests": 1,
                "prompt_tokens": 100,
                "response_tokens": 10,
            }
        }

        time.sleep(6)
        assert openai_service.rate_tracker.total == 0

        start_time = time.monotonic() - 4
        await OpenAIService.call(model="gpt-4")
        await OpenAIService.call(model="gpt-4")
        assert len(elapsed_times) == 5
        assert elapsed_times[-2] - 4 < 1
        assert elapsed_times[-1] - 4 > 5

        time.sleep(6)
        start_time = time.monotonic() - 4
        assert openai_service.rate_tracker.total == 0
        message = await OpenAIService.call(
            usage_tracker=usage_tracker, model="gpt-4", bad_request=True
        )
        assert message is None
        assert openai_service.rate_tracker.total <= 3
        assert usage_tracker == {
            "default": {
                "requests": 1,
                "prompt_tokens": 100,
                "response_tokens": 10,
            }
        }


@pytest.mark.asyncio
async def test_google_search_with_logging(tmp_path):
    """Test searching google for some counties with logging"""

    assert not list(tmp_path.glob("*"))

    logger = logging.getLogger("search_test")
    test_locations = ["El Paso County, Colorado", "Decatur County, Indiana"]
    num_requested_links = 10

    async def search_single(location):
        logger.info("This location is %r", location)
        search_engine = PlaywrightGoogleLinkSearch()
        return await search_engine.results(
            f"Wind energy zoning ordinance {location}",
            num_results=num_requested_links,
        )

    async def search_location_with_logs(
        listener, log_dir, location, level="INFO"
    ):
        with LocationFileLog(
            listener, log_dir, location=location, level=level
        ):
            logger.info("A generic test log")
            return await search_single(location)

    log_dir = tmp_path / "logs"
    log_listener = LogListener(["search_test"], level="DEBUG")
    async with log_listener as ll:
        searchers = [
            asyncio.create_task(
                search_location_with_logs(ll, log_dir, loc, level="DEBUG"),
                name=loc,
            )
            for loc in test_locations
        ]
        output = await asyncio.gather(*searchers)

    expected_words = ["paso", "decatur"]
    assert len(output) == 2
    for query_results, expected_word in zip(output, expected_words):
        assert len(query_results) == 1
        assert len(query_results[0]) == num_requested_links
        assert any(expected_word in link for link in query_results[0])

    log_files = list(log_dir.glob("*"))
    assert len(log_files) == 2
    for fp in log_files:
        assert (
            fp.read_text()
            == f"A generic test log\nThis location is {fp.stem!r}\n"
        )


@pytest.mark.asyncio
async def test_async_file_loader_with_temp_cache(monkeypatch):
    """Test `AsyncFileLoader` with a `TempFileCache` service"""

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

    with open(WHATCOM_DOC_PATH, "r", encoding="utf-8") as fh:
        content = fh.read()

    truth = HTMLDocument([content])

    async with RunningAsyncServices([TempFileCache()]):
        loader = AsyncFileLoader(file_cache_coroutine=TempFileCache.call)
        doc = await loader.fetch(url="Whatcom")
        assert doc.text == truth.text
        assert doc.metadata["source"] == "Whatcom"
        cached_fp = doc.metadata["cache_fn"]
        assert cached_fp.exists()
        assert cached_fp.read_text(encoding="utf-8") == doc.text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
