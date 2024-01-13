# -*- coding: utf-8 -*-
"""ELM Ordinance integration tests"""
import time
from pathlib import Path

import httpx
import pytest
import openai

from elm.ords.services.usage import TimeBoundedUsageTracker, UsageTracker
from elm.ords.services.openai import OpenAIService, usage_from_response
from elm.ords.services.provider import RunningAsyncServices


@pytest.mark.asyncio
async def test_openai_query(sample_openai_response, monkeypatch):
    """Test querying OpenAI while tracking limits and usage"""

    start_time = None
    elapsed_times = []

    async def _test_response(*args, **kwargs):
        time_elapsed = time.time() - start_time
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

    client = openai.AsyncOpenAI()
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        _test_response,
        raising=True,
    )
    rate_tracker = TimeBoundedUsageTracker(max_seconds=10)
    openai_service = OpenAIService(
        client, rate_limit=10, rate_tracker=rate_tracker
    )

    usage_tracker = UsageTracker("my_county", usage_from_response)
    async with RunningAsyncServices([openai_service]):
        start_time = time.time()
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
        assert elapsed_times[2] >= 14

        assert usage_tracker == {
            "default": {
                "requests": 1,
                "prompt_tokens": 100,
                "response_tokens": 10,
            }
        }

        time.sleep(10)
        assert openai_service.rate_tracker.total == 0

        start_time = time.time() - 4
        await OpenAIService.call(model="gpt-4")
        await OpenAIService.call(model="gpt-4")
        assert len(elapsed_times) == 5
        assert elapsed_times[-2] - 4 < 1
        assert elapsed_times[-1] - 4 > 10

        time.sleep(11)
        start_time = time.time() - 4
        assert openai_service.rate_tracker.total == 0
        message = await OpenAIService.call(
            usage_tracker=usage_tracker, model="gpt-4", bad_request=True
        )
        assert message is None
        assert openai_service.rate_tracker.total < 10
        assert usage_tracker == {
            "default": {
                "requests": 1,
                "prompt_tokens": 100,
                "response_tokens": 10,
            }
        }


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
