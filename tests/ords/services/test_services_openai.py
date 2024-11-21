# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Test ELM Ordinance openai services"""
from pathlib import Path

import httpx
import pytest
import openai

from elm.ords.services.openai import (
    count_tokens,
    usage_from_response,
    OpenAIService,
)
from elm.ords.services.usage import UsageTracker


TEST_MESSAGES_1 = [
    {"role": "system", "content": "You are a friendly bot"},
    {"role": "user", "content": "How are you?"},
]
TEST_MESSAGES_2 = [
    {"role": "system", "content": "You are a friendly bot"},
    {"role": "user", "content": "I have 5 apples."},
    {"role": "system", "content": "Great!"},
    {"role": "user", "content": "How many apples do you have?."},
]


@pytest.mark.parametrize(
    "messages, model, token_count",
    [(TEST_MESSAGES_1, "gpt-4", 20), (TEST_MESSAGES_2, "gpt-4", 39)],
)
def test_count_tokens(messages, model, token_count):
    """Test `count_tokens` function"""
    assert count_tokens(messages, model) == token_count


@pytest.mark.parametrize(
    "usage_input, expected_output",
    [
        ({}, {"requests": 1, "prompt_tokens": 100, "response_tokens": 10}),
        (
            {"requests": 10, "response_tokens": 100},
            {"requests": 11, "prompt_tokens": 100, "response_tokens": 110},
        ),
    ],
)
def test_usage_from_response(
    usage_input, expected_output, sample_openai_response
):
    """Test `usage_from_response` function"""
    response = sample_openai_response()
    assert usage_from_response(usage_input, response) == expected_output


@pytest.mark.asyncio
async def test_openai_service(sample_openai_response, monkeypatch):
    """Test querying OpenAI while tracking limits and usage"""

    async def _test_response(*args, **kwargs):
        if kwargs.get("bad_request"):
            response = httpx.Response(404)
            response.request = httpx.Request(method="test", url="test")
            raise openai.BadRequestError(
                "for testing", response=response, body=None
            )
        return sample_openai_response(kwargs=kwargs)

    client = openai.AsyncOpenAI(api_key="dummy")
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        _test_response,
        raising=True,
    )
    openai_service = OpenAIService(client)

    usage_tracker = UsageTracker("my_county", usage_from_response)

    message = await openai_service.process(
        usage_tracker=usage_tracker, model="gpt-4"
    )
    assert openai_service.rate_tracker.total == 13
    assert message == "test_response"

    assert usage_tracker == {
        "default": {
            "requests": 1,
            "prompt_tokens": 100,
            "response_tokens": 10,
        }
    }

    message = await openai_service.process(
        usage_tracker=usage_tracker, model="gpt-4", bad_request=True
    )
    assert message is None
    assert openai_service.rate_tracker.total == 16
    assert usage_tracker == {
        "default": {
            "requests": 1,
            "prompt_tokens": 100,
            "response_tokens": 10,
        }
    }

    await openai_service.process(model="gpt-4")
    assert usage_tracker == {
        "default": {
            "requests": 1,
            "prompt_tokens": 100,
            "response_tokens": 10,
        }
    }


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
