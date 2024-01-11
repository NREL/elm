# -*- coding: utf-8 -*-
"""Test ELM Ordinance openai services"""
from pathlib import Path

import pytest
from openai.types import Completion, CompletionUsage

from elm.ords.services.openai import count_tokens, usage_from_response


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
            {"requests": 11, "prompt_tokens": 100, "response_tokens": 20},
        ),
    ],
)
def test_usage_from_response(usage_input, expected_output):
    """Test `usage_from_response` function"""

    usage = CompletionUsage(
        completion_tokens=10, prompt_tokens=100, total_tokens=110
    )
    llm_response = Completion(
        id="1",
        choices=[],
        created=0,
        model="gpt-4",
        object="text_completion",
        usage=usage,
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
