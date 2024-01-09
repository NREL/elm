# -*- coding: utf-8 -*-
"""Test ELM Ordinance usage utilities"""
import time
import asyncio
from pathlib import Path

import pytest

from elm.ords.utilities.usage import (
    count_openai_tokens,
    TimedEntry,
    UsageTracker,
)


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
def test_count_openai_tokens(messages, model, token_count):
    """Test `count_openai_tokens` function"""
    assert count_openai_tokens(messages, model) == token_count


def test_timed_entry():
    """Test `TimedEntry` class"""

    a = TimedEntry(100)
    assert a > 10000

    time.sleep(1)
    sample_time = time.time()
    time.sleep(1)
    b = TimedEntry(10000)
    assert b > sample_time
    assert a < sample_time

    assert a.value == 100
    assert b.value == 10000


def test_usage_tracker():
    """Test the `UsageTracker` class"""

    tracker = UsageTracker(max_seconds=5)
    assert tracker.total == 0
    tracker.add(500)
    assert tracker.total == 500
    time.sleep(3)
    tracker.add(200)
    assert tracker.total == 700
    time.sleep(3)
    assert tracker.total == 200
    time.sleep(3)
    assert tracker.total == 0


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
