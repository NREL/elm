# -*- coding: utf-8 -*-
"""Test ELM Ordinance service usage functions and classes"""
import time
from pathlib import Path

import pytest

from elm.ords.services.usage import (
    TimedEntry,
    TimeBoundedUsageTracker,
    UsageTracker,
)


def _sample_response_parser(current_usage, response):
    """Sample response to usage conversion function"""
    current_usage["requests"] = current_usage.get("requests", 0) + 1
    if "tokens" in response:
        current_usage["tokens"] = response["tokens"]
    inputs = current_usage.get("inputs", 0)
    current_usage["inputs"] = inputs + response.get("inputs", 0)
    return current_usage


def test_timed_entry():
    """Test `TimedEntry` class"""

    a = TimedEntry(100)
    assert a <= time.monotonic()

    time.sleep(1)
    sample_time = time.monotonic()
    time.sleep(1)
    b = TimedEntry(10000)
    assert b > sample_time
    assert a < sample_time

    assert a.value == 100
    assert b.value == 10000


def test_time_bounded_usage_tracker():
    """Test the `TimeBoundedUsageTracker` class"""

    tracker = TimeBoundedUsageTracker(max_seconds=5)
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


def test_usage_tracker():
    """Test the `UsageTracker` class"""

    tracker = UsageTracker("test", response_parser=_sample_response_parser)
    assert tracker == {}
    assert tracker.totals == {}

    tracker.update_from_model()
    assert tracker == {}
    assert tracker.totals == {}

    tracker.update_from_model({})
    assert tracker == {"default": {"requests": 1, "inputs": 0}}
    assert tracker.totals == {"requests": 1, "inputs": 0}

    tracker.update_from_model({"inputs": 100}, sub_label="parsing")
    tracker.update_from_model()

    assert tracker == {
        "default": {"requests": 1, "inputs": 0},
        "parsing": {"requests": 1, "inputs": 100},
    }
    assert tracker.totals == {"requests": 2, "inputs": 100}

    tracker.update_from_model({"tokens": 5})

    assert tracker == {
        "default": {"requests": 2, "inputs": 0, "tokens": 5},
        "parsing": {"requests": 1, "inputs": 100},
    }
    assert tracker.totals == {"requests": 3, "inputs": 100, "tokens": 5}

    output = {"some": "value"}
    tracker.add_to(output)
    expected_out = {**tracker, "tracker_totals": tracker.totals}
    assert output == {"some": "value", "test": expected_out}


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
