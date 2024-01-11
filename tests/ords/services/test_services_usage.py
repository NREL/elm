# -*- coding: utf-8 -*-
"""Test ELM Ordinance service usage functions and classes"""
import time
from pathlib import Path

import pytest

from elm.ords.services.usage import TimedEntry, TimeBoundedUsageTracker


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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
