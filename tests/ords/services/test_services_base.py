# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Test ELM Ordinances Base Services"""
import time
from pathlib import Path

import pytest

from elm.ords.services.base import RateLimitedService
from elm.ords.services.usage import TimeBoundedUsageTracker


def test_rate_limited_service():
    """Test base implementation of `RateLimitedService` class"""

    class TestService(RateLimitedService):
        """Simple service implementation for tests."""

        async def process(self, *args, **kwargs):
            """Always return 0."""
            return 0

    rate_tracker = TimeBoundedUsageTracker(max_seconds=5)
    service = TestService(rate_limit=100, rate_tracker=rate_tracker)

    assert service.can_process
    service.rate_tracker.add(50)
    assert service.can_process
    service.rate_tracker.add(75)
    assert not service.can_process
    time.sleep(6)
    assert service.can_process


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
