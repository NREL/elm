# -*- coding: utf-8 -*-
"""Test ELM Ordinances Base Services"""
import time
from pathlib import Path

import pytest

from elm.ords.services.base import RateLimitedService
from elm.ords.services.usage import TimeBoundedUsageTracker


def test_rate_limited_service():
    """Test base implementation of `RateLimitedService` class"""

    class TestService(RateLimitedService):
        async def process(self, fut, *args, **kwargs):
            fut.set_result(0)

    usage_tracker = TimeBoundedUsageTracker(max_seconds=5)
    service = TestService(rate_limit=100, usage_tracker=usage_tracker)

    assert service.can_process
    service.usage_tracker.add(50)
    assert service.can_process
    service.usage_tracker.add(75)
    assert not service.can_process
    time.sleep(6)
    assert service.can_process


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
