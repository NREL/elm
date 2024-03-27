# -*- coding: utf-8 -*-
"""Test ELM Ordinance retry utilities"""
import time
import random
from pathlib import Path

import pytest

from elm.utilities.retry import (
    retry_with_exponential_backoff,
    async_retry_with_exponential_backoff,
)
from elm.exceptions import ELMRuntimeError


@pytest.mark.parametrize("jitter, bounds", [(False, (2, 3)), (True, (4, 5))])
def test_sync_retry(jitter, bounds, monkeypatch):
    """Test the `retry_with_exponential_backoff` decorator"""

    monkeypatch.setattr(random, "random", lambda: 1, raising=True)

    @retry_with_exponential_backoff(
        exponential_base=2, max_retries=1, jitter=jitter, errors=(ValueError,)
    )
    def failing_function():
        raise ValueError("I'm broken")

    start_time = time.monotonic()
    with pytest.raises(ELMRuntimeError):
        failing_function()
    elapsed_time = time.monotonic() - start_time
    assert bounds[0] <= elapsed_time < bounds[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("jitter, bounds", [(False, (2, 3)), (True, (4, 5))])
async def test_async_retry(jitter, bounds, monkeypatch):
    """Test the `async_retry_with_exponential_backoff` decorator"""

    monkeypatch.setattr(random, "random", lambda: 1, raising=True)

    @async_retry_with_exponential_backoff(
        exponential_base=2, max_retries=1, jitter=jitter, errors=(ValueError,)
    )
    async def failing_function():
        raise ValueError("I'm broken")

    start_time = time.monotonic()
    with pytest.raises(ELMRuntimeError):
        await failing_function()
    elapsed_time = time.monotonic() - start_time
    assert bounds[0] <= elapsed_time < bounds[1]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
