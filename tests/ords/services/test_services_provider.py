# -*- coding: utf-8 -*-
"""Test Service Provider"""
import asyncio
from pathlib import Path

import pytest

from elm.ords.services.base import Service
from elm.ords.services.provider import RunningAsyncServices, _RunningProvider
from elm.ords.utilities.exceptions import (
    ELMOrdsNotInitializedError,
    ELMOrdsValueError,
)


@pytest.mark.asyncio
async def test_services_provider(service_base_class):
    """Test that services provider works as expected"""

    job_order, TestService = service_base_class

    class AlwaysThreeService(TestService):
        NUMBER = 3
        LEN_SLEEP = 5

    class AlwaysTenService(TestService):
        NUMBER = 5

    with pytest.raises(ELMOrdsNotInitializedError):
        AlwaysTenService._queue()

    with pytest.raises(ELMOrdsValueError):
        async with RunningAsyncServices([]):
            pass

    services = [AlwaysThreeService(), AlwaysTenService()]
    async with RunningAsyncServices(services):
        a3_producers = [
            asyncio.create_task(AlwaysThreeService.call(i)) for i in range(4)
        ]
        a10_producers = [
            asyncio.create_task(AlwaysTenService.call(i)) for i in range(7)
        ]
        out = await asyncio.gather(*(a3_producers + a10_producers))

    expected_job_out = [3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5]
    expected_job_order = [
        (3, 0),
        (5, 0),
        (3, 1),
        (5, 1),
        (3, 2),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (3, 3),
    ]
    assert out == expected_job_out, f"{out=}"
    assert job_order == expected_job_order, f"{job_order=}"


@pytest.mark.asyncio
async def test_services_provider_staggered_jobs(service_base_class):
    """Test that services provider works as expected with staggered jobs"""

    job_order, TestService = service_base_class

    class AlwaysThreeService(TestService):
        NUMBER = 3
        LEN_SLEEP = 5
        STAGGER = 1

    class AlwaysTenService(TestService):
        NUMBER = 5
        LEN_SLEEP = 8
        STAGGER = 1

    services = [AlwaysThreeService(), AlwaysTenService()]
    async with RunningAsyncServices(services):
        a3_producers = [
            asyncio.create_task(AlwaysThreeService.call(i)) for i in range(5)
        ]
        a10_producers = [
            asyncio.create_task(AlwaysTenService.call(i)) for i in range(8)
        ]
        out = await asyncio.gather(*(a3_producers + a10_producers))

    expected_job_out = [3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5]
    expected_job_order = [
        (3, 0),
        (5, 0),
        (3, 1),
        (5, 1),
        (3, 2),
        (5, 2),
        (5, 3),
        (5, 4),
        (3, 3),
        (3, 4),
        (5, 5),
        (5, 6),
        (5, 7),
    ]
    assert out == expected_job_out, f"{out=}"
    assert job_order == expected_job_order, f"{job_order=}"


@pytest.mark.asyncio
async def test_services_provider_no_submissions_allowed_at_start(
    service_base_class,
):
    """Test that services provider works even when service is not ready."""

    job_order, TestService = service_base_class

    class AlwaysThreeService(TestService):
        NUMBER = 3

        def __init__(self):
            super().__init__()
            self.n_requests = -1

        @property
        def can_process(self):
            self.n_requests += 1
            if self.n_requests < 10:
                return False
            return super().can_process

    services = [AlwaysThreeService()]
    async with RunningAsyncServices(services):
        a3_producers = [
            asyncio.create_task(AlwaysThreeService.call(i)) for i in range(4)
        ]
        out = await asyncio.gather(*a3_producers)

    expected_job_out = [3, 3, 3, 3]
    expected_job_order = [(3, 0), (3, 1), (3, 2), (3, 3)]
    assert out == expected_job_out, f"{out=}"
    assert job_order == expected_job_order, f"{job_order=}"


@pytest.mark.asyncio
async def test_services_provider_raises_error():
    """Test that services provider raises error if service does."""

    class BadService(Service):
        @property
        def can_process(self):
            return True

        async def process(self, *args, **kwargs):
            raise ValueError("A test error")

    services = [BadService()]
    with pytest.raises(ValueError) as exc_info:
        async with RunningAsyncServices(services):
            await BadService.call()

    assert "A test error" in str(exc_info)


@pytest.mark.asyncio
async def test_services_provider_submits_as_long_as_needed(monkeypatch):
    """Test that services provider continues to submit jobs while it can."""

    call_cache = []

    async def collect_responses(self):
        call_cache.append(len(self.jobs))
        if not self.jobs:
            return
        complete, __ = await asyncio.wait(
            self.jobs, return_when=asyncio.FIRST_COMPLETED
        )
        for job in complete:
            self.jobs.remove(job)

    monkeypatch.setattr(
        _RunningProvider,
        "collect_responses",
        collect_responses,
        raising=True,
    )

    class FastService(Service):
        @property
        def can_process(self):
            return True

        async def process(self, *args, **kwargs):
            return True

    services = [FastService()]
    async with RunningAsyncServices(services):
        producers = [
            asyncio.create_task(FastService.call()) for _ in range(10)
        ]
        out = await asyncio.gather(*producers)

    assert out == [True] * 10
    assert not call_cache


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
