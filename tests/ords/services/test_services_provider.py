# -*- coding: utf-8 -*-
"""Test Service Provider"""
import asyncio
from pathlib import Path

import pytest

from elm.ords.services.base import Service
from elm.ords.services.provider import RunningAsyncServices
from elm.ords.utilities.exceptions import (
    ELMOrdsNotInitializedError,
    ELMOrdsValueError,
)


@pytest.mark.asyncio
async def test_services_provider():
    """Test that services provider works as expected"""

    job_order = []

    class TestService(Service):
        NUMBER = 0
        LEN_SLEEP = 0

        def __init__(self):
            self.running_jobs = set()

        @property
        def can_process(self):
            return len(self.running_jobs) < self.NUMBER

        async def process(self, fut, job_id):
            self.running_jobs.add(job_id)
            job_order.append((self.NUMBER, job_id))
            await asyncio.sleep(self.LEN_SLEEP)
            fut.set_result(self.NUMBER)
            self.running_jobs.remove(job_id)

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
async def test_services_provider_staggered_jobs():
    """Test that services provider works as expected with staggered jobs"""

    job_order = []

    class TestService(Service):
        NUMBER = 0
        LEN_SLEEP = 0

        def __init__(self):
            self.running_jobs = set()

        @property
        def can_process(self):
            return len(self.running_jobs) < self.NUMBER

        async def process(self, fut, job_id):
            self.running_jobs.add(job_id)
            job_order.append((self.NUMBER, job_id))
            await asyncio.sleep(self.LEN_SLEEP + job_id * 0.5)
            fut.set_result(self.NUMBER)
            self.running_jobs.remove(job_id)

    class AlwaysThreeService(TestService):
        NUMBER = 3
        LEN_SLEEP = 5

    class AlwaysTenService(TestService):
        NUMBER = 5
        LEN_SLEEP = 8

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
async def test_services_provider_no_submissions_allowed_at_start():
    """Test that services provider works as expected"""

    job_order = []

    class AlwaysThreeService(Service):
        NUMBER = 3

        def __init__(self):
            self.running_jobs = set()
            self.n_requests = -1

        @property
        def can_process(self):
            self.n_requests += 1
            if self.n_requests < 10:
                return False
            return len(self.running_jobs) < self.NUMBER

        async def process(self, fut, job_id):
            self.running_jobs.add(job_id)
            job_order.append((self.NUMBER, job_id))
            await asyncio.sleep(0)
            fut.set_result(self.NUMBER)
            self.running_jobs.remove(job_id)

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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])