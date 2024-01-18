# -*- coding: utf-8 -*-
"""ELM service provider classes."""
import asyncio
import logging

from elm.ords.services.queues import (
    initialize_service_queue,
    get_service_queue,
    tear_down_service_queue,
)
from elm.ords.utilities.exceptions import ELMOrdsValueError


logger = logging.getLogger(__name__)


class _RunningProvider:
    """A running provider for a single service."""

    def __init__(self, service, queue):
        """

        Parameters
        ----------
        service : :cls:`elm.ords.services.base.Service`
            An instance of a single async service to run.
        queue : :cls:`asyncio.Queue`
            Queue object for the running service.
        """
        self.service = service
        self.queue = queue
        self.jobs = set()

    async def run(self):
        """Run the service."""
        while True:
            await self.submit_jobs()
            await self.collect_responses()

    async def submit_jobs(self):
        """Submit jobs from the queue to processing.

        The service can limit the number of submissions at a time by
        implementing the ``can_process`` property.

        If the queue is non-empty, the function takes jobs from it
        iteratively and submits them until the ``can_process`` property
        of teh service returns ``False``. A call to ``can_process`` is
        submitted between every job pulled from the queue, so enure that
        method is performant. If the queue is empty, this function does
        one of two things:

            1) If there are no jobs processing, it waits on the queue
               to get more jobs and submits them as they come in
               (assuming the service allows it)
            2) If there are jobs processing, this function returns
               without waiting on more jobs from the queue.

        """
        if not self.service.can_process or self._q_empty_but_still_processing:
            return

        while self.service.can_process:
            fut, args, kwargs = await self.queue.get()
            task = asyncio.create_task(
                self.service.process_using_futures(fut, *args, **kwargs)
            )
            self.queue.task_done()
            self.jobs.add(task)
            await self._allow_service_to_update()

        return

    async def _allow_service_to_update(self):
        """Switch contexts, allowing service to update if it can process"""
        await asyncio.sleep(0)

    @property
    def _q_empty_but_still_processing(self):
        """bool: Queue empty but jobs still running (don't await queue)"""
        return self.queue.empty() and self.jobs

    async def collect_responses(self):
        """Collect responses from the service.

        This call will block further submissions to the service until
        at least one job finishes.
        """
        if not self.jobs:
            return

        complete, __ = await asyncio.wait(
            self.jobs, return_when=asyncio.FIRST_COMPLETED
        )

        for job in complete:
            self.jobs.remove(job)


class RunningAsyncServices:
    """Async context manager for running services."""

    def __init__(self, services):
        """

        Parameters
        ----------
        services : iterable
            An iterable of async services to run during program
            execution.
        """
        self.services = services
        self.__providers = []
        self._validate_services()

    def _validate_services(self):
        """Validate input services."""
        if len(self.services) < 1:
            raise ELMOrdsValueError(
                "Must provide at least one service to run!"
            )

    def _reset_providers(self):
        """Reset running providers"""
        for c in self.__providers:
            c.cancel()
        self.__providers = []

    async def __aenter__(self):
        for service in self.services:
            logger.debug("Initializing Service: %s", service.name)
            queue = initialize_service_queue(service.name)
            task = asyncio.create_task(_RunningProvider(service, queue).run())
            self.__providers.append(task)

    async def __aexit__(self, exc_type, exc, tb):
        try:
            for service in self.services:
                await get_service_queue(service.name).join()
        finally:
            self._reset_providers()
            for service in self.services:
                logger.debug("Tearing down Service: %s", service.name)
                tear_down_service_queue(service.name)
