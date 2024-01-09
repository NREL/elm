# -*- coding: utf-8 -*-
"""ELM abstract Service class."""
import asyncio
import logging
from abc import ABC, abstractmethod

from elm.ords.services.queues import get_service_queue
from elm.ords.utilities.exceptions import ELMOrdsNotInitializedError


logger = logging.getLogger(__name__)


class Service(ABC):
    """Abstract base class for a Service that can be queued to run."""

    @classmethod
    def _queue(cls):
        """Get queue for class."""
        queue = get_service_queue(cls.__name__)
        if queue is None:
            raise ELMOrdsNotInitializedError("Must initialize the queue!")
        return queue

    @classmethod
    async def call(cls, *args, **kwargs):
        """Call the service.

        Parameters
        ----------
        *args, **kwargs
            Positional and keyword arguments to be passed to the
            underlying service processing function.

        Returns
        -------
        obj
            A response object from the underlying service.
        """
        fut = asyncio.Future()
        await cls._queue().put((fut, args, kwargs))
        return await fut

    @property
    def name(self):
        """str: Service name used to pull the correct queue object."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def can_process(self):
        """Check if process function can be called.

        This should be a fast-running method that returns a boolean
        indicating wether or not the service can accept more
        processing calls.
        """

    @abstractmethod
    async def process(self, fut, *args, **kwargs):
        """Process a call to the service.

        Parameters
        ----------
        fut : asyncio.Future
            A future object that should get the result of the processing
            operation. If the processing function returns ``answer``,
            this method should call ``fut.set_result(answer)``.
        *args, **kwargs
            Positional and keyword arguments to be passed to the
            underlying processing function.
        """


class RateLimitedService(Service):
    """Abstract Base Class representing a rate-limited service (e.g. OpenAI)"""

    def __init__(self, rate_limit, usage_tracker):
        """

        Parameters
        ----------
        rate_limit : int | float
            Max usage per duration of the usage tracker. For example,
            if the usage tracker is set to compute the total over
            minute-long intervals, this value should be the max usage
            per minute.
        usage_tracker : `elm.ords.utilities.usage.UsageTracker`
            A UsageTracker instance. This will be used to track usage
            per time interval and compare to `rate_limit`.
        """
        self.rate_limit = rate_limit
        self.usage_tracker = usage_tracker

    @property
    def can_process(self):
        """Check if usage is under the rate limit."""
        return self.usage_tracker.total < self.rate_limit
