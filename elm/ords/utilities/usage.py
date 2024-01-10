# -*- coding: utf-8 -*-
"""ELM Ordinances usage tracking utilities."""
import time
import random
import asyncio
import logging
from collections import deque
from contextlib import contextmanager
from functools import total_ordering, wraps

import openai

from elm.base import ApiBase
from elm.ords.utilities.exceptions import ELMOrdsRuntimeError


logger = logging.getLogger(__name__)


def count_openai_tokens(messages, model):
    """Count the number of tokens in an outgoing set of messages.

    Parameters
    ----------
    messages : list
        A list of message objects, where the latter is represented
        using a dictionary. Each message dictionary must have a
        "content" key containing the string to count tokens for.
    model : str
        The OpenAI model being used. This input will be passed to
        :func:`tiktoken.encoding_for_model`.

    Returns
    -------
    int
        Total number of tokens in the set of messages outgoing to
        OpenAI.

    References
    ----------
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    message_total = sum(
        ApiBase.count_tokens(message["content"], model=model) + 4
        for message in messages
    )
    return message_total + 3


@total_ordering
class TimedEntry:
    """An entry that performs comparisons based on time added, not value.

    Examples
    --------
    >>> a = TimedEntry(100)
    >>> a > 1000
    True
    """

    def __init__(self, value):
        """

        Parameters
        ----------
        value : obj
            Some value to store as an entry.
        """
        self.value = value
        self._time = time.time()

    def __eq__(self, other):
        return self._time == other

    def __lt__(self, other):
        return self._time < other


class TimeBoundedUsageTracker:
    """Track usage of a resource over time.

    This class wraps a double-ended queue, and any inputs older than
    a certain time are dropped. Those values are also subtracted from
    the running total.

    References
    ----------
    https://stackoverflow.com/questions/51485656/efficient-time-bound-queue-in-python
    """

    def __init__(self, max_seconds=65):
        """

        Parameters
        ----------
        max_seconds : int, optional
            Maximum age in seconds of an element before it is dropped
            from consideration. By default, ``65``.
        """
        self.max_seconds = max_seconds
        self._total = 0
        self._q = deque()

    @property
    def total(self):
        """float: Total value of all entries younger than `max_seconds`"""
        self._discard_old_values()
        return self._total

    def add(self, value):
        """Add a value to track.

        Parameters
        ----------
        value : int | float
            A new value to add to the queue. It's total will be added to
            the running total, and it will live for `max_seconds` before
            being discarded.
        """
        self._q.append(TimedEntry(value))
        self._total += value

    def _discard_old_values(self):
        """Discard 'old' values from the queue"""
        cutoff_time = time.time() - self.max_seconds
        try:
            while self._q[0] < cutoff_time:
                self._total -= self._q.popleft().value
        except IndexError:
            pass


def retry_with_exponential_backoff(
    base_delay=1,
    exponential_base=4,
    jitter=True,
    max_retries=3,
    errors=(openai.RateLimitError, openai.APITimeoutError),
):
    """Retry a synchronous function with exponential backoff.

    This decorator works out-of-the-box for OpenAI chat completions
    calls. To configure it for other functions, set the `errors` input
    accordingly.

    Parameters
    ----------
    base_delay : int, optional
        The base delay time, in seconds. This time will be multiplied by
        the exponential_base (plus any jitter) during each retry
        iteration. The multiplication applies *at the first retry*.
        Therefore, if your base delay is ``1`` and your
        `exponential_base` is ``4`` (with no jitter), the delay before
        the first retry will be ``1 * 4 = 4`` seconds. The subsequent
        delay will be ``4 * 4 = 16`` seconds, and so on.
        By default, ``1``.
    exponential_base : int, optional
        The multiplication factor applied to the base `delay` input.
        See description of `delay` for an example. By default, ``4``.
    jitter : bool, optional
        Option to include a random fractional adder (0 - 1) to the
        `exponential_base` before multiplying by the `delay`. This can
        help ensure each function call is submitted slightly offset from
        other calls in a batch and therefore help avoid repeated rate
        limit failures by a batch of submissions arriving simultaneously
        to a service. By default, ``True``.
    max_retries : int, optional
        Max number of retries before raising an `ELMOrdsRuntimeError`.
        By default, ``3``.
    errors : tuple, optional
        The error class(es) to signal a retry. Other errors will be
        propagated without retrying.
        By default, ``(openai.RateLimitError, openai.APITimeoutError)``.

    References
    ----------
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            num_retries = 0
            delay = base_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    num_retries = _handle_retries(num_retries, max_retries, e)
                    delay = _compute_delay(delay, exponential_base, jitter)
                    logger.debug(
                        f"Error: {e}.\nRetrying in {delay:.2f} seconds."
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


def async_retry_with_exponential_backoff(
    base_delay=1,
    exponential_base=4,
    jitter=True,
    max_retries=3,
    errors=(openai.RateLimitError, openai.APITimeoutError),
):
    """Retry an asynchronous function with exponential backoff.

    This decorator works out-of-the-box for OpenAI chat completions
    calls. To configure it for other functions, set the `errors` input
    accordingly.

    Parameters
    ----------
    base_delay : int, optional
        The base delay time, in seconds. This time will be multiplied by
        the exponential_base (plus any jitter) during each retry
        iteration. The multiplication applies *at the first retry*.
        Therefore, if your base delay is ``1`` and your
        `exponential_base` is ``4`` (with no jitter), the delay before
        the first retry will be ``1 * 4 = 4`` seconds. The subsequent
        delay will be ``4 * 4 = 16`` seconds, and so on.
        By default, ``1``.
    exponential_base : int, optional
        The multiplication factor applied to the base `delay` input.
        See description of `delay` for an example. By default, ``4``.
    jitter : bool, optional
        Option to include a random fractional adder (0 - 1) to the
        `exponential_base` before multiplying by the `delay`. This can
        help ensure each function call is submitted slightly offset from
        other calls in a batch and therefore help avoid repeated rate
        limit failures by a batch of submissions arriving simultaneously
        to a service. By default, ``True``.
    max_retries : int, optional
        Max number of retries before raising an `ELMOrdsRuntimeError`.
        By default, ``3``.
    errors : tuple, optional
        The error class(es) to signal a retry. Other errors will be
        propagated without retrying.
        By default, ``(openai.RateLimitError, openai.APITimeoutError)``.

    References
    ----------
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            num_retries = 0
            delay = base_delay

            while True:
                try:
                    return await func(*args, **kwargs)
                except errors as e:
                    num_retries = _handle_retries(num_retries, max_retries, e)
                    delay = _compute_delay(delay, exponential_base, jitter)
                    logger.debug(
                        f"Error: {e}.\nRetrying in {delay:.2f} seconds."
                    )
                    await asyncio.sleep(delay)

        return wrapper

    return decorator


def _handle_retries(num_retries, max_retries, error):
    """Raise error if retry attempts exceed max limit"""
    num_retries += 1
    if num_retries > max_retries:
        msg = f"Maximum number of retries ({max_retries}) exceeded"
        raise ELMOrdsRuntimeError(msg) from error
    return num_retries


def _compute_delay(delay, exponential_base, jitter):
    """Compute the next delay time"""
    return delay * exponential_base * (1 + jitter * random.random())
