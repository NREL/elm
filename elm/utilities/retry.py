# -*- coding: utf-8 -*-
"""ELM retry utilities."""
import time
import random
import asyncio
import logging
from functools import wraps

import openai

from elm.exceptions import ELMRuntimeError


logger = logging.getLogger(__name__)


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
        Max number of retries before raising an `ELMRuntimeError`.
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
                    logger.info(
                        "Error: %s. Retrying in %.2f seconds.", str(e), delay
                    )
                    kwargs = _double_timeout(**kwargs)
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
        Max number of retries before raising an `ELMRuntimeError`.
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
                    logger.info(
                        "Error: %s. Retrying in %.2f seconds.", str(e), delay
                    )
                    kwargs = _double_timeout(**kwargs)
                    await asyncio.sleep(delay)

        return wrapper

    return decorator


def _handle_retries(num_retries, max_retries, error):
    """Raise error if retry attempts exceed max limit"""
    num_retries += 1
    if num_retries > max_retries:
        msg = f"Maximum number of retries ({max_retries}) exceeded"
        raise ELMRuntimeError(msg) from error
    return num_retries


def _compute_delay(delay, exponential_base, jitter):
    """Compute the next delay time"""
    return delay * exponential_base * (1 + jitter * random.random())


def _double_timeout(**kwargs):
    """Double timeout parameter if it exists in kwargs."""
    if "timeout" not in kwargs:
        return kwargs

    prev_timeout = kwargs["timeout"]
    logger.info(
        "Detected 'timeout' key in kwargs. Doubling this input from "
        "%.2f to %.2f for next iteration.",
        prev_timeout,
        prev_timeout * 2,
    )
    kwargs["timeout"] = prev_timeout * 2
    return kwargs
