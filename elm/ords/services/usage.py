# -*- coding: utf-8 -*-
"""ELM Ordinances usage tracking utilities."""
import time
import logging
from collections import deque
from functools import total_ordering, wraps

import openai

logger = logging.getLogger(__name__)


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
