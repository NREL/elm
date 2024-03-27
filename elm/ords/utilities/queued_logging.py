# -*- coding: utf-8 -*-
"""ELM Ordinance queued logging.

This module implements queued logging, mostly following this blog:"
https://www.zopatista.com/python/2019/05/11/asyncio-logging/
"""
import asyncio
import logging
from pathlib import Path
from queue import SimpleQueue
from logging.handlers import QueueHandler, QueueListener


LOGGING_QUEUE = SimpleQueue()


class NoLocationFilter(logging.Filter):
    """Filter that catches all records without a location attribute."""

    def filter(self, record):
        """Filter logging record.

        Parameters
        ----------
        record : logging.LogRecord
            Log record containing the log message + default attributes.
            If the ``location`` attribute is missing or is a string in
            the form "Task-XX", the filter returns ``True`` (i.e. record
            is emitted).

        Returns
        -------
        bool
            If the record's ``location`` attribute is "missing".
        """
        record_location = getattr(record, "location", None)
        return record_location is None or "Task-" in record_location


class LocationFilter(logging.Filter):
    """Filter down to logs from a coroutine processing a specific location."""

    def __init__(self, location):
        """

        Parameters
        ----------
        location : str
            Location identifier. For example, ``"El Paso Colorado"``.
        """
        self.location = location

    def filter(self, record):
        """Filter logging record.

        Parameters
        ----------
        record : logging.LogRecord
            Log record containing the log message + default attributes.
            Must have a ``location`` attribute that is a string
            identifier, or this function will return ``False`` every
            time. The ``location`` identifier will be checked against
            the filter's location attribute to determine the output
            result.

        Returns
        -------
        bool
            If the record's ``location`` attribute matches the filter's
            ``location`` attribute.
        """
        record_location = getattr(record, "location", None)
        return record_location is not None and record_location == self.location


class LocalProcessQueueHandler(QueueHandler):
    """QueueHandler that works within a single process (locally)."""

    def emit(self, record):
        """Emit record with a location attribute equal to current asyncio task.

        Parameters
        ----------
        record : logging.LogRecord
            Log record containing the log message + default attributes.
            This record will get a ``location`` attribute dynamically
            added, with a value equal to the name of the current asyncio
            task (i.e. ``asyncio.current_task().get_name()``).
        """
        record.location = asyncio.current_task().get_name()
        try:
            self.enqueue(record)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.handleError(record)


class LogListener:
    """Class to listen to logging queue from coroutines and write to files."""

    def __init__(self, logger_names, level="INFO"):
        """

        Parameters
        ----------
        logger_names : iterable
            An iterable of string, where each string is a logger name.
            The logger corresponding to each of the names will be
            equipped with a logging queue handler.
        level : str, optional
            Log level to set for each logger. By default, ``"INFO"``.
        """
        self.logger_names = logger_names
        self.level = level
        self._listener = None
        self._queue_handler = LocalProcessQueueHandler(LOGGING_QUEUE)

    def _setup_listener(self):
        """Set up the queue listener"""
        if self._listener is not None:
            return
        self._listener = QueueListener(
            LOGGING_QUEUE, logging.NullHandler(), respect_handler_level=True
        )
        self._listener.handlers = list(self._listener.handlers)

    def _add_queue_handler_to_loggers(self):
        """Add a queue handler to each logger requested by user"""
        for logger_name in self.logger_names:
            logger = logging.getLogger(logger_name)
            logger.addHandler(self._queue_handler)
            logger.setLevel(self.level)

    def _remove_queue_handler_from_loggers(self):
        """Remove the queue handler from each logger requested by user"""
        for logger_name in self.logger_names:
            logging.getLogger(logger_name).removeHandler(self._queue_handler)

    def _remove_all_handlers_from_listener(self):
        """Remove all handlers still attached to listener."""
        if self._listener is None:
            return
        for handler in self._listener.handlers:
            handler.close()
            self._listener.handlers.remove(handler)

    def __enter__(self):
        self._setup_listener()
        self._add_queue_handler_to_loggers()
        self._listener.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._listener.stop()
        self._remove_queue_handler_from_loggers()
        self._remove_all_handlers_from_listener()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        self.__exit__(exc_type, exc, tb)

    def addHandler(self, handler):
        """Add a handler to the queue listener.

        Logs that are sent to the queue will be emitted to the handler.

        Parameters
        ----------
        handler : logging.Handler
            Log handler to parse log records.
        """
        if handler not in self._listener.handlers:
            self._listener.handlers.append(handler)

    def removeHandler(self, handler):
        """Remove a handler from the queue listener.

        Logs that are sent to the queue will no longer be emitted to the
        handler.

        Parameters
        ----------
        handler : logging.Handler
            Log handler to remove from queue listener.
        """
        if handler in self._listener.handlers:
            handler.close()
            self._listener.handlers.remove(handler)


class LocationFileLog:
    """Context manager to write logs for a location to a unique file."""

    def __init__(self, listener, log_dir, location, level="INFO"):
        """

        Parameters
        ----------
        listener : :class:`~elm.ords.utilities.queued_logging.LoggingListener`
            A listener instance. The file handler will be added to this
            listener.
        log_dir : path-like
            Path to output directory to contain log file.
        location : str
            Location identifier. For example, ``"El Paso Colorado"``.
            This string will become part of the file name, so it must
            contain only characters valid in a file name.
        level : str, optional
            Log level. By default, ``"INFO"``.
        """
        self.log_dir = Path(log_dir)
        self.location = location
        self.level = level
        self._handler = None
        self._listener = listener

    def _create_log_dir(self):
        """Create log output directory if it doesn't exist."""
        self.log_dir.mkdir(exist_ok=True, parents=True)

    def _setup_handler(self):
        """Setup the file handler for this location."""
        self._handler = logging.FileHandler(
            self.log_dir / f"{self.location}.log", encoding="utf-8"
        )
        self._handler.setLevel(self.level)
        self._handler.addFilter(LocationFilter(self.location))

    def _break_down_handler(self):
        """Tear down the file handler for this location."""
        if self._handler is None:
            return

        self._handler.close()
        self._handler = None

    def _add_handler_to_listener(self):
        """Add the file handler for this location to the queue listener."""
        if self._handler is None:
            raise ValueError("Must set up handler before listener!")

        self._listener.addHandler(self._handler)

    def _remove_handler_from_listener(self):
        """Remove the file handler for this location from the listener."""
        if self._handler is None:
            return

        self._listener.removeHandler(self._handler)

    def __enter__(self):
        self._create_log_dir()
        self._setup_handler()
        self._add_handler_to_listener()

    def __exit__(self, exc_type, exc, tb):
        self._remove_handler_from_listener()
        self._break_down_handler()

    async def __aenter__(self):
        self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        self.__exit__(exc_type, exc, tb)
