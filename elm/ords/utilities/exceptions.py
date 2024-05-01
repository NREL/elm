# -*- coding: utf-8 -*-
"""Custom Exceptions and Errors for ELM Ordinances. """
import logging

from elm.exceptions import ELMError


logger = logging.getLogger("elm")


class ELMOrdsError(ELMError):
    """Generic ELM Ordinance Error."""

    def __init__(self, *args, **kwargs):
        """Init exception and broadcast message to logger."""
        super().__init__(*args, **kwargs)
        if args:
            logger.error(str(args[0]), stacklevel=2)


class ELMOrdsNotInitializedError(ELMOrdsError):
    """ELM Ordinances not initialized error."""


class ELMOrdsValueError(ELMOrdsError, ValueError):
    """ELM Ordinances ValueError."""


class ELMOrdsRuntimeError(ELMOrdsError, RuntimeError):
    """ELM Ordinances RuntimeError."""
