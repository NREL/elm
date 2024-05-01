# -*- coding: utf-8 -*-
"""Custom Exceptions and Errors for ELM. """


class ELMError(Exception):
    """Generic ELM Error."""


class ELMRuntimeError(ELMError, RuntimeError):
    """ELM RuntimeError."""
