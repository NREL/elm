# -*- coding: utf-8 -*-
"""Custom Exceptions and Errors for ELM. """


class ELMError(Exception):
    """Generic ELM Error."""


class ELMKeyError(ELMError, KeyError):
    """ELM Key Error."""


class ELMInputError(ELMError, ValueError):
    """ELM Input (Value) Error."""


class ELMRuntimeError(ELMError, RuntimeError):
    """ELM RuntimeError."""
