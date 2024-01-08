# -*- coding: utf-8 -*-
"""Fixtures for use across all tests."""
import pytest


LOGGING_META_FILES = {"exceptions.py"}


@pytest.fixture
def assert_message_was_logged(caplog):
    """Assert that a particular (partial) message was logged."""
    caplog.clear()

    def assert_message(msg, log_level=None, clear_records=False):
        """Assert that a message was logged."""
        assert caplog.records

        for record in caplog.records:
            if msg in record.message:
                break
        else:
            raise AssertionError(f"{msg!r} not found in log records")

        # record guaranteed to be defined b/c of "assert caplog.records"
        # pylint: disable=undefined-loop-variable
        if log_level:
            assert record.levelname == log_level
        assert record.filename not in LOGGING_META_FILES
        assert record.funcName != "__init__"
        assert "elm" in record.name

        if clear_records:
            caplog.clear()

    return assert_message
