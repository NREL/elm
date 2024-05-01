# -*- coding: utf-8 -*-
"""Fixtures for use across all tests."""
import asyncio

import pytest
from openai.types import Completion, CompletionUsage, CompletionChoice
from openai.types.chat import ChatCompletionMessage

from elm.ords.services.base import Service


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


@pytest.fixture
def service_base_class():
    """Base implementation of service for testing"""
    job_order = []

    class TestService(Service):
        """Basic service implementation for testing."""

        NUMBER = 0
        LEN_SLEEP = 0
        STAGGER = 0

        def __init__(self):
            """Initialize service."""
            self.running_jobs = set()

        @property
        def can_process(self):
            """True if number of running jobs less that the class number."""
            return len(self.running_jobs) < self.NUMBER

        async def process(self, job_id):
            """Mock processing of input."""
            self.running_jobs.add(job_id)
            job_order.append((self.NUMBER, job_id))
            await asyncio.sleep(self.LEN_SLEEP + self.STAGGER * job_id * 0.5)
            self.running_jobs.remove(job_id)
            return self.NUMBER

    return job_order, TestService


@pytest.fixture
def sample_openai_response():
    """Function to get sample openAI response that can be used for tests"""

    def _get_response(
        content="test_response",
        kwargs=None,
        completion_tokens=10,
        prompt_tokens=100,
        total_tokens=110,
    ):
        usage = CompletionUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )
        choice = CompletionChoice(
            finish_reason="stop",
            index=0,
            logprobs=None,
            text="",
            message=ChatCompletionMessage(content=content, role="assistant"),
        )
        llm_response = Completion(
            id="1",
            choices=[choice],
            created=0,
            model=(kwargs or {}).get("model", "gpt-4"),
            object="text_completion",
            usage=usage,
        )
        return llm_response

    return _get_response
