# -*- coding: utf-8 -*-
"""Test Service Queues"""
import asyncio
from pathlib import Path

import pytest

from elm.ords.services.queues import (
    initialize_service_queue,
    tear_down_service_queue,
    get_service_queue,
)


@pytest.fixture
def service_name():
    """Create a service name that definitely has no queue"""
    name = "test"
    tear_down_service_queue(name)
    yield name
    tear_down_service_queue(name)


def test_initialize_service_queue(service_name):
    """Test initializing a queue"""

    queue = initialize_service_queue(service_name)
    assert isinstance(queue, asyncio.Queue)

    queue2 = initialize_service_queue(service_name)
    assert queue is queue2


def test_tear_down_service_queue(service_name):
    """Test tearing down a queue"""

    tear_down_service_queue(service_name)
    assert get_service_queue(service_name) is None

    initialize_service_queue(service_name)
    queue = get_service_queue(service_name)
    assert isinstance(queue, asyncio.Queue)

    queue2 = get_service_queue(service_name)
    assert queue is queue2

    tear_down_service_queue(service_name)
    assert get_service_queue(service_name) is None


def test_get_service_queue():
    """Test retrieving a queue"""

    assert get_service_queue(service_name) is None

    initialize_service_queue(service_name)
    queue = get_service_queue(service_name)
    assert isinstance(queue, asyncio.Queue)

    tear_down_service_queue(service_name)
    assert get_service_queue(service_name) is None


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
