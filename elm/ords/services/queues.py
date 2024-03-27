# -*- coding: utf-8 -*-
"""Module for "singleton" QUERIES dictionary"""
import asyncio


_QUEUES = {}


def initialize_service_queue(service_name):
    """Initialize an `asyncio.Queue()` for a service.

    Repeated calls to this function return the same queue

    Parameters
    ----------
    service_name : str
        Name of service to initialize queue for.

    Returns
    -------
    asyncio.Queue()
        Queue instance for this service.
    """
    return _QUEUES.setdefault(service_name, asyncio.Queue())


def tear_down_service_queue(service_name):
    """Remove the queue for a service.

    The queue does not have to exist, so repeated calls to this function
    are OK.

    Parameters
    ----------
    service_name : str
        Name of service to delete queue for.
    """
    _QUEUES.pop(service_name, None)


def get_service_queue(service_name):
    """Retrieve the queue for a service.

    Parameters
    ----------
    service_name : str
        Name of service to retrieve queue for.

    Returns
    -------
    asyncio.Queue() | None
        Queue instance for this service, or `None` if the queue was not
        initialized.
    """
    return _QUEUES.get(service_name)
