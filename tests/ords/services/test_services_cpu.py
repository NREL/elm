# -*- coding: utf-8 -*-
"""Test ELM Ordinances CPU Services"""
import logging
import asyncio
from pathlib import Path

import pytest

from elm.ords.services.cpu import ProcessPoolService
from elm.ords.services.provider import RunningAsyncServices
from elm.ords.utilities.queued_logging import LocationFileLog, LogListener


logger = logging.getLogger("elm")


def _log_from_process():
    """Call logger instance from a process"""
    msg = "HELLO WORLD"
    logger.debug(msg)
    return msg


@pytest.mark.asyncio
async def test_logging_within_service(tmp_path):
    """Test that logging within a CPU service doesn't crash the process"""

    class ProcessLogging(ProcessPoolService):
        """Subclass for testing"""

        @property
        def can_process(self):
            return True

        async def process(self):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.pool, _log_from_process)

    log_listener = LogListener(["elm"], level="DEBUG")
    services = [ProcessLogging()]

    async with RunningAsyncServices(services), log_listener as ll:
        with LocationFileLog(
            ll, tmp_path, location="test_loc", level="DEBUG"
        ):
            msg = await ProcessLogging.call()

    assert msg == "HELLO WORLD"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
