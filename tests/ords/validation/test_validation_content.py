# -*- coding: utf-8 -*-
"""Test ELM Ordinance content validation tests. """
from pathlib import Path

import pytest

from elm.ords.validation.content import (
    ValidationWithMemory,
    possibly_mentions_wind,
)


@pytest.mark.asyncio
async def test_validation_with_mem():
    """Test the `ValidationWithMemory` class (basic execution)"""

    sys_messages = []
    test_prompt = "Looking for key {key!r}"

    class MockStructuredLLMCaller:
        """Mock LLM caller for tests."""

        async def call(self, sys_msg, content, *__, **___):
            """Mock LLM call and record system message"""
            sys_messages.append(sys_msg)
            return {"test": True} if content == 0 else {}

    text_chunks = list(range(7))
    validator = ValidationWithMemory(MockStructuredLLMCaller(), text_chunks, 3)

    out = await validator.parse_from_ind(0, test_prompt, key="test")
    assert out
    assert sys_messages == ["Looking for key 'test'"]
    assert validator.memory == [{"test": True}, {}, {}, {}, {}, {}, {}]

    out = await validator.parse_from_ind(2, test_prompt, key="test")
    assert out
    assert sys_messages == ["Looking for key 'test'"] * 3
    assert validator.memory == [
        {"test": True},
        {"test": False},
        {"test": False},
        {},
        {},
        {},
        {},
    ]

    out = await validator.parse_from_ind(6, test_prompt, key="test")
    assert not out
    assert sys_messages == ["Looking for key 'test'"] * 6
    assert validator.memory == [
        {"test": True},
        {"test": False},
        {"test": False},
        {},
        {"test": False},
        {"test": False},
        {"test": False},
    ]


@pytest.mark.parametrize(
    "text,truth",
    [
        ("Wind SETBACKS", True),
        (" WECS SETBACKS", True),
        ("Window SETBACKS", False),
        ("SWECS SETBACKS", False),
        ("(wind LWET)", True),
        ("Wind SWECS", False),
        ("Wind WES", False),
        ("Wind WES\n", True),
        ("wind turbines and wind towers", True),
    ],
)
def test_possibly_mentions_wind(text, truth):
    """Test for `possibly_mentions_wind` function (basic execution)"""

    assert possibly_mentions_wind(text) == truth


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
