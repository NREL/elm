# -*- coding: utf-8 -*-
"""ELM Web searches using search engines tests"""
from pathlib import Path

import pytest

from elm.web.search.base import APISearchEngineLinkSearch


def test_api_key_read_from_env(monkeypatch):
    """Test that API search engine reads environ"""
    monkeypatch.setenv("TEST_API_KEY_VAR", "TEST-KEY")


    class MockAPISearchEngine(APISearchEngineLinkSearch):
        API_KEY_VAR = "TEST_API_KEY_VAR"

        async def _search(self, *__, **___):
            return []

    assert MockAPISearchEngine().api_key == "TEST-KEY"


def test_no_api_key_var():
    """Test that API search engine does not break if var name is None"""

    class MockAPISearchEngine(APISearchEngineLinkSearch):
        async def _search(self, *__, **___):
            return []

    assert MockAPISearchEngine().api_key is None


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
