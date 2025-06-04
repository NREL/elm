# -*- coding: utf-8 -*-
"""ELM Web searches using search engines tests"""
import os
from pathlib import Path

import pytest

import elm.web.search.duckduckgo
import elm.web.search.google
from elm.web.search.base import APISearchEngineLinkSearch


SE_API_TO_TEST = [(elm.web.search.duckduckgo.APIDuckDuckGoSearch,
                   {"verify": False})]
if os.getenv(elm.web.search.google.APIGoogleCSESearch.API_KEY_VAR):
    SE_API_TO_TEST.append((elm.web.search.google.APIGoogleCSESearch, {}))


def test_api_key_read_from_env(monkeypatch):
    """Test that API search engine reads environ"""
    monkeypatch.setenv("TEST_API_KEY_VAR", "TEST-KEY")

    class MockAPISearchEngine(APISearchEngineLinkSearch):
        """MockAPISearchEngine"""

        API_KEY_VAR = "TEST_API_KEY_VAR"

        async def _search(self, *__, **___):
            return []

    assert MockAPISearchEngine().api_key == "TEST-KEY"


def test_no_api_key_var():
    """Test that API search engine does not break if var name is None"""

    class MockAPISearchEngine(APISearchEngineLinkSearch):
        """MockAPISearchEngine"""

        async def _search(self, *__, **___):
            return []

    assert MockAPISearchEngine().api_key is None


@pytest.mark.parametrize("queries", [['1. "NREL elm"'],
                                     ['1. "NREL elm"', "NREL reV"],])
@pytest.mark.parametrize("se", SE_API_TO_TEST)
@pytest.mark.asyncio
async def test_basic_search_query(queries, se):
    """Test basic web search query functionality"""

    num_results = 7
    se_class, kwargs = se
    search_engine = se_class(**kwargs)
    out = await search_engine.results(*queries, num_results=num_results)

    assert len(out) == len(queries)
    for results in out:
        assert len(results) == num_results
        assert all(link.startswith("http") for link in results)
        assert all("+" not in link for link in results)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
