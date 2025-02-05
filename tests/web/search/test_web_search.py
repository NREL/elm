# -*- coding: utf-8 -*-
"""ELM Web searches using search engines tests"""
import os
from pathlib import Path

import pytest
from rebrowser_playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError
)

import elm.web.search.google
import elm.web.search.duckduckgo


SE_TO_TEST = [(elm.web.search.google.PlaywrightGoogleLinkSearch, {}),
              (elm.web.search.duckduckgo.PlaywrightDuckDuckGoLinkSearch, {})]
if CSE_URL := os.getenv("GOOGLE_CSE_URL"):
    SE_TO_TEST.append((elm.web.search.google.PlaywrightGoogleCSELinkSearch,
                       {"cse_url": CSE_URL}))


@pytest.mark.parametrize(
    "queries",
    [
        ['1. "Python Programming Language"'],
        ['1. "Python Programming Language"', "Python"],
    ],
)
@pytest.mark.parametrize("se", SE_TO_TEST)
@pytest.mark.parametrize("num_results", [1, 10, 50])
@pytest.mark.asyncio
async def test_basic_search_query(queries, se, num_results):
    """Test basic google web search query functionality"""

    se_class, kwargs = se
    search_engine = se_class(**kwargs)
    out = await search_engine.results(*queries, num_results=num_results)

    assert len(out) == len(queries)
    for results in out:
        assert len(results) == min(num_results,
                                   search_engine.MAX_RESULTS_PER_PAGE)
        assert all(link.startswith("http") for link in results)


@pytest.mark.parametrize("se", SE_TO_TEST)
@pytest.mark.asyncio
async def test_search_query_with_timeout(monkeypatch, se):
    """Test google web search query with a timeout"""

    se_class, kwargs = se
    og_tps = se_class._perform_search

    async def _tps(obj, page, search_query):
        if search_query == "Python":
            raise PlaywrightTimeoutError("test")
        return await og_tps(obj, page, search_query)

    monkeypatch.setattr(se_class, "_perform_search", _tps, raising=True)

    search_engine = se_class(**kwargs)
    out = await search_engine.results('1. "Python Programming Language"',
                                      "Python", num_results=3)

    assert len(out) == 2
    assert len(out[0]) == 3
    assert all(link.startswith("http") for link in out[0])
    assert not out[1]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
