# -*- coding: utf-8 -*-
"""ELM Web google searching tests"""
from pathlib import Path

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

import elm.web.google_search


@pytest.mark.parametrize(
    "queries",
    [
        ['1. "Python Programming Language"'],
        ['1. "Python Programming Language"', "Python"],
    ],
)
@pytest.mark.parametrize("num_results", [1, 10, 50])
@pytest.mark.asyncio
async def test_basic_search_query(queries, num_results):
    """Test basic google web search query functionality"""

    search_engine = elm.web.google_search.PlaywrightGoogleLinkSearch()
    out = await search_engine.results(*queries, num_results=num_results)

    assert len(out) == len(queries)
    for results in out:
        assert len(results) == min(
            num_results,
            search_engine.EXPECTED_RESULTS_PER_PAGE,
        )
        assert all(link.startswith("http") for link in results)


@pytest.mark.asyncio
async def test_search_query_with_timeout(monkeypatch):
    """Test google web search query with a timeout"""

    og_tps = elm.web.google_search._perform_google_search

    async def _tps(page, search_query):
        if search_query == "Python":
            raise PlaywrightTimeoutError("test")
        return await og_tps(page, search_query)

    monkeypatch.setattr(
        elm.web.google_search,
        "_perform_google_search",
        _tps,
        raising=True,
    )

    search_engine = elm.web.google_search.PlaywrightGoogleLinkSearch()
    out = await search_engine.results(
        '1. "Python Programming Language"', "Python", num_results=3
    )

    assert len(out) == 2
    assert len(out[0]) == 3
    assert all(link.startswith("http") for link in out[0])
    assert not out[1]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
