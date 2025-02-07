# -*- coding: utf-8 -*-
"""ELM Web searches using search engines tests"""
from pathlib import Path

import pytest

from elm.web.search.run import (_single_se_search, _down_select_urls,
                                _init_se, _load_docs, _single_se_search)
from elm.web.search.google import (APIGoogleCSESearch,
                                   PlaywrightGoogleLinkSearch)
from elm.exceptions import ELMKeyError


@pytest.mark.asyncio
async def test_single_se_search_name_dne():
    """Test error for unknown search engine"""
    with pytest.raises(ELMKeyError) as err:
        await _single_se_search("DNE", None, None, None, None, None)

    assert "'se_name' must be one of" in str(err)


def test_down_select_urls_empty():
    """Test down selecting empty list"""
    assert _down_select_urls([]) == set()


def test_down_select_urls_empty_queries():
    """Test down selecting when all URLs results are empty"""
    assert _down_select_urls([[[]], [[]]]) == set()


def test_down_select_urls_diff_lens():
    """Test down selecting URLs result lengths differ"""
    assert _down_select_urls([[['ab']], [['bc', 'cd']]]) == {'ab', 'bc', 'cd'}


def test_down_select_urls_one_empty():
    """Test down selecting URLs when one result is empty"""
    assert _down_select_urls([[[]], [['bc', 'cd']]]) == {'bc', 'cd'}


def test_init_se():
    """Test initializing a playwright search engine"""
    test_kwargs = {"pw_launch_kwargs": {"test": 1}}
    se, *__ = _init_se("PlaywrightGoogleLinkSearch", test_kwargs)
    assert isinstance(se, PlaywrightGoogleLinkSearch)
    assert se.launch_kwargs == {"test": 1}
    assert test_kwargs == {"pw_launch_kwargs": {"test": 1}}


def test_init_se_pop_kwargs():
    """Test that kwargs are correctly popped in func"""
    test_kwargs = {"pw_launch_kwargs": {"test": 1},
                   "google_cse_api_kwargs": {"api_key": "test_key"}}

    se, *__ = _init_se("APIGoogleCSESearch", test_kwargs)
    assert isinstance(se, APIGoogleCSESearch)
    assert se.api_key == "test_key"
    assert test_kwargs == {"pw_launch_kwargs": {"test": 1}}


@pytest.mark.asyncio
async def test_load_docs_empty():
    """Test loading docs for no URLs"""
    assert await _load_docs(set()) == []


@pytest.mark.asyncio
async def test_single_se_search_bad_build():
    """Test that bad init of SE gives no results"""
    test_kwargs = {"google_cse_api_kwargs": {"dne_arg": "test_key"}}
    results = await _single_se_search("APIGoogleCSESearch", [""], None, None,
                                      None, test_kwargs)
    assert results == set()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
