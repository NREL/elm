# -*- coding: utf-8 -*-
"""ELM Web Scraping - functions to run a search"""
import pprint
import asyncio
import random
import logging
from collections import namedtuple
from itertools import zip_longest, chain
from contextlib import AsyncExitStack

from elm.web.file_loader import AsyncFileLoader
from elm.web.search.bing import PlaywrightBingLinkSearch
from elm.web.search.duckduckgo import (APIDuckDuckGoSearch,
                                       PlaywrightDuckDuckGoLinkSearch)
from elm.web.search.dux import DuxDistributedGlobalSearch
from elm.web.search.google import (APIGoogleCSESearch, APISerperSearch,
                                   CamoufoxGoogleLinkSearch,
                                   PlaywrightGoogleCSELinkSearch,
                                   PlaywrightGoogleLinkSearch)
from elm.web.search.tavily import APITavilySearch
from elm.web.search.yahoo import PlaywrightYahooLinkSearch
from elm.exceptions import ELMKeyError, ELMInputError


logger = logging.getLogger(__name__)


_RESULTS_PER_QUERY = 10
_SE_OPT = namedtuple('_SE_OPT', ['se_class', 'uses_browser', 'kwg_key_name'])
SEARCH_ENGINE_OPTIONS = {
    "APIDuckDuckGoSearch": _SE_OPT(APIDuckDuckGoSearch, False,
                                   "ddg_api_kwargs"),
    "APIGoogleCSESearch": _SE_OPT(APIGoogleCSESearch, False,
                                  "google_cse_api_kwargs"),
    "APISerperSearch": _SE_OPT(APISerperSearch, False,
                               "google_serper_api_kwargs"),
    "APITavilySearch": _SE_OPT(APITavilySearch, False, "tavily_api_kwargs"),
    "CamoufoxGoogleLinkSearch": _SE_OPT(CamoufoxGoogleLinkSearch, True,
                                        "cf_google_se_kwargs"),
    "DuxDistributedGlobalSearch": _SE_OPT(DuxDistributedGlobalSearch, False,
                                          "ddgs_kwargs"),
    "PlaywrightBingLinkSearch": _SE_OPT(PlaywrightBingLinkSearch, True,
                                        "pw_bing_se_kwargs"),
    "PlaywrightDuckDuckGoLinkSearch": _SE_OPT(PlaywrightDuckDuckGoLinkSearch,
                                              True, "pw_ddg_se_kwargs"),
    "PlaywrightGoogleCSELinkSearch": _SE_OPT(PlaywrightGoogleCSELinkSearch,
                                             True, "pw_google_cse_kwargs"),
    "PlaywrightGoogleLinkSearch": _SE_OPT(PlaywrightGoogleLinkSearch, True,
                                          "pw_google_se_kwargs"),
    "PlaywrightYahooLinkSearch": _SE_OPT(PlaywrightYahooLinkSearch, True,
                                         "pw_yahoo_se_kwargs")
}
"""Supported search engines"""
_DEFAULT_SE = ("PlaywrightGoogleLinkSearch", "PlaywrightDuckDuckGoLinkSearch",
               "DuxDistributedGlobalSearch")


async def web_search_links_as_docs(queries, search_engines=_DEFAULT_SE,
                                   num_urls=None, ignore_url_parts=None,
                                   search_semaphore=None,
                                   browser_semaphore=None, task_name=None,
                                   use_fallback_per_query=True,
                                   on_search_complete_hook=None,
                                   **kwargs):
    """Retrieve top ``N`` search results as document instances

    Parameters
    ----------
    queries : collection of str
        Collection of strings representing google queries. Documents for
        the top `num_urls` google search results (from all of these
        queries _combined_ will be returned from this function.
    search_engines : iterable of str
        Ordered collection of search engine names to attempt for web
        search. If the first search engine in the list returns a set
        of URLs, then iteration will end and documents for each URL will
        be returned. Otherwise, the next engine in this list will be
        used to run the web search. If this also fails, the next engine
        is used and so on. If all web searches fail, an empty list is
        returned. See :obj:`~elm.web.search.run.SEARCH_ENGINE_OPTIONS`
        for supported search engine options.
        By default, ``("PlaywrightGoogleLinkSearch",
        "PlaywrightDuckDuckGoLinkSearch",
        "DuxDistributedGlobalSearch")``.
    num_urls : int, optional
        Number of unique top Google search result to return as docs. The
        google search results from all queries are interleaved and the
        top `num_urls` unique URL's are downloaded as docs. If this
        number is less than ``len(queries)``, some of your queries may
        not contribute to the final output. By default, ``None``, which
        sets ``num_urls = 3 * len(queries)``.
    ignore_url_parts : iterable of str, optional
        Optional URL components to blacklist. For example, supplying
        `ignore_url_parts={"wikipedia.org"}` will ignore all URLs that
        contain "wikipedia.org". By default, ``None``.
    search_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers used to submit search engine queries open
        concurrently. For backwards-compatibility, if this input is
        ``None``, the input from `browser_semaphore` will be used in its
        place (i.e. the searches and file downloads will be limited
        using the same semaphore). By default, ``None``.
    browser_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers used to download files open concurrently.
        If ``None``, no limits are applied. By default, ``None``.
    task_name : str, optional
        Optional task name to use in :func:`asyncio.create_task`.
        By default, ``None``.
    use_fallback_per_query : bool, default=True
        Option to use the fallback list of search engines on a per-query
        basis. This means if a single query fails with one search
        engine, the fallback search engines will be attempted for that
        query. If this input is ``False``, the fallback search engines
        are only used if *all* search queries fail for a single search
        engine. By default, ``True``.
    on_search_complete_hook : callable, optional
        If provided, this async callable will be called after the search
        engine links have been retrieved. A single argument will be
        passed to this function containing a list of URL's that were the
        result of the search queries (this list cna be empty if the
        search failed). By default, ``None``.
    **kwargs
        Keyword-argument pairs to initialize
        :class:`elm.web.file_loader.AsyncFileLoader`. This input can
        also include and any/all of the following keywords:

            - ddg_api_kwargs
            - google_cse_api_kwargs
            - google_serper_api_kwargs
            - tavily_api_kwargs
            - ddgs_kwargs
            - cf_google_se_kwargs
            - pw_bing_se_kwargs
            - pw_ddg_se_kwargs
            - pw_google_cse_kwargs
            - pw_google_se_kwargs
            - pw_yahoo_se_kwargs
            - pw_launch_kwargs

        Each of these inputs should be a dictionary with
        keyword-argument pairs that you can use to initialize the search
        engines in the `search_engines` input. If ``pw_launch_kwargs``
        is detected, it will be added to the kwargs for all of the
        PLaywright-based search engines so that you do not have to
        repeatedly specify the launch parameters. For example, you may
        specify ``pw_launch_kwargs={"headless": False}`` to
        have all Playwright-based searches show the browser and _also_
        specify ``google_serper_api_kwargs={"api_key": "..."}`` to
        specify the API key for the Google Serper search.

    Returns
    -------
    list of :class:`elm.web.document.BaseDocument`
        List of documents representing the top `num_urls` results from
        the google searches across all `queries`.
    """
    if search_semaphore is None:
        # backward-compatibility
        search_semaphore = browser_semaphore

    fpq = use_fallback_per_query
    urls = await search_with_fallback(queries, search_engines=search_engines,
                                      num_urls=num_urls,
                                      ignore_url_parts=ignore_url_parts,
                                      browser_semaphore=search_semaphore,
                                      task_name=task_name,
                                      use_fallback_per_query=fpq, **kwargs)
    if on_search_complete_hook is not None:
        await on_search_complete_hook(urls)

    logger.debug("Downloading documents for URLS: \n\t-%s", "\n\t-".join(urls))
    docs = await load_docs(urls, browser_semaphore, **kwargs)
    return docs


async def search_with_fallback(queries, search_engines=_DEFAULT_SE,
                               num_urls=None, ignore_url_parts=None,
                               browser_semaphore=None, task_name=None,
                               use_fallback_per_query=True, **kwargs):
    """Retrieve search query URLs using multiple search engines if needed

    Parameters
    ----------
    queries : collection of str
        Collection of strings representing google queries. Documents for
        the top `num_urls` google search results (from all of these
        queries _combined_ will be returned from this function.
    search_engines : iterable of str
        Ordered collection of search engine names to attempt for web
        search. If the first search engine in the list returns a set
        of URLs, then iteration will end and documents for each URL will
        be returned. Otherwise, the next engine in this list will be
        used to run the web search. If this also fails, the next engine
        is used and so on. If all web searches fail, an empty list is
        returned. See :obj:`~elm.web.search.run.SEARCH_ENGINE_OPTIONS`
        for supported search engine options.
        By default, ``("PlaywrightGoogleLinkSearch", )``.
    num_urls : int, optional
        Number of unique top Google search result to return as docs. The
        google search results from all queries are interleaved and the
        top `num_urls` unique URL's are downloaded as docs. If this
        number is less than ``len(queries)``, some of your queries may
        not contribute to the final output. By default, ``None``, which
        sets ``num_urls = 3 * len(queries)``.
    ignore_url_parts : iterable of str, optional
        Optional URL components to blacklist. For example, supplying
        `ignore_url_parts={"wikipedia.org"}` will ignore all URLs that
        contain "wikipedia.org". By default, ``None``.
    browser_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    task_name : str, optional
        Optional task name to use in :func:`asyncio.create_task`.
        By default, ``None``.
    use_fallback_per_query : bool, default=True
        Option to use the fallback list of search engines on a per-query
        basis. This means if a single query fails with one search
        engine, the fallback search engines will be attempted for that
        query. If this input is ``False``, the fallback search engines
        are only used if *all* search queries fail for a single search
        engine. By default, ``True``.
    **kwargs
        Keyword-argument pairs to initialize search engines. This input
        can include and any/all of the following keywords:

            - ddg_api_kwargs
            - google_cse_api_kwargs
            - google_serper_api_kwargs
            - tavily_api_kwargs
            - ddgs_kwargs
            - cf_google_se_kwargs
            - pw_bing_se_kwargs
            - pw_ddg_se_kwargs
            - pw_google_cse_kwargs
            - pw_google_se_kwargs
            - pw_yahoo_se_kwargs
            - pw_launch_kwargs

        Each of these inputs should be a dictionary with
        keyword-argument pairs that you can use to initialize the search
        engines in the `search_engines` input. If ``pw_launch_kwargs``
        is detected, it will be added to the kwargs for all of the
        PLaywright-based search engines so that you do not have to
        repeatedly specify the launch parameters. For example, you may
        specify ``pw_launch_kwargs={"headless": False}`` to
        have all Playwright-based searches show the browser and _also_
        specify ``google_serper_api_kwargs={"api_key": "..."}`` to
        specify the API key for the Google Serper search.

    Returns
    -------
    set
        Set of URL's for the queries (empty if search failed).

    Raises
    ------
    ELMInputError
        If `search_engines` input is empty.
    """
    num_urls = num_urls or 3 * len(queries)
    if len(search_engines) < 1:
        msg = f"Must provide at least one search engine! Got {search_engines=}"
        logger.error(msg)
        raise ELMInputError(msg)

    if use_fallback_per_query:
        urls = await _multi_se_search(search_engines, queries, num_urls,
                                      ignore_url_parts, browser_semaphore,
                                      task_name, kwargs)
        if urls:
            return urls
    else:
        for se_name in search_engines:
            logger.debug("Searching web using %r", se_name)
            urls = await _single_se_search(se_name, queries, num_urls,
                                           ignore_url_parts, browser_semaphore,
                                           task_name, kwargs)
            if urls:
                return urls

    logger.warning("No web results found using %d search engines: %r",
                   len(search_engines), search_engines)
    return set()


async def load_docs(urls, browser_semaphore=None, **kwargs):
    """Load a document for each input URL

    Parameters
    ----------
    urls : iterable of str
        Iterable of URL's (as strings) to fetch.
    browser_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently for document retrieval. If
        ``None``, no limits are applied. By default, ``None``.
    kwargs
        Keyword-argument pairs to initialize
        :class:`elm.web.file_loader.AsyncFileLoader`.

    Returns
    -------
    list
        List of non-empty document instances containing information from
        the URL's. If a URL could not be fetched (i.e. document instance
        is empty), it will not be included in the output list.
    """
    logger.trace("Downloading docs for the following URL's:\n%r", urls)
    logger.trace("kwargs for AsyncFileLoader:\n%s",
                 pprint.PrettyPrinter().pformat(kwargs))
    file_loader = AsyncFileLoader(browser_semaphore=browser_semaphore,
                                  **kwargs)
    docs = await file_loader.fetch_all(*urls)

    page_lens = {doc.attrs.get("source", "Unknown"): len(doc.pages)
                 for doc in docs}
    logger.debug("Loaded the following number of pages for docs:\n%s",
                 pprint.PrettyPrinter().pformat(page_lens))
    return [doc for doc in docs if not doc.empty]


async def _single_se_search(se_name, queries, num_urls, ignore_url_parts,
                            browser_sem, task_name, kwargs):
    """Search for links using a single search engine"""
    _validate_se_name(se_name)
    links = await _run_search(se_name, queries, browser_sem, task_name, kwargs)
    return _down_select_urls(links, num_urls=num_urls,
                             ignore_url_parts=ignore_url_parts)


async def _multi_se_search(search_engines, queries, num_urls,
                           ignore_url_parts, browser_sem, task_name, kwargs):
    """Search for links using one or more search engines as fallback"""
    outputs = {q: None for q in queries}
    remaining_queries = list(queries)
    for se_name in search_engines:
        _validate_se_name(se_name)

        logger.debug("Searching web using %r", se_name)
        links = await _run_search(se_name, remaining_queries, browser_sem,
                                  task_name, kwargs)
        logger.trace("Links: %r", links)

        failed_queries = []
        for q, se_result in zip(remaining_queries, links):
            if not se_result or not se_result[0]:
                failed_queries.append(q)
                continue
            outputs[q] = se_result

        remaining_queries = failed_queries
        logger.trace("Remaining queries to search: %r", remaining_queries)

        if not remaining_queries:
            break

    links = [link or [[]] for link in outputs.values()]

    return _down_select_urls(links, num_urls=num_urls,
                             ignore_url_parts=ignore_url_parts)


async def _run_search(se_name, queries, browser_sem, task_name, kwargs):
    """Run a search for multiple queries on a single search engine"""
    searchers = [asyncio.create_task(_single_query_search(se_name, query,
                                                          browser_sem, kwargs),
                                     name=task_name) for query in queries]
    return await asyncio.gather(*searchers)


async def _single_query_search(se_name, query, browser_sem, kwargs):
    """Execute a single search query on a single search engine"""
    try:
        search_engine, uses_browser = _init_se(se_name, kwargs)
    except Exception as e:
        logger.error("Could not instantiate %s", se_name)
        logger.exception(e)
        return [[]]

    if uses_browser:
        # help avoid some detection by staggering the browser launches
        await asyncio.sleep(random.uniform(1, 10))
        return await _single_query_pw(search_engine, query,
                                      browser_sem=browser_sem)

    return await _single_query_api(search_engine, query)


async def _single_query_pw(search_engine, question, browser_sem):
    """Perform a single browser-based search"""
    if browser_sem is None:
        browser_sem = AsyncExitStack()

    logger.trace("Single search browser_semaphore=%r", browser_sem)
    async with browser_sem:
        logger.trace("Starting %s search for %r with browser_semaphore=%r",
                     search_engine._SE_NAME, question, browser_sem)
        return await search_engine.results(question,
                                           num_results=_RESULTS_PER_QUERY)


async def _single_query_api(search_engine, question):
    """Perform a single api-based search"""
    logger.trace("Starting %s search for %r", search_engine._SE_NAME, question)
    return await search_engine.results(question,
                                       num_results=_RESULTS_PER_QUERY)


def _init_se(se_name, kwargs):
    """Initialize a search engine class"""
    se_class, uses_browser, kwarg_key = SEARCH_ENGINE_OPTIONS[se_name]
    init_kwargs = {}
    if uses_browser:
        init_kwargs = kwargs.get("pw_launch_kwargs", {})

    # Cannot use 'pop' here because _run_search is invoked once per query.
    init_kwargs.update(kwargs.get(kwarg_key, {}))
    return se_class(**init_kwargs), uses_browser


def _down_select_urls(search_results, num_urls=5, ignore_url_parts=None):
    """Select the top N URLs"""
    ignore_url_parts = _as_set(ignore_url_parts)
    all_urls = chain.from_iterable(zip_longest(*[results[0]
                                                 for results
                                                 in search_results]))
    urls = set()
    for url in all_urls:
        if not url or any(substr in url for substr in ignore_url_parts):
            continue
        urls.add(url)
        if len(urls) == num_urls:
            break
    return urls


def _as_set(user_input):
    """Convert user input (possibly None or str) to set of strings"""
    if isinstance(user_input, str):
        user_input = {user_input}
    return set(user_input or [])


def _validate_se_name(se_name):
    """Validate user search engine name input"""
    if se_name not in SEARCH_ENGINE_OPTIONS:
        msg = (f"'se_name' must be one of: {list(SEARCH_ENGINE_OPTIONS)}\n"
               f"Got {se_name=}")
        logger.error(msg)
        raise ELMKeyError(msg)
