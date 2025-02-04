# -*- coding: utf-8 -*-
"""ELM Web Scraping - Google search."""
import pprint
import asyncio
import logging
from itertools import zip_longest, chain
from contextlib import AsyncExitStack

from elm.web.file_loader import AsyncFileLoader
from elm.web.search.base import PlaywrightSearchEngineLinkSearch


logger = logging.getLogger(__name__)


class PlaywrightGoogleLinkSearch(PlaywrightSearchEngineLinkSearch):
    """Search for top links on the main google search engine

    Purpose:
        Search Google using Playwright engine.
    Responsibilities:
        1. Launch browser using Playwright and
        navigate to Google.
        2. Submit a query to Google Search.
        3. Get list of resulting URLs.
    Key Relationships:
        Relies on `Playwright <https://playwright.dev/python/>`_ for
        web access.

    .. end desc
    """
    MAX_RESULTS_PER_PAGE = 5
    """Number of results displayed per Google page.

    This value used to be 10, but the addition of extra divs like a set
    of youtube links has brought this number down.
    """

    _SE_NAME = "Google"
    _SE_URL = "https://www.google.com"
    _SE_SR_TAG = '[jsname="UWckNb"]'

    async def _perform_search(self, page, search_query):
        """Fill in search bar with user query and hit enter"""
        logger.trace("Finding search bar for query: %r", search_query)
        await page.get_by_label("Search", exact=True).fill(search_query)
        logger.trace("Hitting enter for query: %r", search_query)
        await page.keyboard.press('Enter')


async def google_results_as_docs(
    queries,
    num_urls=None,
    browser_semaphore=None,
    task_name=None,
    **file_loader_kwargs,
):
    """Retrieve top ``N`` google search results as document instances.

    Parameters
    ----------
    queries : collection of str
        Collection of strings representing google queries. Documents for
        the top `num_urls` google search results (from all of these
        queries _combined_ will be returned from this function.
    num_urls : int, optional
        Number of unique top Google search result to return as docs. The
        google search results from all queries are interleaved and the
        top `num_urls` unique URL's are downloaded as docs. If this
        number is less than ``len(queries)``, some of your queries may
        not contribute to the final output. By default, ``None``, which
        sets ``num_urls = 3 * len(queries)``.
    browser_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    task_name : str, optional
        Optional task name to use in :func:`asyncio.create_task`.
        By default, ``None``.
    **file_loader_kwargs
        Keyword-argument pairs to initialize
        :class:`elm.web.file_loader.AsyncFileLoader` with. If found, the
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.google_search.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.

    Returns
    -------
    list of :class:`elm.web.document.BaseDocument`
        List of documents representing the top `num_urls` results from
        the google searches across all `queries`.
    """
    pw_launch_kwargs = file_loader_kwargs.get("pw_launch_kwargs", {})
    urls = await _find_urls(
        queries,
        num_results=10,
        browser_sem=browser_semaphore,
        task_name=task_name,
        **pw_launch_kwargs
    )
    num_urls = num_urls or 3 * len(queries)
    urls = _down_select_urls(urls, num_urls=num_urls)
    logger.debug("Downloading documents for URLS: \n\t-%s", "\n\t-".join(urls))
    docs = await _load_docs(urls, browser_semaphore, **file_loader_kwargs)
    return docs


async def _find_urls(
    queries, num_results=10, browser_sem=None, task_name=None, **kwargs
):
    """Parse google search output for URLs."""
    searchers = [
        asyncio.create_task(
            _search_single(
                query, browser_sem, num_results=num_results, **kwargs
            ),
            name=task_name,
        )
        for query in queries
    ]
    return await asyncio.gather(*searchers)


async def _search_single(question, browser_sem, num_results=10, **kwargs):
    """Perform a single google search."""
    if browser_sem is None:
        browser_sem = AsyncExitStack()

    search_engine = PlaywrightGoogleLinkSearch(**kwargs)
    logger.trace("Single search browser_semaphore=%r", browser_sem)
    async with browser_sem:
        logger.trace("Starting search for %r with browser_semaphore=%r",
                     question, browser_sem)
        return await search_engine.results(question, num_results=num_results)


def _down_select_urls(search_results, num_urls=5):
    """Select the top 5 URLs."""
    all_urls = chain.from_iterable(
        zip_longest(*[results[0] for results in search_results])
    )
    urls = set()
    for url in all_urls:
        if not url:
            continue
        urls.add(url)
        if len(urls) == num_urls:
            break
    return urls


async def _load_docs(urls, browser_semaphore=None, **kwargs):
    """Load a document for each input URL."""
    logger.trace("Downloading docs for the following URL's:\n%r", urls)
    logger.trace("kwargs for AsyncFileLoader:\n%s",
                 pprint.PrettyPrinter().pformat(kwargs))
    file_loader = AsyncFileLoader(
        browser_semaphore=browser_semaphore, **kwargs
    )
    docs = await file_loader.fetch_all(*urls)

    page_lens = {doc.metadata.get("source", "Unknown"): len(doc.pages)
                 for doc in docs}
    logger.debug("Loaded the following number of pages for docs:\n%s",
                 pprint.PrettyPrinter().pformat(page_lens))
    return [doc for doc in docs if not doc.empty]
