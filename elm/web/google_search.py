# -*- coding: utf-8 -*-
"""ELM Web Scraping - Google search."""
import pprint
import asyncio
import logging
from itertools import zip_longest, chain
from contextlib import AsyncExitStack

from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from elm.web.document import PDFDocument
from elm.web.file_loader import AsyncFileLoader
from elm.web.utilities import clean_search_query


logger = logging.getLogger(__name__)
_SEARCH_RESULT_TAG = '[jsname="UWckNb"]'


class PlaywrightGoogleLinkSearch:
    """Search for top results on google and return their links

    Purpose:
        Search Google using Playwright engine.
    Responsibilities:
        1. Launch browser using Playwright and navigate to Google.
        2. Submit a query to Google Search.
        3. Get list of resulting URLs.
    Key Relationships:
        Relies on `Playwright <https://playwright.dev/python/>`_ for
        web access.

    .. end desc
    """

    EXPECTED_RESULTS_PER_PAGE = 10
    """Number of results displayed per Google page. """

    PAGE_LOAD_TIMEOUT = 90_000
    """Default page load timeout value in milliseconds"""

    def __init__(self, **launch_kwargs):
        """

        Parameters
        ----------
        **launch_kwargs
            Keyword arguments to be passed to
            `playwright.chromium.launch`. For example, you can pass
            ``headless=False, slow_mo=50`` for a visualization of the
            search.
        """
        self.launch_kwargs = launch_kwargs
        self._browser = None

    async def _load_browser(self, pw_instance):
        """Launch a chromium instance and load a page"""
        self._browser = await pw_instance.chromium.launch(**self.launch_kwargs)

    async def _close_browser(self):
        """Close browser instance and reset internal attributes"""
        logger.trace("Closing browser...")
        await self._browser.close()
        self._browser = None

    async def _search(self, query, num_results=10):
        """Search google for links related to a query."""
        logger.debug("Searching Google: %r", query)
        num_results = min(num_results, self.EXPECTED_RESULTS_PER_PAGE)

        logger.trace("Loading browser page for query: %r", query)
        page = await self._browser.new_page()
        logger.trace("Navigating to google for query: %r", query)
        await _navigate_to_google(page, timeout=self.PAGE_LOAD_TIMEOUT)
        logger.trace("Performing google search for query: %r", query)
        await _perform_google_search(page, query)
        logger.trace("Extracting links for query: %r", query)
        return await _extract_links(page, num_results)

    async def _skip_exc_search(self, query, num_results=10):
        """Perform search while ignoring timeout errors"""
        try:
            return await self._search(query, num_results=num_results)
        except PlaywrightTimeoutError as e:
            logger.exception(e)
            return []

    async def _get_links(self, queries, num_results):
        """Get links for multiple queries"""
        outer_task_name = asyncio.current_task().get_name()
        async with async_playwright() as pw_instance:
            await self._load_browser(pw_instance)
            searches = [
                asyncio.create_task(
                    self._skip_exc_search(query, num_results=num_results),
                    name=outer_task_name,
                )
                for query in queries
            ]
            logger.trace("Kicking off search for %d queries", len(searches))
            results = await asyncio.gather(*searches)
            logger.trace("Got results for link search:\n%r", results)
            await self._close_browser()
        return results

    async def results(self, *queries, num_results=10):
        """Retrieve links for the first `num_results` of each query.

        This function executes a google search for each input query and
        returns a list of links corresponding to the top `num_results`.

        Parameters
        ----------
        num_results : int, optional
            Number of top results to retrieve for each query. Note that
            this value can never exceed the number of results per page
            (typically 10). If you pass in a larger value, it will be
            reduced to the number of results per page.
            By default, ``10``.

        Returns
        -------
        list
            List equal to the length of the input queries, where each
            entry is another list containing the top `num_results`
            links.
        """
        queries = map(clean_search_query, queries)
        return await self._get_links(queries, num_results)


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


async def filter_documents(
    documents, validation_coroutine, task_name=None, **kwargs
):
    """Filter documents by applying a filter function to each.

    Parameters
    ----------
    documents : iter of :class:`elm.web.document.BaseDocument`
        Iterable of documents to filter.
    validation_coroutine : coroutine
        A coroutine that returns ``False`` if the document should be
        discarded and ``True`` otherwise. This function should take a
        single :class:`elm.web.document.BaseDocument` instance as the
        first argument. The function may have other arguments, which
        will be passed down using `**kwargs`.
    task_name : str, optional
        Optional task name to use in :func:`asyncio.create_task`.
        By default, ``None``.
    **kwargs
        Keyword-argument pairs to pass to `validation_coroutine`. This
        should not include the document instance itself, which will be
        independently passed in as the first argument.

    Returns
    -------
    list of :class:`elm.web.document.BaseDocument`
        List of documents that passed the validation check, sorted by
        text length, with PDF documents taking the highest precedence.
    """
    searchers = [
        asyncio.create_task(
            validation_coroutine(doc, **kwargs), name=task_name
        )
        for doc in documents
    ]
    output = await asyncio.gather(*searchers)
    filtered_docs = [doc for doc, check in zip(documents, output) if check]
    return sorted(
        filtered_docs,
        key=lambda doc: (not isinstance(doc, PDFDocument), len(doc.text)),
    )


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


async def _navigate_to_google(page, timeout=90_000):
    """Navigate to Google domain."""
    await page.goto("https://www.google.com")
    logger.trace("Waiting for google to load")
    await page.wait_for_load_state("networkidle", timeout=timeout)


async def _perform_google_search(page, search_query):
    """Fill in search bar with user query and click search button"""
    logger.trace("Finding search bar for query: %r", search_query)
    await page.get_by_label("Search", exact=True).fill(search_query)
    logger.trace("Closing autofill for query: %r", search_query)
    await _close_autofill_suggestions(page)
    logger.trace("Hitting search button for query: %r", search_query)
    await page.get_by_role("button", name="Google Search").click()


async def _close_autofill_suggestions(page):
    """Google autofill suggestions often get in way of search button.

    We get around this by closing the suggestion dropdown before
    looking for the search button. Looking for the "Google Search"
    button doesn't work because it is sometimes obscured by the dropdown
    menu. Clicking the "Google" logo can also fail when they add
    seasonal links/images (e.g. holiday logos). Current solutions is to
    look for a specific div at the top of the page.
    """
    await page.locator("#gb").click()


async def _extract_links(page, num_results):
    """Extract links for top `num_results` on page"""
    links = await asyncio.to_thread(page.locator, _SEARCH_RESULT_TAG)
    return [
        await links.nth(i).get_attribute("href") for i in range(num_results)
    ]
