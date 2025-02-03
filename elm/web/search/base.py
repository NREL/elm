# -*- coding: utf-8 -*-
"""ELM Web Scraping - Base class for search engine search"""
import asyncio
import logging
from abc import ABC, abstractmethod

from rebrowser_playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from elm.web.utilities import clean_search_query, pw_page


logger = logging.getLogger(__name__)


class PlaywrightSearchEngineLinkSearch(ABC):
    """Search for top results on a given search engine and return links"""

    MAX_RESULTS_PER_PAGE = 10
    """Number of results displayed per search engine page"""

    PAGE_LOAD_TIMEOUT = 90_000
    """Default page load timeout value in milliseconds"""

    _SE_NAME = "<unknown se>"

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
        """Search web for links related to a query"""
        logger.debug("Searching %s: %r", self._SE_NAME, query)
        num_results = min(num_results, self.MAX_RESULTS_PER_PAGE)

        async with pw_page(self._browser) as page:
            await _navigate_to_search_engine(page, se_url=self._SE_URL,
                                             timeout=self.PAGE_LOAD_TIMEOUT)
            logger.trace("Performing %s search for query: %r", self._SE_NAME,
                         query)
            await self._perform_search(page, query)
            logger.trace("Extracting links for query: %r", query)
            return await _extract_links(page, num_results,
                                        search_result_tag=self._SE_SR_TAG)

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
            logger.trace("Kicking off %s search for %d queries",
                         self._SE_NAME, len(searches))
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

    @property
    @abstractmethod
    def _SE_URL(self):
        """str: Search engine URL"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _SE_SR_TAG(self):
        """str: Search engine search results tag"""
        raise NotImplementedError

    @abstractmethod
    async def _perform_search(self, page, search_query):
        """Search query using search engine"""
        raise NotImplementedError


async def _navigate_to_search_engine(page, se_url, timeout=90_000):
    """Navigate to search engine domain"""
    await page.goto(se_url)
    logger.trace("Waiting for load")
    await page.wait_for_load_state("networkidle", timeout=timeout)


async def _extract_links(page, num_results, search_result_tag):
    """Extract links for top `num_results` on page"""
    links = await asyncio.to_thread(page.locator, search_result_tag)
    return [
        await links.nth(i).get_attribute("href") for i in range(num_results)
    ]
