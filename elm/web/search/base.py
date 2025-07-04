# -*- coding: utf-8 -*-
"""ELM Web Scraping - Base class for search engine search"""
import os
import random
import asyncio
import logging
from urllib.parse import quote
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

from rebrowser_playwright.async_api import async_playwright
from playwright_stealth import StealthConfig

from elm.web.utilities import PWKwargs, clean_search_query, pw_page


logger = logging.getLogger(__name__)


class SearchEngineLinkSearch(ABC):
    """Abstract class to retrieve links for a query using a search engine"""

    _SE_NAME = "<unknown se>"

    async def results(self, *queries, num_results=10):
        """Retrieve links for the first `num_results` of each query

        This function executes a search for each input query and
        returns a list of links corresponding to the top `num_results`.

        Parameters
        ----------
        *queries : str
            One or more queries to search for.
        num_results : int, optional
            Maximum number of top results to retrieve for each query.
            Note that this value can never exceed the number of results
            per page (typically 10). If you pass in a larger value, it
            will be reduced to the number of results per page. There is
            also no guarantee that the search query will return this
            many results - the actual number of results returned is
            determined by the number of results on a page (excluding
            ads). You can, however, use this input to limit the number
            of results returned. By default, ``10``.

        Returns
        -------
        list
            List equal to the length of the input queries, where each
            entry is another list containing no more than `num_results`
            links.
        """
        queries = map(clean_search_query, queries)
        return await self._get_links(queries, num_results)

    async def _get_links(self, queries, num_results):
        """Get links for multiple queries"""
        outer_task_name = asyncio.current_task().get_name()

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
        return results

    async def _skip_exc_search(self, query, num_results=10):
        """Perform search while ignoring errors"""
        try:
            return await self._search(query, num_results=num_results)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = "Could not complete search for query=%r\nGot error type: %r"
            logger.exception(msg, query, type(e))
            return []

    async def _move_mouse(self, page):
        """Simulate mouse movement"""
        logger.trace("Moving mouse")
        vp = page.viewport_size or {}
        logger.trace("VP: %r", vp)
        width, height = vp.get('width', 300), vp.get('height', 300)
        min_width, max_width = int(width * 0.1), int(width * 0.9)
        min_height, max_height = int(height * 0.1), int(height * 0.9)
        logger.trace("Moving mouse to random position within: %d-%d, %d-%d",
                     min_width, max_width, min_height, max_height)

        await page.mouse.move(random.randint(min_width, max_width),
                              random.randint(min_height, max_height),
                              steps=10)
        await asyncio.sleep(random.uniform(1.5, 3.5))

    async def _move_and_click(self, page, input_el):
        """Move mouse to an element and click on it"""
        box = await input_el.bounding_box()
        if box is None:
            return await input_el.click()

        x = box["x"] + int(box["width"] / random.uniform(1.1, 10))
        y = box["y"] + int(box["height"] / random.uniform(1.1, 10))

        await page.mouse.move(x, y, steps=random.randint(5, 30))
        await asyncio.sleep(random.uniform(0.2, 0.6))
        return await page.mouse.click(x, y)

    @abstractmethod
    async def _search(self, query, num_results=10):
        """Search web for links related to a query"""
        raise NotImplementedError


class PlaywrightSearchEngineLinkSearch(SearchEngineLinkSearch):
    """Abstract class to search the web using playwright and a search engine"""

    MAX_RESULTS_CONSIDERED_PER_PAGE = 10
    """Number of results considered per search engine page"""

    PAGE_LOAD_TIMEOUT = 60_000
    """Default page load timeout value in milliseconds"""

    _SC = StealthConfig(navigator_user_agent=False)

    def __init__(self, use_homepage=True, use_scrapling_stealth=False,
                 **launch_kwargs):
        """

        Parameters
        ----------
        use_homepage : bool, default=True
            If ``True``, the browser will be navigated to the search
            engine homepage and the query will be input into the search
            bar. If ``False``, the query will be embedded in the URL
            and the browser will navigate directly to the filled-out
            URL. By default, ``False``.
        use_scrapling_stealth : bool, default=False
            Option to use scrapling stealth scripts instead of
            tf-playwright-stealth. If set to ``True``, the `_SC` class
            attribute will be ignored. By default, ``False``.
        **launch_kwargs
            Keyword arguments to be passed to
            `playwright.chromium.launch`. For example, you can pass
            ``headless=False, slow_mo=50`` for a visualization of the
            search.
        """
        self.use_homepage = use_homepage
        self.use_scrapling_stealth = use_scrapling_stealth
        self.launch_kwargs = PWKwargs.launch_kwargs()
        self.launch_kwargs.update(launch_kwargs)
        self._browser = None

    async def _load_browser(self, pw_instance):
        """Launch a chromium instance and load a page"""
        self._browser = await pw_instance.chromium.launch(**self.launch_kwargs)

    async def _close_browser(self):
        """Close browser instance and reset internal attributes"""
        logger.trace("Closing browser...")
        if self._browser is None:
            return

        await self._browser.close()
        self._browser = None

    @asynccontextmanager
    async def _browser_page(self):
        """Get page to use for search"""
        page_kwargs = {"browser": self._browser, "stealth_config": self._SC,
                       "ignore_https_errors": True,  # no sensitive inputs
                       "timeout": self.PAGE_LOAD_TIMEOUT,
                       "use_scrapling_stealth": self.use_scrapling_stealth}

        async with pw_page(**page_kwargs) as page:
            yield page

    async def _search(self, query, num_results=10):
        """Search web for links related to a query"""
        logger.debug("Searching %s: %r", self._SE_NAME, query)
        num_results = min(num_results, self.MAX_RESULTS_CONSIDERED_PER_PAGE)

        async with self._browser_page() as page:
            if self.use_homepage:
                logger.trace("Navigating to %s homepage", self._SE_NAME)
                await _navigate_to_se_url(page, se_url=self._SE_URL,
                                          timeout=self.PAGE_LOAD_TIMEOUT)
                logger.trace("Performing %s search for query: %r",
                             self._SE_NAME, query)
                await self._perform_homepage_search(page, query)
            else:
                url = self._SE_QUERY_URL.format(quote(query))
                logger.trace("Submitting URL: %r", url)
                await _navigate_to_se_url(page, se_url=url,
                                          timeout=self.PAGE_LOAD_TIMEOUT)
            logger.trace("Extracting links for query: %r", query)
            return await self._extract_links(page, num_results, query)

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

    async def _extract_links(self, page, num_results, query):
        """Extract links for top `num_results` on page"""
        await page.wait_for_load_state("networkidle",
                                       timeout=self.PAGE_LOAD_TIMEOUT)
        await page.wait_for_selector(self._SE_SR_TAG)
        locator = page.locator(self._SE_SR_TAG)
        count = await locator.count()
        links = []

        for i in range(count):
            element = locator.nth(i)
            try:
                link = await element.get_attribute("href")
                if link is not None:
                    links.append(link)
            except Exception:
                logger.exception("Skipped extracting link %d for query %r",
                                 i, query)

            if len(links) >= num_results:
                break

        return links

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

    @property
    @abstractmethod
    def _SE_QUERY_URL(self):
        """str: Search engine query URL template"""
        raise NotImplementedError

    @abstractmethod
    async def _perform_homepage_search(self, page, search_query):
        """Search query using search engine homepage"""
        raise NotImplementedError


class APISearchEngineLinkSearch(SearchEngineLinkSearch):
    """Abstract class to search the web using a search engine API"""

    API_KEY_VAR = None
    """Name of environment variable that should contain the API key"""

    def __init__(self, api_key=None):
        """

        Parameters
        ----------
        api_key : str, optional
            API key for search engine. If ``None``, will look up the API
            key using the :obj:`API_KEY_VAR` environment variable.
            By default, ``None``.
        """
        self.api_key = api_key or os.environ.get(self.API_KEY_VAR or "")


async def _navigate_to_se_url(page, se_url, timeout=90_000):
    """Navigate to search engine url"""
    await page.goto(se_url)
    logger.trace("Waiting for load")
    await page.wait_for_load_state("networkidle", timeout=timeout)
