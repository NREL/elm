# -*- coding: utf-8 -*-
"""ELM Web Scraping - DuckDuckGo search"""
import random
import asyncio
import logging

from duckduckgo_search import DDGS

from elm.web.search.base import (PlaywrightSearchEngineLinkSearch,
                                 SearchEngineLinkSearch)


logger = logging.getLogger(__name__)


_DDGS_SEMAPHORE = asyncio.Semaphore(1)


class PlaywrightDuckDuckGoLinkSearch(PlaywrightSearchEngineLinkSearch):
    """Search for top links on the main DuckDuckGo search engine"""

    MAX_RESULTS_CONSIDERED_PER_PAGE = 10
    """Number of results displayed per DuckDuckGo page"""

    _SE_NAME = "DuckDuckGo"
    _SE_URL = "https://duckduckgo.com/"
    _SE_SR_TAG = '[data-testid="result-extras-url-link"]'

    async def _perform_search(self, page, search_query):
        """Fill in search bar with user query and hit enter"""
        logger.trace("Finding search bar for query: %r", search_query)
        await (page
               .get_by_label("Search with DuckDuckGo", exact=True)
               .fill(search_query))
        logger.trace("Hitting enter for query: %r", search_query)
        await page.keyboard.press('Enter')


class APIDuckDuckGoSearch(SearchEngineLinkSearch):
    """Search the web for links using the DuckDuckGo API"""

    _SE_NAME = "DuckDuckGo API"

    def __init__(self, region="wt-wt", backend="auto", timeout=10,
                 verify=True, sleep_min_seconds=10, sleep_max_seconds=20):
        """

        Parameters
        ----------
        region : str, optional
            DDG search region param. By default, ``"wt-wt"``, which
            signifies no region.
        backend : {auto, html, lite}, optional
            Option for DDG search type.

                - auto: select randomly between HTML and Lite backends
                - html: collect data from https://html.duckduckgo.com
                - lite: collect data from https://lite.duckduckgo.com

            By default, ``"auto"``.
        timeout : int, optional
            Timeout for HTTP requests, in seconds. By default, ``10``.
        verify : bool, optional
            Apply SSL verification when making the request.
            By default, ``True``.
        sleep_min_seconds : int, optional
            Minimum number of seconds to sleep between queries. We
            recommend not setting this below ``5`` seconds to avoid
            rate limiting errors thrown by DuckDuckGo.
            By default, ``10``.
        sleep_max_seconds : int, optional
            Maximum number of seconds to sleep between queries.
            By default, ``20``.
        """
        self.region = region
        self.backend = backend
        self.timeout = timeout
        self.verify = verify
        self.sleep_min_seconds = sleep_min_seconds
        self.sleep_max_seconds = sleep_max_seconds

    async def _search(self, query, num_results=10):
        """Search web for links related to a query"""

        ddgs = DDGS(timeout=self.timeout, verify=self.verify)
        results = ddgs.text(query, region=self.region,
                            backend=self.backend,
                            max_results=num_results)

        return list(filter(None, (info.get('href', "").replace("+", "%20")
                                  for info in results)))

    async def _skip_exc_search(self, query, num_results=10):
        """Sleep between DDG searched to avoid rate limiting"""
        async with _DDGS_SEMAPHORE:
            try:
                out = await self._search(query, num_results=num_results)
            except Exception as e:
                logger.exception(e)
                out = []

            await self._sleep_after_query(query)

        return out

    async def _sleep_after_query(self, query):
        """Sleep for a random time after a query"""
        delay = random.uniform(self.sleep_min_seconds, self.sleep_max_seconds)
        logger.debug("DDG search sleeping for %.2f seconds after query: %s",
                     delay, query)
        await asyncio.sleep(delay)
