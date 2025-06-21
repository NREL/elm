# -*- coding: utf-8 -*-
"""ELM Web Scraping - Bing search"""
import random
import asyncio
import logging

from elm.web.search.base import PlaywrightSearchEngineLinkSearch


logger = logging.getLogger(__name__)


class PlaywrightBingLinkSearch(PlaywrightSearchEngineLinkSearch):
    """Search for top links on the main Bing search engine"""

    MAX_RESULTS_CONSIDERED_PER_PAGE = 3
    """Number of results considered per Bing page.

    This value used to be 10, but the addition of extra divs like a set
    of youtube links has brought this number down."""

    _SE_NAME = "Bing"
    _SE_URL = "https://www.bing.com/"
    _SE_SR_TAG = '[redirecturl]'
    _SE_QUERY_URL = "https://www.bing.com/search?q={}&FORM=QBLH"

    async def _perform_homepage_search(self, page, search_query):
        """Fill in search bar with user query and hit enter"""
        await self._move_mouse(page)

        logger.trace("Finding search bar for query: %r", search_query)
        search_bar = page.locator('[id="sb_form_q"]')
        await self._move_and_click(page, search_bar)
        await asyncio.sleep(random.uniform(0.5, 1.5))

        logger.trace("Typing in query: %r", search_query)
        await page.keyboard.type(search_query, delay=random.randint(80, 150))
        await asyncio.sleep(random.uniform(0.5, 1.5))

        logger.trace("Hitting enter for query: %r", search_query)
        await page.keyboard.press('Enter')
