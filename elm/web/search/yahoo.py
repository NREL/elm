# -*- coding: utf-8 -*-
"""ELM Web Scraping - Yahoo search"""
import random
import asyncio
import logging

from elm.web.search.base import PlaywrightSearchEngineLinkSearch


logger = logging.getLogger(__name__)


class PlaywrightYahooLinkSearch(PlaywrightSearchEngineLinkSearch):
    """Search for top links on the main Yahoo search engine"""

    MAX_RESULTS_CONSIDERED_PER_PAGE = 10
    """Number of results considered per Yahoo page.

    This value used to be 10, but the addition of extra divs like a set
    of youtube links has brought this number down."""

    _SE_NAME = "Yahoo"
    _SE_URL = "https://search.yahoo.com/"
    _SE_SR_TAG = '[referrerpolicy="origin"]'
    _SE_QUERY_URL = (
        "https://search.yahoo.com/search?p={}&fr=sfp&fr2=p%3As%2Cv%3Asfp"
    )

    async def _perform_homepage_search(self, page, search_query):
        """Fill in search bar with user query and hit enter"""
        await self._move_mouse(page)

        logger.trace("Clicking on search bar")
        search_bar = page.locator('[id="yschsp"]')
        await self._move_and_click(page, search_bar)
        await asyncio.sleep(random.uniform(0.5, 1.5))

        logger.trace("Typing in query: %r", search_query)
        await page.keyboard.type(search_query, delay=random.randint(80, 150))
        await asyncio.sleep(random.uniform(0.5, 1.5))

        logger.trace("Hitting enter for query: %r", search_query)
        await page.keyboard.press('Enter')
