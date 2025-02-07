# -*- coding: utf-8 -*-
"""ELM Web Scraping - Yahoo search"""
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

    async def _perform_search(self, page, search_query):
        """Fill in search bar with user query and hit enter"""
        logger.trace("Finding search bar for query: %r", search_query)
        await page.locator('[id="yschsp"]').fill(search_query)
        logger.trace("Hitting enter for query: %r", search_query)
        await page.keyboard.press('Enter')
