# -*- coding: utf-8 -*-
"""ELM Web Scraping - Bing search"""
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

    async def _perform_search(self, page, search_query):
        """Fill in search bar with user query and hit enter"""
        logger.trace("Finding search bar for query: %r", search_query)
        await page.locator('[id="sb_form_q"]').fill(search_query)
        logger.trace("Hitting enter for query: %r", search_query)
        await page.keyboard.press('Enter')
