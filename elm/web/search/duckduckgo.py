# -*- coding: utf-8 -*-
"""ELM Web Scraping - DuckDuckGo search"""
import logging

from elm.web.search.base import PlaywrightSearchEngineLinkSearch


logger = logging.getLogger(__name__)


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
