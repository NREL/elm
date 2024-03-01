# -*- coding: utf-8 -*-
"""ELM Web Scraping - Google search."""
import asyncio
import logging

from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from elm.web.utilities import clean_search_query


logger = logging.getLogger(__name__)
_SEARCH_RESULT_TAG = '[jsname="UWckNb"]'


class PlaywrightGoogleLinkSearch:
    """Search for top results on google and return their links"""

    EXPECTED_RESULTS_PER_PAGE = 10
    """Number of results displayed per Google page. """

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
        await self._browser.close()
        self._browser = None

    async def _search(self, query, num_results=10):
        """Search google for links related to a query."""
        logger.debug("Searching Google: %r", query)
        num_results = min(num_results, self.EXPECTED_RESULTS_PER_PAGE)

        page = await self._browser.new_page()
        await _navigate_to_google(page)
        await _perform_google_search(page, query)
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
            results = await asyncio.gather(*searches)
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


async def _navigate_to_google(page):
    """Navigate to Google domain."""
    await page.goto("https://www.google.com")
    await page.wait_for_load_state("networkidle")


async def _perform_google_search(page, search_query):
    """Fill in search bar with user query and click search button"""
    await page.get_by_label("Search", exact=True).fill(search_query)
    await _close_autofill_suggestions(page)
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
