# -*- coding: utf-8 -*-
"""ELM Web Scraping - Google search."""
import os
import json
import random
import asyncio
import logging
import requests
from contextlib import asynccontextmanager

from camoufox.async_api import AsyncCamoufox
from googleapiclient.discovery import build
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from elm.web.search.base import (PlaywrightSearchEngineLinkSearch,
                                 APISearchEngineLinkSearch)


logger = logging.getLogger(__name__)


class PlaywrightGoogleLinkSearch(PlaywrightSearchEngineLinkSearch):
    """Search for top links on the main google search engine

    Purpose:
        Search Google using Playwright engine.
    Responsibilities:
        1. Launch browser using Playwright and
        navigate to Google.
        2. Submit a query to Google Search.
        3. Get list of resulting URLs.
    Key Relationships:
        Relies on `Playwright <https://playwright.dev/python/>`_ for
        web access.

    .. end desc
    """
    MAX_RESULTS_CONSIDERED_PER_PAGE = 10
    """Number of results considered per Google page.

    This value used to be 10, but the addition of extra divs like a set
    of youtube links has brought this number down.
    """

    _SE_NAME = "Google"
    _SE_URL = "https://www.google.com"
    _SE_SR_TAG = '[jsname="UWckNb"]'
    _SC = None
    _SE_QUERY_URL = (
        "https://www.google.com/search?q={}&gl=sg&hl=en&udm=14&start=0&num=10"
    )

    async def _perform_homepage_search(self, page, search_query):
        """Fill in search bar with user query and hit enter"""
        await self._move_mouse(page)

        logger.trace("Clicking on search bar for query: %r", search_query)
        await self._click_on_search_bar(page)

        logger.trace("Filling in search bar for query: %r", search_query)
        await self._fill_in_search_bar(page, search_query)

        logger.trace("Hitting enter for query: %r", search_query)
        await page.keyboard.press('Enter')

    async def _click_on_search_bar(self, page):
        """Find the search bar and click it"""
        try:
            search_bar = page.get_by_label("Search", exact=True)
            return await self._move_and_click(page, search_bar)
        except PlaywrightTimeoutError:
            pass

        search_bar = page.locator('[name="q"]')
        try:
            return self._move_and_click(page, search_bar)
        except PlaywrightTimeoutError:
            pass

        search_bar = page.locator('[autofocus]')
        return self._move_and_click(page, search_bar)

    async def _fill_in_search_bar(self, page, search_query):
        """Attempt to find and fill the search bar several ways"""
        await asyncio.sleep(random.uniform(0.5, 1.5))

        logger.trace("Typing in query: %r", search_query)
        await page.keyboard.type(search_query, delay=random.randint(80, 150))
        return await asyncio.sleep(random.uniform(0.5, 1.5))


class CamoufoxGoogleLinkSearch(PlaywrightGoogleLinkSearch):
    """Search for top links on Google using Camoufox browser"""

    def __init__(self, use_homepage=True, **launch_kwargs):
        """

        Parameters
        ----------
        use_homepage : bool, default=True
            If ``True``, the browser will be navigated to the search
            engine homepage and the query will be input into the search
            bar. If ``False``, the query will be embedded in the URL
            and the browser will navigate directly to the filled-out
            URL. By default, ``False``.
        **launch_kwargs
            Keyword arguments to be passed to
            `playwright.firefox.launch`. For example, you can pass
            ``headless=False, slow_mo=50`` for a visualization of the
            search.
        """
        self.use_homepage = use_homepage
        self.launch_kwargs = {"humanize": 0.1, "headless": True}
        self.launch_kwargs.update(launch_kwargs)
        self._browser = None

    async def _load_browser(self, pw_instance):
        """Empty implementation since we are using camoufox"""

    @asynccontextmanager
    async def _browser_page(self):
        """Get page to use for search"""
        async with AsyncCamoufox(**self.launch_kwargs) as browser:
            yield await browser.new_page()


class PlaywrightGoogleCSELinkSearch(PlaywrightSearchEngineLinkSearch):
    """Search for top links on a custom google search engine

    Resources
    ---------
    https://programmablesearchengine.google.com/controlpanel/create
    """

    _SE_NAME = "Google CSE"
    _SE_SR_TAG = "a.gs-title[href]"
    PAGE_LOAD_TIMEOUT = 10_000
    _SE_QUERY_URL = None

    def __init__(self, cse_url, use_homepage=False, **launch_kwargs):
        """

        Parameters
        ----------
        cse_url : str
            URL of the custom google programmable search engine.
        use_homepage : bool, default=False
            If ``True``, the browser will be navigated to the search
            engine homepage and the query will be input into the search
            bar. If ``False``, the query will be embedded in the URL
            and the browser will navigate directly to the filled-out
            URL. By default, ``False``.
        **launch_kwargs
            Keyword arguments to be passed to
            `playwright.chromium.launch`. For example, you can pass
            ``headless=False, slow_mo=50`` for a visualization of the
            search.
        """
        super().__init__(use_homepage=use_homepage, **launch_kwargs)
        self._cse_url = cse_url
        self._SE_QUERY_URL = (
            f"https://cse.google.com/cse?cx={cse_url}#gsc.tab=0"
            "&gsc.q={}&gsc.sort="
        )

    @property
    def _SE_URL(self):
        """str: URL for the custom google programmable search engine"""
        if not self._cse_url.endswith("#gsc.tab=0"):
            self._cse_url = f"{self._cse_url}#gsc.tab=0"
        return self._cse_url

    async def _perform_homepage_search(self, page, search_query):
        """Fill in search bar with user query and hit enter"""
        logger.trace("Finding search bar for query: %r", search_query)
        await page.get_by_label("search", exact=True).fill(search_query)
        logger.trace("Hitting enter for query: %r", search_query)
        await page.keyboard.press('Enter')

    async def _extract_links(self, page, num_results, query):
        """Extract links for top `num_results` on page"""
        await page.wait_for_load_state("networkidle",
                                       timeout=self.PAGE_LOAD_TIMEOUT)
        await page.wait_for_selector(self._SE_SR_TAG)
        locator = page.locator(self._SE_SR_TAG)

        count = await locator.count() // 2
        links = []

        for i in range(count):
            element = locator.nth(i * 2)
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


class APIGoogleCSESearch(APISearchEngineLinkSearch):
    """Search the web for links using a Google CSE API"""

    _BUILD_ARGS = {"serviceName": "customsearch", "version": "v1"}
    _SE_NAME = "Google CSE API"

    API_KEY_VAR = "GOOGLE_API_KEY"
    """Environment variable that should contain the Google CSE API key"""
    CSE_ID_VAR = "GOOGLE_CSE_ID"
    """Environment variable that should contain CSE ID"""

    def __init__(self, api_key=None, cse_id=None):
        """

        Parameters
        ----------
        api_key : str, optional
            API key for search engine. If ``None``, will look up the API
            key using the :obj:`API_KEY_VAR` environment variable.
            By default, ``None``.
        """
        super().__init__(api_key=api_key)
        self.cse_id = cse_id or os.environ.get(self.CSE_ID_VAR or "")

    async def _search(self, query, num_results=10):
        """Search web for links related to a query"""
        build_args = dict(self._BUILD_ARGS)
        build_args["developerKey"] = self.api_key

        search_args = {"q": query, "cx": self.cse_id, "num": num_results}

        results = build(**build_args).cse().list(**search_args).execute()
        results = (results or {}).get('items', [])
        return list(filter(None, (info.get("link") for info in results)))


class APISerperSearch(APISearchEngineLinkSearch):
    """Search the web for links using the Google Serper API"""

    _SE_NAME = "Google Serper API"
    _URL = "https://google.serper.dev/search"

    API_KEY_VAR = "SERPER_API_KEY"
    """Environment variable that should contain the Google Serper API key"""

    def __init__(self, api_key=None, verify=False):
        """

        Parameters
        ----------
        api_key : str, optional
            API key for serper search API. If ``None``, will look up the
            API key using the ``"SERPER_API_KEY"`` environment variable.
            By default, ``None``.
        verify : bool, default=False
            Option to use SSL verification when making request to API
            endpoint. By default, ``False``.
        """
        super().__init__(api_key=api_key)
        self.verify = verify

    async def _search(self, query, num_results=10):
        """Search web for links related to a query"""

        payload = json.dumps({"q": query, "num": num_results})
        headers = {'X-API-KEY': self.api_key,
                   'Content-Type': 'application/json'}

        response = requests.request("POST", self._URL, headers=headers,
                                    data=payload, verify=self.verify)
        results = json.loads(response.text).get('organic', {})
        return list(filter(None, (result.get("link", "").replace("+", "%20")
                                  for result in results)))
