# -*- coding: utf-8 -*-
"""ELM Web Scraping - DuxDistributedGlobalSearch"""
import logging

from ddgs import DDGS

from elm.web.search.base import SearchEngineLinkSearch


logger = logging.getLogger(__name__)


class DuxDistributedGlobalSearch(SearchEngineLinkSearch):
    """Search the web for links using DuxDistributedGlobalSearch"""

    _SE_NAME = "DuxDistributedGlobalSearch"

    def __init__(self, region="us-en", safesearch="moderate", timelimit=None,
                 page=1, backend=("google", "bing", "yahoo", "duckduckgo"),
                 timeout=10, verify=False):
        """

        Parameters
        ----------
        region : str, optional
            DuxDistributedGlobalSearch search region param.
            By default, ``"us-en"``.
        safesearch : {on, moderate, off}, optional
            The `safesearch` setting for search engines.
            By default, ``None``.
        timelimit : {d, w, m, y}, optional
            The time limit used to bound the search results:

                -d: last day
                -w: last week
                -m: last month
                -y: last year

            By default, ``None``.
        page : int, default=1
            The page of results to return. By default, ``1``.
        backend : str or iter of str, optional
            Option for DuxDistributedGlobalSearch backend:

                - auto: Randomly select 3 search engines to use
                - all: All available search engines are used
                - wikipedia: Wikipedia
                - google: Google
                - bing: Bing
                - brave: Brave
                - mojeek: Mojeek
                - yahoo: Yahoo
                - yandex: Yandex
                - duckduckgo: Duckduckgo

            Can also be a list or tuple of a combination of these.
            By default, ``("google", "bing", "yahoo", "duckduckgo")``.
        timeout : int, optional
            Timeout for HTTP requests, in seconds. By default, ``10``.
        verify : bool, optional
            Apply SSL verification when making the request.
            By default, ``False``.
        """
        self.region = region
        self.safesearch = safesearch
        self.timelimit = timelimit
        self.page = page
        self.backend = backend
        self.timeout = timeout
        self.verify = verify

    async def _search(self, query, num_results=10):
        """Search web for links related to a query"""

        ddgs = DDGS(timeout=self.timeout, verify=self.verify)
        results = ddgs.text(query, region=self.region,
                            safesearch=self.safesearch,
                            timelimit=self.timelimit,
                            page=self.page,
                            backend=self.backend,
                            num_results=num_results)

        return list(filter(None, (info.get('href', "").replace("+", "%20")
                                  for info in results)))

