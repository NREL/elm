# -*- coding: utf-8 -*-
"""ELM Web Scraping - Tavily API search"""
import logging

from tavily import TavilyClient

from elm.web.search.base import APISearchEngineLinkSearch


logger = logging.getLogger(__name__)


class APITavilySearch(APISearchEngineLinkSearch):
    """Search the web for links using the Tavily API"""

    _SE_NAME = "Tavily API"

    API_KEY_VAR = "TAVILY_API_KEY"
    """Environment variable that should contain the Tavily API key"""

    async def _search(self, query, num_results=10):
        """Search web for links related to a query"""

        client = TavilyClient(api_key=self.api_key)
        response = client.search(query=query, max_results=num_results)
        results = response.get("results", [])
        return list(filter(None, (info.get('url', "").replace("+", "%20")
                                  for info in results)))
