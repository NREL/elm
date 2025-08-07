# -*- coding: utf-8 -*-
"""ELM Web Scraping - Tavily API search"""
import json
import logging
import requests

from tavily import TavilyClient
from tavily.errors import (UsageLimitExceededError, InvalidAPIKeyError,
                           BadRequestError, ForbiddenError)

from elm.web.search.base import APISearchEngineLinkSearch


logger = logging.getLogger(__name__)


class _PatchedTavilyClient(TavilyClient):
    """Patch `TavilyClient` to accept verify keyword"""

    def __init__(self, api_key=None, proxies=None, verify=False):
        """

        Parameters
        ----------
        api_key : str, optional
            API key for search engine. If ``None``, will look up the API
            key using the ``"TAVILY_API_KEY"`` environment variable.
            By default, ``None``.
        verify : bool, default=False
            Option to use SSL verification when making request to API
            endpoint. By default, ``False``.
        """
        super().__init__(api_key=api_key, proxies=proxies)
        self.verify = verify

    def _search(self, query, search_depth="basic", topic="general",
                time_range=None, days=7, max_results=5, include_domains=None,
                exclude_domains=None, include_answer=False,
                include_raw_content=False, include_images=False, timeout=60,
                **kwargs):
        """Internal search method to send the request to the API"""

        data = {"query": query, "search_depth": search_depth, "topic": topic,
                "time_range": time_range, "days": days,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "max_results": max_results, "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "include_images": include_images}

        if kwargs:
            data.update(kwargs)

        timeout = min(timeout, 120)

        response = requests.post(self.base_url + "/search",
                                 data=json.dumps(data), headers=self.headers,
                                 timeout=timeout, proxies=self.proxies,
                                 verify=self.verify)

        if response.status_code == 200:
            return response.json()
        else:
            detail = ""
            try:
                detail = response.json().get("detail", {}).get("error", None)
            except Exception:
                pass

            if response.status_code == 429:
                raise UsageLimitExceededError(detail)
            elif response.status_code in [403,432,433]:
                raise ForbiddenError(detail)
            elif response.status_code == 401:
                raise InvalidAPIKeyError(detail)
            elif response.status_code == 400:
                raise BadRequestError(detail)
            else:
                raise response.raise_for_status()

    def _extract(self, urls, include_images=False, extract_depth="basic",
                 timeout=60, **kwargs):
        """
        Internal extract method to send the request to the API.
        """
        data = {"urls": urls, "include_images": include_images,
                "extract_depth": extract_depth}
        if kwargs:
            data.update(kwargs)

        timeout = min(timeout, 120)

        response = requests.post(self.base_url + "/extract",
                                 data=json.dumps(data), headers=self.headers,
                                 timeout=timeout, proxies=self.proxies,
                                 verify=self.verify)

        if response.status_code == 200:
            return response.json()
        else:
            detail = ""
            try:
                detail = response.json().get("detail", {}).get("error", None)
            except Exception:
                pass

            if response.status_code == 429:
                raise UsageLimitExceededError(detail)
            elif response.status_code in [403,432,433]:
                raise ForbiddenError(detail)
            elif response.status_code == 401:
                raise InvalidAPIKeyError(detail)
            elif response.status_code == 400:
                raise BadRequestError(detail)
            else:
                raise response.raise_for_status()


class APITavilySearch(APISearchEngineLinkSearch):
    """Search the web for links using the Tavily API"""

    _SE_NAME = "Tavily API"

    API_KEY_VAR = "TAVILY_API_KEY"
    """Environment variable that should contain the Tavily API key"""

    def __init__(self, api_key=None, verify=False):
        """

        Parameters
        ----------
        api_key : str, optional
            API key for serper search API. If ``None``, will look up the
            API key using the ``"TAVILY_API_KEY"`` environment variable.
            By default, ``None``.
        verify : bool, default=False
            Option to use SSL verification when making request to API
            endpoint. By default, ``False``.
        """
        super().__init__(api_key=api_key)
        self.verify = verify

    async def _search(self, query, num_results=10):
        """Search web for links related to a query"""

        client = _PatchedTavilyClient(api_key=self.api_key, verify=self.verify)
        response = client.search(query=query, max_results=num_results)
        results = response.get("results", [])
        return list(filter(None, (info.get('url', "").replace("+", "%20")
                                  for info in results)))
