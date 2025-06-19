# -*- coding: utf-8 -*-
"""ELM Document retrieval from a website"""

import asyncio
from math import inf as infinity

from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.utils import normalize_url_for_deep_crawl
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    DomainFilter,
    URLPatternFilter,
    ContentTypeFilter
)
from crawl4ai.async_configs import BrowserConfig
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

from elm.web.file_loader import AsyncFileLoader
from elm.web.document import HTMLDocument


BEST_ZONING_ORDINANCE_KEYWORDS = {"pdf": 1152,
                                  "zoning": 576,
                                  "ordinance": 288,
                                  "planning": 144,
                                  "plan": 72,
                                  "government": 36,
                                  "code": 12,
                                  "area": 12,
                                  "land development": 6,
                                  "land": 3,
                                  "municipal": 1,
                                  "department": 1}
"""Keywords and their associated point values for scoring URLs.

These keywords are used to prioritize URLs based on their relevance
to contain documents of interest.

The general scoring strategy employed here is that the top keyword
should be worth more points than the sum of all of the keywords below
it. This is just the default strategy, and can/should be modified by
users to better suit their own use cases.
"""

BLACKLIST_SUBSTRINGS = ["*mailto:*",
                        "*tel:*",
                        "*fax:*",
                        "*javascript:*",
                        "*login*",
                        "*signup*",
                        "*sign up*",
                        r"*sign%20up*",
                        "*signin*",
                        "*sign in*",
                        r"*sign%20in*",
                        "*register*",
                        "*subscribe*",
                        "*donate*",
                        "*shop*",
                        "*cart*",
                        "*careers*",
                        "*events*",
                        "*calendar*"]
"""Substrings used to exclude URLs that are not relevant to the search"""


class ELMLinkScorer:
    """Custom URL scorer for ELM website crawling"""

    def __call__(self, keyword_points=None):
        """

        Parameters
        ----------
        keyword_points : dict, optional
            Dictionary mapping keywords to their associated point values
            for scoring URLs. If ``None``, uses
            :obj:`BEST_ZONING_ORDINANCE_KEYWORDS`.
            By default, ``None``.
        """
        self.keyword_points = keyword_points or BEST_ZONING_ORDINANCE_KEYWORDS

    def score(self, link):
        """Score a link based on its title text and URL.

        The score is calculated by summing the point values of the
        keywords found in both the link title text and the URL. The
        higher the score, the more relevant the link is considered to be
        for the purpose of retrieving zoning ordinance documents.

        Parameters
        ----------
        link : crawl4ai.Link
            The link to be scored, which is expected to be a dictionary
            containing at least the keys "text" and "href".
            The "text" key should contain the link text, and the "href"
            key should contain the URL.

        Returns
        -------
        int
            Score for the link based on the presence of keywords
            in the link text and URL.
        """
        return (self._assign_value(link.get("text", ""))
                + self._assign_value(link.get("href", "")))

    def _assign_value(self, text):
        """Assign a score based on the presence of keywords in the text"""
        score = 0
        for kw, kw_score in self.keyword_points.items():
            if kw in text.casefold():
                score += kw_score
        return score
