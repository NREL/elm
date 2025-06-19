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

_BLACKLIST_SUBSTRINGS = ["*mailto:*",
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

ELM_URL_FILTER = URLPatternFilter(reverse=True, patterns=_BLACKLIST_SUBSTRINGS)
"""Filter used to exclude URLs that are not relevant to the search"""


class ELMLinkScorer:
    """Custom URL scorer for ELM website crawling"""

    def __init__(self, keyword_points=None):
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


class ELMWebsiteCrawlingStrategy(BestFirstCrawlingStrategy):
    """Custom crawling strategy for ELM website searching"""

    BATCH_SIZE = 10
    """Number of URLs to process in each batch"""

    async def link_discovery(self, result, source_url, current_depth, visited,
                             next_links, depths):
        """
        Extract links from the crawl result, validate them, and append new
        URLs (with their parent references) to next_links.
        Also updates the depths dictionary.

        .. NOTE:: Overridden from original BestFirstCrawlingStrategy to
                  implement return full link dictionary instead of just
                  URLs.
        """
        new_depth = current_depth + 1
        if new_depth > self.max_depth:
            return

        remaining_capacity = self.max_pages - self._pages_crawled
        if remaining_capacity <= 0:
            self.logger.info(f"Max pages limit ({self.max_pages}) reached, "
                             "stopping link discovery")
            return

        links = [(link, False) for link in result.links.get("internal", [])]
        if self.include_external:
            links += [(link, True)
                      for link in result.links.get("external", [])]

        valid_links = []
        for link, is_external in links:
            url = link.get("href")
            base_url = normalize_url_for_deep_crawl(url, source_url)
            if base_url in visited:
                continue
            if not await self.can_process_url(url, new_depth):
                self.stats.urls_skipped += 1
                continue

            valid_links.append((link, is_external))

        for link, is_external in valid_links:
            url = link.get("href")
            depths[url] = new_depth
            next_links.append((link, url, source_url, is_external))

    async def _arun_best_first(self, start_url, crawler, config):
        """
        Core best-first crawl method using a priority queue.

        The queue items are tuples of
        (score, depth, url, parent_url, is_external).
        Lower scores are treated as higher priority.
        URLs are processed in batches for efficiency.

        .. NOTE:: Overridden from original BestFirstCrawlingStrategy to
                  implement pass the full link dictionary to the
                  `url_scorer` (instead of just the URL string). Also
                  fixes queue scoring bug in the original code. Also
                  don't allow discovering links from external URLs.

        """
        queue = asyncio.PriorityQueue()
        await queue.put((0, 0, start_url, None, False))
        visited = set()
        depths = {start_url: 0}

        while not queue.empty() and not self._cancel_event.is_set():
            if self._pages_crawled >= self.max_pages:
                self.logger.info(f"Max pages limit ({self.max_pages}) "
                                 "reached, stopping crawl")
                break

            batch = []
            for _ in range(self.BATCH_SIZE):
                if queue.empty():
                    break
                item = await queue.get()
                score, depth, url, parent_url, is_external = item
                if url in visited:
                    continue
                visited.add(url)
                batch.append(item)

            if not batch:
                continue

            urls = [item[2] for item in batch]
            batch_config = config.clone(deep_crawl_strategy=None, stream=True)
            stream_gen = await crawler.arun_many(urls=urls,
                                                 config=batch_config)
            async for result in stream_gen:
                result_url = result.url
                corresponding = next((item
                                      for item in batch
                                      if item[2] == result_url), None)
                if not corresponding:
                    continue

                score, depth, url, parent_url, is_external = corresponding
                result.metadata = result.metadata or {}
                result.metadata["depth"] = depth
                result.metadata["parent_url"] = parent_url
                result.metadata["score"] = -1 * score

                if result.success:
                    self._pages_crawled += 1

                yield result

                if result.success and not is_external:
                    new_links = []
                    await self.link_discovery(result, result_url, depth,
                                              visited, new_links, depths)

                    for link, new_url, new_parent, is_external in new_links:
                        new_depth = depths.get(new_url, depth + 1)
                        new_score = (-1 * self.url_scorer.score(link)
                                     if self.url_scorer else 0)
                        await queue.put((new_score, new_depth, new_url,
                                         new_parent, is_external))
