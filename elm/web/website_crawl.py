# -*- coding: utf-8 -*-
# flake8: noqa
# pylint: disable=no-member
"""ELM Document retrieval from a website"""

import logging
from asyncio import PriorityQueue
from math import inf as infinity
from contextlib import aclosing
from functools import lru_cache

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
from crawl4ai.utils import normalize_url_for_deep_crawl
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (FilterChain, URLPatternFilter,
                                            ContentTypeFilter, URLFilter)

from elm.web.file_loader import AsyncFileLoader
from elm.web.document import HTMLDocument


logger = logging.getLogger(__name__)
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

_BLACKLIST_SUBSTRINGS = ["*login*",
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

_SCORE_KEY = "website_link_relevance_score"


class PeekablePriorityQueue(PriorityQueue):
    """A priority queue that allows peeking at the next item"""

    def peek(self):
        """Peek at the next item in the queue without removing it"""
        if self.empty():
            return None
        return self._queue[0]


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

    async def score(self, links):
        """Score a link based on its title text and URL.

        The score is calculated by summing the point values of the
        keywords found in both the link title text and the URL. The
        higher the score, the more relevant the link is considered to be
        for the purpose of retrieving zoning ordinance documents.

        Parameters
        ----------
        links : list of dicts
            A list of dictionaries representing links to be scored.
            Each dictionary should contain at least the keys "text" and
            "href". The "text" key should contain the link title text,
            and the "href" key should contain the URL.

        Returns
        -------
        links : list of dicts
            The input list of links with an additional key "score" added
            to each dictionary. Each "score" key contains the calculated
            score for that link.
        """
        for link in links:
            link["score"] = (self._assign_value(link.get("text", ""))
                             + self._assign_value(link.get("href", "")))

        return links

    def _assign_value(self, text):
        """Assign a score based on the presence of keywords in the text"""
        score = 0
        for kw, kw_score in self.keyword_points.items():
            if kw in text.casefold():
                score += kw_score
        return score


class ContentTypeExcludeFilter(URLFilter):
    """Content type to exclude filter using fast lookups"""

    __slots__ = ("exclude_ext", )

    # Fast extension to mime type mapping
    EXCLUDE_EXTENSIONS = {# Images
                          'bmp', 'gif', 'ico', 'jpeg', 'jpg', 'png', 'svg',
                          'tiff', 'webp',
                          # Audio
                          'aac', 'm4a', 'mp3', 'ogg', 'wav',
                          # Video
                          'avi', 'flv', 'mkv', 'mp4', 'mpeg', 'mov', 'webm',
                          'wmv',
                          # Applications
                          '7z', 'exe', 'gz', 'json', 'msi', 'pdf', 'rar',
                          'tar', 'xml', 'zip',
                          # Fonts
                          'otf', 'ttf', 'woff', 'woff2',
                          # OpenDocument Formats
                          'odp', 'ods', 'odt',
                          # Archives
                          'bz2', 'tar.gz', 'tgz',
                          # Others
                          'ai', 'apk', 'bin', 'deb', 'dmg', 'eps', 'epub',
                          'iso', 'jar', 'mid', 'midi', 'ps', 'rpm', 'rtf',
                          'sqlite', 'swf'}
    """File extensions to exclude from the crawl."""

    def __init__(self, exclude_extensions=None):
        """

        Parameters
        ----------
        exclude_extensions : str or list of str, optional
            File extensions to exclude from the crawl. If a string is
            provided, it is treated as a single extension. If a list is
            provided, each item in the list is treated as an extension.
            If ``None``, uses the default set of excluded extensions
            defined in
            :obj:`ContentTypeExcludeFilter.EXCLUDE_EXTENSIONS`.
            By default, ``None``.
        """
        super().__init__()

        if isinstance(exclude_extensions, str):
            exclude_extensions = [exclude_extensions]

        self.exclude_ext = frozenset(t.casefold()
                                     for t in (exclude_extensions
                                               or self.EXCLUDE_EXTENSIONS))

    @lru_cache(maxsize=1000)
    def _check_url_cached(self, url: str) -> bool:
        """Cached URL checking"""

        ext = ContentTypeFilter._extract_extension(url)
        if not ext:
            return True

        return ext not in self.exclude_ext

    def apply(self, url: str) -> bool:
        """Fast extension check with caching"""
        result = self._check_url_cached(url)
        self._update_stats(result)
        return result


class ELMWebsiteCrawlingStrategy(BestFirstCrawlingStrategy):
    """Custom crawling strategy for ELM website searching"""

    BATCH_SIZE = 10
    """Number of URLs to process in each batch"""

    ONE_SCORE_AT_A_TIME = True
    """Whether to batch process only links with the same score.

    This works best if the score is an integer value, since scores are
    compared directly using the `==` operator.
    """

    @classmethod
    async def found_enough_docs(cls, out_docs):
        """Check if enough documents have been found.

        If :obj:`ELMWebsiteCrawlingStrategy.ONE_SCORE_AT_A_TIME` is
        ``True``, this function returns ``True`` when 5 documents have
        been found, otherwise it returns ``True`` when 8 documents
        have been found.

        Parameters
        ----------
        out_docs : list
            List of documents found during the crawl.

        Returns
        -------
        bool
            Whether enough documents have been found to stop the crawl.
        """
        doc_threshold = 5 if cls.ONE_SCORE_AT_A_TIME else 8
        return len(out_docs) >= doc_threshold

    async def link_discovery(self, result, source_url, current_depth, visited,
                             next_links):
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

        links = []
        for link in result.links.get("internal", []):
            link["is_external"] = False
            links.append(link)
        if self.include_external:
            for link in result.links.get("external", []):
                link["is_external"] = True
                links.append(link)

        valid_links = []
        for link in links:
            url = link.get("href")
            if not url:
                self.logger.debug("Link without href, skipping")
                continue
            base_url = normalize_url_for_deep_crawl(url, source_url)
            if base_url in visited:
                continue
            if not await self.can_process_url(url, new_depth):
                self.stats.urls_skipped += 1
                continue

            valid_links.append(link)

        for link in valid_links:
            link["source_url"] = source_url
            link["depth"] = new_depth
            next_links.append(link)

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
        queue = PeekablePriorityQueue()
        await queue.put((0, 0, start_url, None, False))
        visited = set()

        while not queue.empty() and not self._cancel_event.is_set():
            if self._pages_crawled >= self.max_pages:
                self.logger.info(f"Max pages limit ({self.max_pages}) "
                                 "reached, stopping crawl")
                break

            batch = []
            curr_score = None
            for _ in range(self.BATCH_SIZE):
                if queue.empty():
                    break
                if self.ONE_SCORE_AT_A_TIME and curr_score is not None:
                    item = queue.peek()
                    if item[0] != curr_score:
                        break

                item = await queue.get()
                score, depth, url, parent_url, is_external = item
                if url in visited:
                    continue
                visited.add(url)
                batch.append(item)

                if self.ONE_SCORE_AT_A_TIME and curr_score is None:
                    curr_score = score

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
                                              visited, new_links)

                    if self.url_scorer:
                        new_links = await self.url_scorer(new_links)

                    for link in new_links:
                        await queue.put((-1 * link.get("score", 0),
                                         link.get("depth", depth + 1),
                                         link.get("href"),
                                         link.get("source_url"),
                                         link.get("is_external", False)))


class ELMWebsiteCrawler:
    """Crawl a website for documents of interest"""

    def __init__(self, validator, file_loader_kwargs=None,
                 browser_config_kwargs=None, crawl_strategy_kwargs=None,
                 crawler_config_kwargs=None, cte_kwargs=None,
                 extra_url_filters=None, include_external=False,
                 url_scorer=None, max_pages=100, page_limit=None):
        """

        Parameters
        ----------
        validator : callable
            An async callable that takes a document instance (containing
            the text from a PDF or a webpage) and returns a boolean
            indicating whether the text passes the validation check.
            This is used to determine whether or not to keep (i.e.
            return) the document.
        file_loader_kwargs : dict, optional
            Additional keyword-value argument pairs to pass to the
            :class:`~elm.web.file_loader.AsyncFileLoader` class.
            By default, ``None``.
        browser_config_kwargs : dict, optional
            Additional keyword-value argument pairs to pass to the
            :class:`crawl4ai.async_configs.BrowserConfig` class.
            By default, ``None``.
        crawl_strategy_kwargs : dict, optional
            Additional keyword-value argument pairs to pass to the
            :class:`ELMWebsiteCrawlingStrategy` class.
            By default, ``None``.
        crawler_config_kwargs : dict, optional
            Additional keyword-value argument pairs to pass to the
            :class:`crawl4ai.async_configs.CrawlerRunConfig` class.
            By default, ``None``.
        cte_kwargs : dict, optional
            Additional keyword-value argument pairs to pass to the
            :class:`ContentTypeExcludeFilter` class. This filter is used
            to exclude URLs based on their content type.
            By default, ``None``.
        extra_url_filters : list, optional
            Additional URL filters to apply during crawling. Each filter
            must have a (non-async) ``apply`` method that takes a URL
            and returns a boolean indicating whether the URL should be
            included in the crawl. By default, ``None``.
        include_external : bool, optional
            Whether to include external links in the crawl.
            By default, ``False``.
        url_scorer : callable, optional
            An async callable that takes a list of dictionaries
            containing URL information and assigns each dictionary a
            `score` key representing the score for that URL. The input
            URL dictionaries will each have at least one key: "href".
            This key will contain the URL of the link. The dictionary
            may also have other attributes such as "text", which
            contains the link title text. If ``None``, uses the
            :meth:`ELMLinkScorer.score` method to score the URLs.
            By default, ``None``.
        max_pages : int, optional
            Maximum number of **successful** pages to crawl.
            By default, ``100``.
        page_limit : int, optional
            Maximum number of pages to crawl regardless of success
            status. If ``None``, a page limit of 2 * `max_pages` is
            used. To set no limit (not recommended), use ``math.inf``.
            By default, ``None``.
        """
        self.validator = validator
        self.page_limit = page_limit or 2 * max_pages

        flk = {"verify_ssl": False}
        flk.update(file_loader_kwargs or {})
        self.afl = AsyncFileLoader(**flk)

        bck = {"headless": True, "verbose": False}
        bck.update(browser_config_kwargs or {})
        self.browser_config = BrowserConfig(**bck)

        cte_kwargs = cte_kwargs or {}
        url_filters = [ELM_URL_FILTER, ContentTypeExcludeFilter(**cte_kwargs)]
        url_filters += extra_url_filters or []
        self.filter_chain = FilterChain(url_filters)

        strategy_kwargs = {"max_depth": infinity,
                           "include_external": include_external,
                           "filter_chain": self.filter_chain,
                           "url_scorer": url_scorer or ELMLinkScorer().score,
                           "max_pages": max_pages,
                           "logger": logger}
        strategy_kwargs.update(crawl_strategy_kwargs or {})
        self.crawl_strategy = ELMWebsiteCrawlingStrategy(**strategy_kwargs)

        cck = {"deep_crawl_strategy": self.crawl_strategy,
               "scraping_strategy": LXMLWebScrapingStrategy(),
               "stream": True,
               "verbose": False,
               "pdf": False,
               "log_console": False,
               "exclude_social_media_domains": ["youtube.com"],
               "exclude_social_media_links": True,
               "semaphore_count": 1}
        cck.update(crawler_config_kwargs or {})
        self.config = CrawlerRunConfig(**cck)

    async def run(self, base_url, termination_callback=None,
                  on_result_hook=None, return_c4ai_results=False):
        """Crawl a website for documents of interest

        Parameters
        ----------
        base_url : str
            The base URL to start crawling from.
        termination_callback : callable, optional
            An async callable that takes a list of documents and returns
            a boolean indicating whether to stop crawling. If ``None``,
            the :meth:`ELMWebsiteCrawlingStrategy.found_enough_docs` is
            used, which simply terminates when roughly a handful of
            documents have been found. By default, ``None``.
        on_result_hook : callable, optional
            An async callable that is called every time a result is
            found during the crawl. This can be used to perform
            additional processing on each result or to monitor the crawl
            progress. The callable should accept a single argument,
            which is the crawl result object. If ``None``, no additional
            processing is done on the results. By default, ``None``.
        return_c4ai_results : bool, optional
            Whether to return the raw crawl4ai results along with the
            documents. If ``True``, returns a tuple of (documents,
            crawl4ai_results). If ``False``, returns only the documents.
            By default, ``False``.

        Returns
        -------
        out_docs
            List of document instances that passed the validation
            check. Each document contains the text from a PDF or a
            webpage, and has an attribute `source` that contains the
            URL of the document.
        results : list, optional
            List of crawl4ai results containing metadata about the
            crawled pages. This is only returned if
            `return_c4ai_results` is ``True``.
        """

        results = []
        out_docs = []
        should_stop = (termination_callback
                       or ELMWebsiteCrawlingStrategy.found_enough_docs)
        page_count = 0
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            crawl_results = await crawler.arun(base_url, config=self.config)
            async with aclosing(crawl_results) as agen:
                async for result in agen:
                    page_count += 1
                    if page_count > self.page_limit:
                        logger.debug("Exiting crawl due to page limit")
                        break
                    if not result.success:
                        continue
                    results.append(result)
                    logger.debug("Crawled %s", result.url)
                    if on_result_hook:
                        await on_result_hook(result)

                    score = result.metadata.get("score", 0)
                    depth = result.metadata.get("depth", 0)
                    logger.trace("\t- Depth: %d | Score: %.2f", depth, score)

                    doc = await self._doc_from_result(result)
                    doc.attrs[_SCORE_KEY] = score
                    if doc.empty:
                        logger.debug("Empty document, skipping")
                        continue

                    if await self.validator(doc):
                        logger.debug("Document passed validation check")
                        out_docs.append(doc)

                    if await should_stop(out_docs):
                        logger.debug("Exiting crawl early")
                        break

        logger.info("Crawled %d pages", len(results))
        logger.info("Found %d potential documents", len(out_docs))
        logger.debug("Average score: %.2f", _compute_avg_score(results))

        depth_counts = {}
        for result in results:
            depth = result.metadata.get("depth", 0)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        logger.debug("Pages crawled by depth:")
        for depth, count in sorted(depth_counts.items()):
            logger.debug(f"  Depth {depth}: {count} pages")

        if out_docs:
            out_docs.sort(key=lambda x: -1 * x.attrs[_SCORE_KEY])

        if return_c4ai_results:
            return out_docs, results

        return out_docs

    async def _doc_from_result(self, result):
        """Get document instance from crawling result"""
        if result.markdown and len(md := result.markdown.strip()) > 3:
            doc = HTMLDocument([md])
            doc.attrs["source"] = result.url
            return doc

        return await self.afl.fetch(result.url)


def _compute_avg_score(results):
    """Compute the average score of the crawled results"""
    if len(results) <= 0:
        return 0
    return sum(r.metadata.get('score', 0) for r in results) / len(results)
