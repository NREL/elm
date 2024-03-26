# -*- coding: utf-8 -*-
"""ELM Ordinance county file downloading logic"""
import pprint
import asyncio
import logging
from itertools import zip_longest, chain
from contextlib import AsyncExitStack

from elm.ords.llm import StructuredLLMCaller
from elm.ords.extraction import check_for_ordinance_info
from elm.ords.services.threaded import TempFileCache
from elm.ords.validation.location import CountyValidator
from elm.web.document import PDFDocument
from elm.web.file_loader import AsyncFileLoader
from elm.web.google_search import PlaywrightGoogleLinkSearch


logger = logging.getLogger(__name__)
QUESTION_TEMPLATES = [
    '0. "wind energy conversion system zoning ordinances {location}"',
    '1. "{location} wind WECS zoning ordinance"',
    '2. "Where can I find the legal text for commercial wind energy '
    'conversion system zoning ordinances in {location}?"',
    '3. "What is the specific legal information regarding zoning '
    'ordinances for commercial wind energy conversion systems in {location}?"',
]


async def _search_single(
    location, question, browser_sem, num_results=10, **kwargs
):
    """Perform a single google search."""
    if browser_sem is None:
        browser_sem = AsyncExitStack()

    search_engine = PlaywrightGoogleLinkSearch(**kwargs)
    async with browser_sem:
        return await search_engine.results(
            question.format(location=location),
            num_results=num_results,
        )


async def _find_urls(location, num_results=10, browser_sem=None, **kwargs):
    """Parse google search output for URLs."""
    searchers = [
        asyncio.create_task(
            _search_single(
                location, q, browser_sem, num_results=num_results, **kwargs
            ),
            name=location,
        )
        for q in QUESTION_TEMPLATES
    ]
    return await asyncio.gather(*searchers)


def _down_select_urls(search_results, num_urls=5):
    """Select the top 5 URLs."""
    all_urls = chain.from_iterable(
        zip_longest(*[results[0] for results in search_results])
    )
    urls = set()
    for url in all_urls:
        if not url:
            continue
        urls.add(url)
        if len(urls) == num_urls:
            break
    return urls


async def _load_docs(urls, text_splitter, browser_semaphore=None, **kwargs):
    """Load a document for each input URL."""
    loader_kwargs = {
        "html_read_kwargs": {"text_splitter": text_splitter},
        "file_cache_coroutine": TempFileCache.call,
        "browser_semaphore": browser_semaphore,
    }
    loader_kwargs.update(kwargs)
    file_loader = AsyncFileLoader(**loader_kwargs)
    docs = await file_loader.fetch_all(*urls)

    logger.debug(
        "Loaded the following number of pages for docs: %s",
        pprint.PrettyPrinter().pformat(
            {
                doc.metadata.get("source", "Unknown"): len(doc.pages)
                for doc in docs
            }
        ),
    )
    return [doc for doc in docs if not doc.empty]


async def _down_select_docs_correct_location(
    docs, location, county, state, **kwargs
):
    """Remove all documents not pertaining to the location."""
    llm_caller = StructuredLLMCaller(**kwargs)
    county_validator = CountyValidator(llm_caller)
    searchers = [
        asyncio.create_task(
            county_validator.check(doc, county=county, state=state),
            name=location,
        )
        for doc in docs
    ]
    output = await asyncio.gather(*searchers)
    correct_loc_docs = [doc for doc, check in zip(docs, output) if check]
    return sorted(
        correct_loc_docs,
        key=lambda doc: (not isinstance(doc, PDFDocument), len(doc.text)),
    )


async def _check_docs_for_ords(docs, text_splitter, **kwargs):
    """Check documents to see if they contain ordinance info."""
    ord_docs = []
    for doc in docs:
        doc = await check_for_ordinance_info(doc, text_splitter, **kwargs)
        if doc.metadata["contains_ord_info"]:
            ord_docs.append(doc)
    return ord_docs


def _parse_all_ord_docs(all_ord_docs):
    """Parse a list of documents and get the result for the best match."""
    if not all_ord_docs:
        return None

    return sorted(all_ord_docs, key=_ord_doc_sorting_key)[-1]


def _ord_doc_sorting_key(doc):
    """All text sorting key"""
    year, month, day = doc.metadata.get("date", (-1, -1, -1))
    return year, isinstance(doc, PDFDocument), -1 * len(doc.text), month, day


async def download_county_ordinance(
    location,
    text_splitter,
    num_urls=5,
    file_loader_kwargs=None,
    browser_semaphore=None,
    **kwargs
):
    """Download the ordinance document for a single county.

    Parameters
    ----------
    location : elm.ords.utilities.location.Location
        Location objects representing the county.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    num_urls : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword-argument pairs to initialize
        :class:`elm.web.file_loader.AsyncFileLoader` with. The
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.google_search.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.
    browser_semaphore : asyncio.Semaphore, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    **kwargs
        Keyword-value pairs used to initialize an
        `elm.ords.llm.LLMCaller` instance.

    Returns
    -------
    elm.web.document.BaseDocument | None
        Document instance for the downloaded document, or ``None`` if no
        document was found.
    """
    file_loader_kwargs = file_loader_kwargs or {}
    pw_launch_kwargs = file_loader_kwargs.get("pw_launch_kwargs", {})
    urls = await _find_urls(
        location.full_name,
        num_results=10,
        browser_sem=browser_semaphore,
        **pw_launch_kwargs
    )
    urls = _down_select_urls(urls, num_urls=num_urls)
    logger.debug("Downloading documents for URLS: \n\t-%s", "\n\t-".join(urls))
    docs = await _load_docs(
        urls, text_splitter, browser_semaphore, **file_loader_kwargs
    )
    docs = await _down_select_docs_correct_location(
        docs,
        location=location.full_name,
        county=location.name,
        state=location.state,
        **kwargs
    )
    docs = await _check_docs_for_ords(docs, text_splitter, **kwargs)
    logger.info(
        "Found %d potential ordinance documents for %s",
        len(docs),
        location.full_name,
    )
    return _parse_all_ord_docs(docs)
