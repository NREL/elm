# -*- coding: utf-8 -*-
"""ELM Ordinance county file downloading logic"""
import asyncio
import logging
from itertools import zip_longest, chain

from elm.ords.extraction import extract_ordinance_info
from elm.ords.services.temp_file_cache import TempFileCache
from elm.ords.validation.location import CountyValidator
from elm.web.document import PDFDocument
from elm.web.file_loader import AsyncFileLoader
from elm.web.google_search import PlaywrightGoogleLinkSearch


logger = logging.getLogger(__name__)
QUESTION_TEMPLATES = [
    '0. "wind energy conversion system zoning ordinances {location}"',
    '1. "{location} wind WECS zoning ordinance"',
    '2. "Where can I find the legal text for commercial wind energy conversion system zoning ordinances in {location}?"',
    '3. "What is the specific legal information regarding zoning ordinances for commercial wind energy conversion systems in {location}?"',
]


async def _search_single(location, question, num_results=10):
    """Perform a single google search."""
    search_engine = PlaywrightGoogleLinkSearch()
    return await search_engine.results(
        question.format(location=location),
        num_results=num_results,
    )


async def _find_urls(location, num_results=10):
    """Parse google search output for URLs."""
    searchers = [
        asyncio.create_task(
            _search_single(location, q, num_results=num_results),
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


async def _load_docs(urls, text_splitter):
    """Load a document for each input URL."""
    file_loader = AsyncFileLoader(
        html_read_kwargs={"text_splitter": text_splitter},
        file_cache_coroutine=TempFileCache.call,
    )
    return await file_loader.fetch_all(*urls)


async def _down_select_docs_correct_location(
    docs, llm_caller, location, county, state
):
    """Remove all documents not pertaining to the location."""
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


async def _check_docs_for_ords(docs, llm_caller, text_splitter):
    """Check documents to see if they contain ordinance info."""
    ord_docs = []
    for doc in docs:
        doc = await extract_ordinance_info(doc, llm_caller, text_splitter)
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
    location, llm_caller, text_splitter, num_urls=5
):
    """Download the ordinance document for a single county.

    Parameters
    ----------
    location : elm.ords.utilities.location.Location
        Location objects representing the county.
    llm_caller : elm.ords.llm.StructuredLLMCaller
        StructuredLLMCaller instance used for validation and checking
        the contents of the document.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    num_urls : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.

    Returns
    -------
    elm.web.document.BaseDocument | None
        Document instance for the downloaded document, or ``None`` if no
        document was found.
    """
    urls = await _find_urls(location.full_name, num_results=10)
    urls = _down_select_urls(urls, num_urls=num_urls)
    docs = await _load_docs(urls, text_splitter)
    docs = await _down_select_docs_correct_location(
        docs,
        llm_caller,
        location.full_name,
        county=location.name,
        state=location.state,
    )
    docs = await _check_docs_for_ords(docs, llm_caller, text_splitter)
    logger.info(
        "Found %d potential ordinance documents for %s",
        len(docs),
        location.full_name,
    )
    return _parse_all_ord_docs(docs)
