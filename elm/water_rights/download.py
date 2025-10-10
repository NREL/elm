# -*- coding: utf-8 -*-
"""ELM Ordinance county file downloading logic"""
import logging

from elm.ords.llm import StructuredLLMCaller
from elm.water_rights.extraction import check_for_ordinance_info
from elm.ords.services.threaded import TempFileCache
from elm.water_rights.validation.location import CountyValidator
from elm.web.search import web_search_links_as_docs
from elm.web.utilities import filter_documents


logger = logging.getLogger(__name__)

QUESTION_TEMPLATES = [
    "{location} rules",
    "{location} management plan",
    "{location} well permits",
    "{location} well permit requirements",
    "requirements to drill a water well in {location}",
    ]

async def download_county_ordinance(
    location,
    text_splitter,
    num_urls=5,
    file_loader_kwargs=None,
    browser_semaphore=None,
    **kwargs
):
    """Download ordinance documents for a single conservation district.

    Parameters
    ----------
    location : :class:`elm.ords.utilities.location.Location`
        Location objects representing the conservation district.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Raw text from HTML pages will be passed through
        this splitter to split the single wep page into multiple pages
        for the output document. Langchain's text splitters should work
        for this input.
    num_urls : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword-argument pairs to initialize
        :class:`elm.web.file_loader.AsyncFileLoader` with. If found, the
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.google_search.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.
    browser_semaphore : :class:`asyncio.Semaphore`, optional
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
    docs = await _docs_from_google_search(
        location,
        text_splitter,
        num_urls,
        browser_semaphore,
        **(file_loader_kwargs or {})
    )

    docs = await _down_select_docs_correct_location(
        docs, location=location, **kwargs
    )

    docs = await _down_select_docs_correct_content(
        docs, location=location, text_splitter=text_splitter, **kwargs
    )

    logger.info(
        "Found %d potential ordinance documents for %s",
        len(docs),
        location.full_name,
    )

    return docs


async def _docs_from_google_search(
    location, text_splitter, num_urls, browser_semaphore, **file_loader_kwargs
):
    """Download docs from google location queries. """
    queries = [
        question.format(location=location.name)
        for question in QUESTION_TEMPLATES
    ]

    file_loader_kwargs.update(
        {
            "html_read_kwargs": {"text_splitter": text_splitter},
            "file_cache_coroutine": TempFileCache.call,
        }
    )
    return await web_search_links_as_docs(
        queries,
        num_urls=num_urls,
        browser_semaphore=browser_semaphore,
        task_name=location.full_name,
        **file_loader_kwargs,
    )


async def _down_select_docs_correct_location(docs, location, **kwargs):
    """Remove all documents not pertaining to the location."""
    llm_caller = StructuredLLMCaller(**kwargs)
    county_validator = CountyValidator(llm_caller, score_thresh=0.6)
    return await filter_documents(
        docs,
        validation_coroutine=county_validator.check,
        task_name=location.full_name,
        county=location.name,
        county_acronym=location.acronym,
        state=location.state,
    )


async def _down_select_docs_correct_content(docs, location, **kwargs):
    """Remove all documents that don't contain ordinance info."""
    return await filter_documents(
        docs,
        validation_coroutine=_contains_ords,
        task_name=location.full_name,
        **kwargs,
    )


async def _contains_ords(doc, **kwargs):
    """Helper coroutine that checks for ordinance info. """
    doc = await check_for_ordinance_info(doc, **kwargs)
    return doc.attrs.get("contains_ord_info", False)
