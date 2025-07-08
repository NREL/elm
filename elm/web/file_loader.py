# -*- coding: utf-8 -*-
"""ELM Web file loader class."""
import asyncio
import logging

import aiohttp
from fake_useragent import UserAgent

from elm.utilities.parse import read_pdf
from elm.web.document import PDFDocument, HTMLDocument
from elm.web.html_pw import load_html_with_pw
from elm.web.utilities import DEFAULT_HEADERS
from elm.utilities.retry import async_retry_with_exponential_backoff
from elm.exceptions import ELMRuntimeError


logger = logging.getLogger(__name__)


async def _read_pdf_doc(pdf_bytes, **kwargs):
    """Default read PDF function (runs in main thread)"""
    verbose = kwargs.pop("verbose", True)
    pages = read_pdf(pdf_bytes, verbose=verbose)
    return PDFDocument(pages, **kwargs)


async def _read_html_doc(text, **kwargs):
    """Default read HTML function (runs in main thread)"""
    return HTMLDocument([text], **kwargs)


class AsyncFileLoader:
    """Async web file (PDF or HTML) loader

    Purpose:
        Save content from links as files.
    Responsibilities:
        1. Retrieve data from a URL.
        2. Determine wether information should be stored as a PDF or
           HTML document.
    Key Relationships:
        Returns either :class:`~elm.web.document.PDFDocument` or
        :class:`~elm.web.document.HTMLDocument`. Uses `aiohttp` to
        access the web.

    .. end desc
    """

    PAGE_LOAD_TIMEOUT = 60_000
    """Default page load timeout value in milliseconds"""

    def __init__(
        self,
        header_template=None,
        verify_ssl=True,
        aget_kwargs=None,
        pw_launch_kwargs=None,
        pdf_read_kwargs=None,
        html_read_kwargs=None,
        pdf_read_coroutine=None,
        html_read_coroutine=None,
        pdf_ocr_read_coroutine=None,
        file_cache_coroutine=None,
        browser_semaphore=None,
        use_scrapling_stealth=False,
        **__,  # consume any extra kwargs
    ):
        """

        Parameters
        ----------
        header_template : dict, optional
            Optional GET header template. If not specified, uses
            :obj:`~elm.web.utilities.DEFAULT_HEADERS`.
            By default, ``None``.
        verify_ssl : bool, optional
            Option to use aiohttp's default SSL check. If ``False``,
            SSL certificate validation is skipped. By default, ``True``.
        aget_kwargs : dict, optional
            Other kwargs to pass to :meth:`aiohttp.ClientSession.get`.
            By default, ``None``.
        pw_launch_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            :meth:`async_playwright.chromium.launch` (only used when
            reading HTML). By default, ``None``.
        pdf_read_kwargs : dict, optional
            Keyword-value argument pairs to pass to the
            `pdf_read_coroutine`. By default, ``None``.
        html_read_kwargs : dict, optional
            Keyword-value argument pairs to pass to the
            `html_read_coroutine`. By default, ``None``.
        pdf_read_coroutine : callable, optional
            PDF file read coroutine. Must by an async function. Should
            accept PDF bytes as the first argument and kwargs as the
            rest. Must return a :obj:`elm.web.document.PDFDocument`.
            If ``None``, a default function that runs in the main thread
            is used. By default, ``None``.
        html_read_coroutine : callable, optional
            HTML file read coroutine. Must by an async function. Should
            accept HTML text as the first argument and kwargs as the
            rest. Must return a :obj:`elm.web.document.HTMLDocument`.
            If ``None``, a default function that runs in the main thread
            is used. By default, ``None``.
        pdf_ocr_read_coroutine : callable, optional
            PDF OCR file read coroutine. Must by an async function.
            Should accept PDF bytes as the first argument and kwargs as
            the rest. Must return a :obj:`elm.web.document.PDFDocument`.
            If ``None``, PDF OCR parsing is not attempted, and any
            scanned PDF URL's will return a blank document.
            By default, ``None``.
        file_cache_coroutine : callable, optional
            File caching coroutine. Can be used to cache files
            downloaded by this class. Must accept an
            :obj:`~elm.web.document.Document` instance as the first
            argument and the file content to be written as the second
            argument. If this method is not provided, no document
            caching is performed. By default, ``None``.
        browser_semaphore : asyncio.Semaphore, optional
            Semaphore instance that can be used to limit the number of
            playwright browsers open concurrently. If ``None``, no
            limits are applied. By default, ``None``.
        use_scrapling_stealth : bool, default=False
            Option to use scrapling stealth scripts instead of
            tf-playwright-stealth. By default, ``False``.
        """
        self.pw_launch_kwargs = pw_launch_kwargs or {}
        self.pdf_read_kwargs = pdf_read_kwargs or {}
        self.html_read_kwargs = html_read_kwargs or {}
        self.get_kwargs = {
            "headers": self._header_from_template(header_template),
            "ssl": None if verify_ssl else False,
            **(aget_kwargs or {}),
        }
        self.pdf_read_coroutine = pdf_read_coroutine or _read_pdf_doc
        self.html_read_coroutine = html_read_coroutine or _read_html_doc
        self.pdf_ocr_read_coroutine = pdf_ocr_read_coroutine
        self.file_cache_coroutine = file_cache_coroutine
        self.browser_semaphore = browser_semaphore
        self.uss = use_scrapling_stealth

    def _header_from_template(self, header_template):
        """Compile header from user or default template"""
        headers = header_template or DEFAULT_HEADERS
        headers = dict(headers)
        if not headers.get("User-Agent"):
            headers["User-Agent"] = UserAgent().random
        return headers

    async def fetch_all(self, *urls):
        """Fetch documents for all requested URL's.

        Parameters
        ----------
        *urls
            Iterable of URL's (as strings) to fetch.

        Returns
        -------
        list
            List of documents, one per requested URL.
        """
        outer_task_name = asyncio.current_task().get_name()
        fetches = [
            asyncio.create_task(self.fetch(url), name=outer_task_name)
            for url in urls
        ]
        return await asyncio.gather(*fetches)

    async def fetch(self, url):
        """Fetch a document for the given URL.

        Parameters
        ----------
        url : str
            URL for the document to pull down.

        Returns
        -------
        :class:`elm.web.document.Document`
            Document instance containing text, if the fetch was
            successful.
        """
        try:
            doc, raw_content = await self._fetch_doc_with_url_in_metadata(url)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = ("Encountered error of type %r while fetching document from "
                   "%s:")
            err_type = type(e)
            logger.exception(msg, err_type, url)
            return HTMLDocument(pages=[])

        doc = await self._cache_doc(doc, raw_content)
        return doc

    async def _fetch_doc_with_url_in_metadata(self, url):
        """Fetch doc contents and add URL to metadata"""
        doc, raw_content = await self._fetch_doc(url)
        doc.attrs["source"] = url
        return doc, raw_content

    async def _fetch_doc(self, url):
        """Fetch a doc by trying pdf read, then HTML read, then PDF OCR"""

        async with aiohttp.ClientSession() as session:
            try:
                logger.debug("Fetching content from %r", url)
                url_bytes = await self._fetch_content_with_retry(url, session)
            except ELMRuntimeError:
                logger.exception("Could not fetch content from %r", url)
                return PDFDocument(pages=[]), None

        logger.debug("Got content from %r", url)
        doc = await self.pdf_read_coroutine(url_bytes, **self.pdf_read_kwargs)
        if not doc.empty:
            return doc, url_bytes

        logger.debug("PDF read failed; fetching HTML content from %r", url)
        text = await load_html_with_pw(url, self.browser_semaphore,
                                       timeout=self.PAGE_LOAD_TIMEOUT,
                                       use_scrapling_stealth=self.uss,
                                       **self.pw_launch_kwargs)
        doc = await self.html_read_coroutine(text, **self.html_read_kwargs)
        if not doc.empty:
            return doc, doc.text

        if self.pdf_ocr_read_coroutine:
            logger.debug("HTML read failed; fetching OCR content from %r", url)
            doc = await self.pdf_ocr_read_coroutine(
                url_bytes, **self.pdf_read_kwargs
            )

        return doc, url_bytes

    @async_retry_with_exponential_backoff(
        base_delay=2,
        exponential_base=1.5,
        jitter=False,
        max_retries=3,
        errors=(
            aiohttp.ClientConnectionError,
            aiohttp.client_exceptions.ClientError,
        ),
    )
    async def _fetch_content_with_retry(self, url, session):
        """Fetch content from URL with several retry attempts"""
        async with session.get(url, **self.get_kwargs) as response:
            return await response.read()

    async def _cache_doc(self, doc, raw_content):
        """Cache doc if user provided a coroutine"""
        if doc.empty or not raw_content:
            return doc

        if not self.file_cache_coroutine:
            return doc

        cache_fn = await self.file_cache_coroutine(doc, raw_content)
        if cache_fn is not None:
            doc.attrs["cache_fn"] = cache_fn
        return doc
