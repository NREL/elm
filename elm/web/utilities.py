# -*- coding: utf-8 -*-
"""ELM Web Scraping utilities."""
import uuid
import hashlib
import logging
import asyncio
from pathlib import Path
from copy import deepcopy
from random import uniform, randint
from contextlib import asynccontextmanager

from slugify import slugify
from fake_useragent import UserAgent
from playwright_stealth import stealth_async
from scrapling.engines import PlaywrightEngine

from elm.web.document import PDFDocument


logger = logging.getLogger(__name__)
DEFAULT_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
"""Default HTML header template"""
_BT_RENAME = {"chromium": "Chrome"}
# block pages by resource type. e.g. image, stylesheet
BLOCK_RESOURCE_TYPES = [
    "beacon",
    "csp_report",
    "font",
    "image",
    "imageset",
    "media",
    "object",
    "texttrack",
    'websocket',
    #  can block stylsheets and scripts, though it's not recommended:
    # 'stylesheet',
    # 'script',
    # 'xhr',
]
# block popular 3rd party resources like tracking and advertisements.
BLOCK_RESOURCE_NAMES = [
    "adzerk",
    "analytics",
    "cdn.api.twitter",
    "doubleclick",
    "exelator",
    "facebook",
    "fontawesome",
    "google",
    "google-analytics",
    "googletagmanager",
    "lit.connatix",  # <- not sure about this one
]


def clean_search_query(query):
    """Check if the first character is a digit and remove it if so.

    Some search tools (e.g., Google) will fail to return results if the
    query has a leading digit: 1. "LangCh..."

    This function will take all the text after the first double quote
    (") if a digit is detected at the beginning of the string.

    Parameters
    ----------
    query : str
        Input query that may or may not contain a leading digit.

    Returns
    -------
    str
        Cleaned query.
    """
    query = query.strip()
    if len(query) < 1:
        return query

    if not query[0].isdigit():
        return query.strip()

    if (first_quote_pos := query[:-1].find('"')) == -1:
        return query.strip()

    last_ind = -1 if query.endswith('"') else None

    # fmt: off
    return query[first_quote_pos + 1:last_ind].strip()


def compute_fn_from_url(url, make_unique=False):
    """Compute a unique file name from URL string.

    File name will always be 128 characters or less, unless the
    `make_unique` argument is set to true. In that case, the max
    length is 164 (a UUID is tagged onto the filename).

    Parameters
    ----------
    url : str
        Input URL to convert into filename.
    make_unique : bool, optional
        Option to add a UUID at the end of the file name to make it
        unique. By default, ``False``.

    Returns
    -------
    str
        Valid filename representation of the URL.
    """
    url = url.replace("https", "").replace("http", "").replace("www", "")
    url = slugify(url)
    url = url.replace("-", "").replace("_", "")

    url = _shorten_using_sha(url)

    if make_unique:
        url = f"{url}{uuid.uuid4()}".replace("-", "")

    return url


def _shorten_using_sha(fn):
    """Reduces FN to 128 characters"""
    if len(fn) <= 128:
        return fn

    out = hashlib.sha256(bytes(fn[64:], encoding="utf-8")).hexdigest()
    return f"{fn[:64]}{out}"


def write_url_doc_to_file(doc, file_content, out_dir, make_name_unique=False):
    """Write a file pulled from URL to disk.

    Parameters
    ----------
    doc : elm.web.document.Document
        Document containing meta information about the file. Must have a
        "source" key in the `metadata` dict containing the URL, which
        will be converted to a file name using
        :func:`compute_fn_from_url`.
    file_content : str | bytes
        File content, typically string text for HTML files and bytes
        for PDF file.
    out_dir : path-like
        Path to directory where file should be stored.
    make_name_unique : bool, optional
        Option to make file name unique by adding a UUID at the end of
        the file name. By default, ``False``.

    Returns
    -------
    Path
        Path to output file.
    """
    out_fn = compute_fn_from_url(
        url=doc.attrs["source"], make_unique=make_name_unique
    )
    out_fp = Path(out_dir) / f"{out_fn}.{doc.FILE_EXTENSION}"
    with open(out_fp, **doc.WRITE_KWARGS) as fh:
        fh.write(file_content)
    return out_fp


@asynccontextmanager
async def pw_page(browser, intercept_routes=False, stealth_config=None,
                  ignore_https_errors=False, timeout=30000):
    """Create new page from playwright browser context

    Parameters
    ----------
    browser : :class:`playwright.Browser`
        A playwright browser instance.
    intercept_routes : bool, default=False
        Option to intercept all requests and abort blocked ones.
        Be default, ``False``.
    stealth_config : :class:`playwright_stealth.StealthConfig`, optional
        Optional playwright stealth configuration object.
        By default, ``None``, which uses all the default stealth
        options.
    ignore_https_errors : bool, default=False
        Option to ignore https errors (i.e. SSL cert errors). This is
        not generally safe to do - you are susceptible to MITM attacks.
        However, if you are doing a simple scrape without providing
        any sensitive information (which you probably shouldn't be doing
        programmatically anyways), then it's probably ok to ignore these
        errors. By default, ``False``.
    timeout : int, default=30,000
        Default navigation and page load timeout (in milliseconds) to
        assign to page instance. By default, ``30_000``.

    Yields
    ------
    :class:`playwright.Page`
        A new page that can be used for visiting websites.
    """
    browser_type = _BT_RENAME.get(browser.browser_type.name, "random")
    ck = PWKwargs.context_kwargs(browser_type=browser_type,
                                 ignore_https_errors=ignore_https_errors)

    context = await browser.new_context(**ck)

    try:
        logger.trace("Loading browser page")
        page = await context.new_page()
        page.set_default_navigation_timeout(timeout)
        page.set_default_timeout(timeout)

        await page.set_extra_http_headers(DEFAULT_HEADERS)
        for script in PWKwargs.stealth_scripts():
            await page.add_init_script(path=script)

        await stealth_async(page, stealth_config)
        if intercept_routes:
            logger.trace("Intercepting requests and aborting blocked ones")
            await page.route("**/*", _intercept_route)
        yield page
    finally:
        await context.close()


async def _intercept_route(route):  # pragma: no cover
    """intercept all requests and abort blocked ones

    Source: https://scrapfly.io/blog/how-to-block-resources-in-playwright/
    """
    if route.request.resource_type in BLOCK_RESOURCE_TYPES:
        return await route.abort()

    if any(key in route.request.url for key in BLOCK_RESOURCE_NAMES):
        return await route.abort()

    return await route.continue_()


async def filter_documents(
    documents, validation_coroutine, task_name=None, **kwargs
):
    """Filter documents by applying a filter function to each.

    Parameters
    ----------
    documents : iter of :class:`elm.web.document.BaseDocument`
        Iterable of documents to filter.
    validation_coroutine : coroutine
        A coroutine that returns ``False`` if the document should be
        discarded and ``True`` otherwise. This function should take a
        single :class:`elm.web.document.BaseDocument` instance as the
        first argument. The function may have other arguments, which
        will be passed down using `**kwargs`.
    task_name : str, optional
        Optional task name to use in :func:`asyncio.create_task`.
        By default, ``None``.
    **kwargs
        Keyword-argument pairs to pass to `validation_coroutine`. This
        should not include the document instance itself, which will be
        independently passed in as the first argument.

    Returns
    -------
    list of :class:`elm.web.document.BaseDocument`
        List of documents that passed the validation check, sorted by
        text length, with PDF documents taking the highest precedence.
    """
    searchers = [
        asyncio.create_task(
            validation_coroutine(doc, **kwargs), name=task_name
        )
        for doc in documents
    ]
    output = await asyncio.gather(*searchers)
    filtered_docs = [doc for doc, check in zip(documents, output) if check]
    return sorted(
        filtered_docs,
        key=lambda doc: (not isinstance(doc, PDFDocument), len(doc.text)),
    )


class PWKwargs:
    """Class to compile Playwright launch and context arguments"""

    _PE = PlaywrightEngine(stealth=True)
    SKIP_SCRIPS = []
    """List of scrapling stealth script names to skip"""

    @classmethod
    def launch_kwargs(cls):
        """dict: kwargs to use for `playwright.chromium.launch()`"""
        return deepcopy(cls._PE._PlaywrightEngine__launch_kwargs())

    @classmethod
    def context_kwargs(cls, browser_type, ignore_https_errors=False):
        """dict: kwargs to use for `browser.new_context()`"""
        ck = deepcopy(cls._PE._PlaywrightEngine__context_kwargs())

        logger.trace("Loading browser context for browser type %r",
                     browser_type)
        ua = UserAgent(browsers=[browser_type],
                       platforms=["desktop", "mobile"]).random
        logger.trace("User agent is:\n\t- %s", ua)

        vp = {"width": randint(800, 1400), "height": randint(800, 1400)}
        logger.trace("Screen size is:\n\t- %r", vp)

        ck.update({"base_url": "http://127.0.0.1:443",
                   "device_scale_factor": uniform(0.8, 1.2),
                   "extra_http_headers": DEFAULT_HEADERS,
                   "user_agent": ua,
                   "viewport": vp,
                   "screen": vp,
                   "ignore_https_errors": ignore_https_errors})
        return ck


    @classmethod
    def stealth_scripts(cls):
        """iterator: Iterator of scrapling scripts to use for stealth"""
        scripts = deepcopy(cls._PE._PlaywrightEngine__stealth_scripts())
        for script in scripts:
            if any(name in script for name in cls.SKIP_SCRIPS):
                continue
            yield script
