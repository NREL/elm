# -*- coding: utf-8 -*-
"""ELM Web Scraping utilities."""
import uuid
import hashlib
import logging
import asyncio
from pathlib import Path
from copy import deepcopy
from random import randint, choice
from contextlib import asynccontextmanager

import httpx
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
_BT_RENAME = {"chromium": "Chrome", "firefox": "Firefox",
              "webkit": "Safari"}
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


async def get_redirected_url(url, **kwargs):
    """Get the final URL after following redirects.

    Parameters
    ----------
    url : str
        URL to check for redirects.
    **kwargs
        Keyword-value arguments to pass to the
        :class:`httpx.AsyncClient` client.

    Returns
    -------
    str
        The final URL after following redirects, or the original URL if
        no redirects are found or an error occurs.
    """
    kwargs["follow_redirects"] = True
    try:
        async with httpx.AsyncClient(**kwargs) as client:
            response = await client.head(url)
            return str(response.url)
    except httpx.RequestError as e:
        return url


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
                  ignore_https_errors=False, timeout=30000,
                  use_scrapling_stealth=False):
    """Create new page from playwright browser context

    Parameters
    ----------
    browser : :class:`playwright.Browser`
        A playwright browser instance.
    intercept_routes : bool, default=False
        Option to intercept all requests and abort blocked ones.
        Be default, ``False``.
    stealth_config : :class:`playwright_stealth.Stealth`, optional
        Optional tf-playwright-stealth StealthConfig configuration
        object instance. By default, ``None``, which uses all the
        default stealth options.
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
    use_scrapling_stealth : bool, default=False
        Option to use scrapling stealth scripts instead of
        tf-playwright-stealth. If set to ``True``, the `stealth_config`
        argument will be ignored. By default, ``False``.

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
        if use_scrapling_stealth:
            logger.trace("Using scrapling stealth scripts")
            for script in PWKwargs.stealth_scripts():
                await page.add_init_script(path=script)
        else:
            logger.trace("Using tf-playwright-stealth stealth scripts")
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

    USE_REALISTIC_VIEWPORTS = True
    """bool: Use realistic viewport sizes for the browser context"""

    _VIEWPORTS = {
        "desktop": [
            {"width": 1080, "height": 600, "device_scale_factor": 1},
            {"width": 1280, "height": 1024, "device_scale_factor": 1},
            {"width": 1366, "height": 768, "device_scale_factor": 1},
            {"width": 1440, "height": 900, "device_scale_factor": 1},
            {"width": 1536, "height": 864, "device_scale_factor": 1},
            {"width": 1600, "height": 900, "device_scale_factor": 1},
            {"width": 1680, "height": 1050, "device_scale_factor": 1},
            {"width": 1920, "height": 1080, "device_scale_factor": 1},
            {"width": 2560, "height": 1440, "device_scale_factor": 1.25},
            {"width": 3840, "height": 2160, "device_scale_factor": 2},
        ],
        "mobile": [
            # iPhone 6/7/8
            {"width": 375, "height": 667, "device_scale_factor": 2},
            # iPhone X/11 Pro
            {"width": 375, "height": 812, "device_scale_factor": 3},
            # iPhone 12/13/14
            {"width": 390, "height": 844, "device_scale_factor": 3},
            # iPhone XR/11
            {"width": 414, "height": 896, "device_scale_factor": 2},
            # iPhone 14 Pro Max
            {"width": 428, "height": 926, "device_scale_factor": 3},
            # Galaxy S9
            {"width": 360, "height": 800, "device_scale_factor": 2.5},
            # Pixel 7 Pro
            {"width": 412, "height": 915, "device_scale_factor": 2.625},
            # Pixel 6
            {"width": 393, "height": 873, "device_scale_factor": 2.75},
            # Pixel 5a
            {"width": 360, "height": 780, "device_scale_factor": 3},
        ],
        "tablet": [
            # iPad portrait
            {"width": 768, "height": 1024, "device_scale_factor": 2},
            # iPad landscape
            {"width": 1024, "height": 768, "device_scale_factor": 2},
            # Android tablet
            {"width": 800, "height": 1280, "device_scale_factor": 1.5},
            # Samsung Tab
            {"width": 962, "height": 601, "device_scale_factor": 1.5},
        ]
    }

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

        platforms = (["desktop"]
                     if cls.USE_REALISTIC_VIEWPORTS
                     else ["desktop", "mobile", "tablet"])
        ua_info = (UserAgent(browsers=[browser_type], platforms=platforms)
                   .getBrowser("random"))
        logger.trace("User agent is:\n\t- %s", ua_info["useragent"])
        platform_type = ua_info["type"].casefold()

        if cls.USE_REALISTIC_VIEWPORTS:
            vp = deepcopy(choice(cls._VIEWPORTS[platform_type]))
            dsf = vp.pop("device_scale_factor")
        else:
            vp = {"width": randint(800, 1400), "height": randint(800, 1400)}
            dsf = choice([1, 1.25, 1.5, 2, 2.5, 2.625, 2.75, 3])
        logger.trace("Screen size is:\n\t- %r", vp)

        ck.update({"base_url": "http://127.0.0.1:443",
                   "device_scale_factor": dsf,
                   "extra_http_headers": DEFAULT_HEADERS,
                   "user_agent": ua_info["useragent"],
                   "viewport": vp,
                   "screen": vp,
                   "is_mobile": platform_type in {"mobile", "tablet"},
                   "has_touch": platform_type in {"mobile", "tablet"},
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
