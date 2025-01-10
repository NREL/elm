# -*- coding: utf-8 -*-
"""ELM Web HTML loading with Playwright

We use Playwright so that javascript text is rendered before we scrape.
"""
import logging
from contextlib import AsyncExitStack

from playwright.async_api import async_playwright
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

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


async def _intercept_route(route):  # pragma: no cover
    """intercept all requests and abort blocked ones

    Source: https://scrapfly.io/blog/how-to-block-resources-in-playwright/
    """
    if route.request.resource_type in BLOCK_RESOURCE_TYPES:
        return await route.abort()

    if any(key in route.request.url for key in BLOCK_RESOURCE_NAMES):
        return await route.abort()

    return await route.continue_()


async def load_html_with_pw(  # pragma: no cover
    url, browser_semaphore=None, timeout=90_000, **pw_launch_kwargs
):
    """Extract HTML from URL using Playwright.

    Parameters
    ----------
    url : str
        URL to pull HTML for.
    browser_semaphore : asyncio.Semaphore, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    timeout : int, optional
        Maximum time to wait for page loading state time in
        milliseconds. Pass `0` to disable timeout.
        By default, ``90,000``.
    **pw_launch_kwargs
        Keyword-value argument pairs to pass to
        :meth:`async_playwright.chromium.launch`.

    Returns
    -------
    str
        HTML from page.
    """
    try:
        text = await _load_html(url, browser_semaphore=browser_semaphore,
                                timeout=timeout, **pw_launch_kwargs)
    except (PlaywrightError, PlaywrightTimeoutError):
        text = ""
    return text


async def _load_html(  # pragma: no cover
    url, browser_semaphore=None, timeout=90_000, **pw_launch_kwargs
):
    """Load html using playwright"""
    logger.trace("`_load_html` pw_launch_kwargs=%r", pw_launch_kwargs)
    logger.trace("browser_semaphore=%r", browser_semaphore)

    if browser_semaphore is None:
        browser_semaphore = AsyncExitStack()

    logger.trace("Loading HTML using playwright")
    async with async_playwright() as p, browser_semaphore:
        logger.trace("launching chromium; browser_semaphore=%r",
                     browser_semaphore)
        browser = await p.chromium.launch(**pw_launch_kwargs)
        logger.trace("Loading new page")
        page = await browser.new_page()
        logger.trace("Intercepting requests and aborting blocked ones")
        await page.route("**/*", _intercept_route)
        logger.trace("Navigating to: %r", url)
        await page.goto(url)
        logger.trace("Waiting for load with timeout: %r", timeout)
        await page.wait_for_load_state("networkidle", timeout=timeout)
        text = await page.content()

    return text
