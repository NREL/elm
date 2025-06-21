# -*- coding: utf-8 -*-
"""ELM Web HTML loading with Playwright

We use Playwright so that javascript text is rendered before we scrape.
"""
import logging
from contextlib import AsyncExitStack

from rebrowser_playwright.async_api import async_playwright
from rebrowser_playwright.async_api import Error as PlaywrightError
from rebrowser_playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError
)

from elm.web.utilities import pw_page, PWKwargs


logger = logging.getLogger(__name__)


async def load_html_with_pw(url, browser_semaphore=None, # pragma: no cover
                            timeout=90_000, use_scrapling_stealth=False,
                            **pw_launch_kwargs):
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
    use_scrapling_stealth : bool, default=False
        Option to use scrapling stealth scripts instead of
        tf-playwright-stealth. By default, ``False``.
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
                                timeout=timeout,
                                use_scrapling_stealth=use_scrapling_stealth,
                                **pw_launch_kwargs)
    except (PlaywrightError, PlaywrightTimeoutError):
        text = ""
    return text


async def _load_html( url, browser_semaphore=None, # pragma: no cover
                     timeout=90_000, use_scrapling_stealth=False,
                     **pw_launch_kwargs):
    """Load html using playwright"""
    logger.trace("`_load_html` pw_launch_kwargs=%r", pw_launch_kwargs)
    logger.trace("browser_semaphore=%r", browser_semaphore)

    if browser_semaphore is None:
        browser_semaphore = AsyncExitStack()

    launch_kwargs = PWKwargs.launch_kwargs()
    launch_kwargs.update(pw_launch_kwargs)

    logger.trace("Loading HTML using playwright")
    async with async_playwright() as p, browser_semaphore:
        logger.trace("launching chromium; browser_semaphore=%r",
                     browser_semaphore)
        browser = await p.chromium.launch(**launch_kwargs)
        page_kwargs = {"browser": browser, "intercept_routes": True,
                       "timeout": timeout, "ignore_https_errors": True,
                       "use_scrapling_stealth": use_scrapling_stealth}
        async with pw_page(**page_kwargs) as page:
            logger.trace("Navigating to: %r", url)
            await page.goto(url)
            logger.trace("Waiting for load with timeout: %r", timeout)
            await page.wait_for_load_state("networkidle", timeout=timeout)
            text = await page.content()

    return text
