# -*- coding: utf-8 -*-
"""ELM Web HTML loading with Playwright

We use Playwright so that javascript text is rendered before we scrape.
"""
import logging

import playwright
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
        logger.debug(
            "Blocking background resource %s blocked type %s",
            route.request,
            route.request.resource_type,
        )
        return await route.abort()

    if any(key in route.request.url for key in BLOCK_RESOURCE_NAMES):
        logger.debug(
            "Blocking background resource %s blocked name %s",
            route.request,
            route.request.url,
        )
        return await route.abort()

    return await route.continue_()


async def load_html_with_pw(url, **pw_launch_kwargs):  # pragma: no cover
    """Extract HTML from URL using Playwright.

    Parameters
    ----------
    url : str
        URL to pull HTML for.
    **pw_launch_kwargs
        Keyword-value argument pairs to pass to
        :meth:`async_playwright.chromium.launch`.

    Returns
    -------
    str
        HTML from page.
    """
    try:
        text = await _load_html(url, **pw_launch_kwargs)
    except (PlaywrightError, PlaywrightTimeoutError):
        text = ""
    return text


async def _load_html(url, **pw_launch_kwargs):  # pragma: no cover
    """Load html using playwright"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(**pw_launch_kwargs)
        page = await browser.new_page()
        await page.route("**/*", _intercept_route)
        await page.goto(url)
        await page.wait_for_load_state("networkidle", timeout=90_000)
        text = await page.content()

    return text
