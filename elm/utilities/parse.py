# -*- coding: utf-8 -*-
"""ELM parsing utilities."""
import io
import re
import logging

import html2text
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def is_multi_col(text, separator="    "):
    """Does the text look like it has multiple vertical text columns?

    Parameters
    ----------
    text : str
        Input text, which may or may not contain multiple vertical
        columns.
    separator : str
        Heuristic split string to look for spaces between columns

    Returns
    -------
    out : bool
        True if more than one vertical text column
    """
    n_cols = [len(line.strip().split(separator)) for line in text.split("\n")]
    return np.median(n_cols) >= 2


def remove_blank_pages(pages):
    """Remove any blank pages from the iterable.

    Parameters
    ----------
    pages : iterable
        Iterable of string objects. Objects in this iterable that do not
        contain any text will be removed.

    Returns
    -------
    list
        List of strings with content, or empty list.
    """
    return [page for page in pages if any(page.strip())]


def html_to_text(html, ignore_links=True):
    """Call to `HTML2Text` class with basic args.

    Parameters
    ----------
    html : str
        HTML text extracted from the web.
    ignore_links : bool, optional
        Option to ignore links in HTML when parsing.
        By default, ``True``.

    Returns
    -------
    str
        Text extracted from the input HTML.
    """
    h = html2text.HTML2Text()
    h.ignore_links = ignore_links
    h.ignore_images = True
    h.bypass_tables = True
    return h.handle(html)


def format_html_tables(text, **kwargs):
    """Format tables within HTML text into pretty markdown.

    Note that if pandas does not detect enough tables in the text to
    match the "<table>" tags, no replacement is performed at all.

    Parameters
    ----------
    text : str
        HTML text, possible containing tables enclosed by the
        "<table>" tag.
    **kwargs
        Keyword-arguments to pass to ``pandas.DataFrame.to_markdown``
        function. Must not contain the `"headers"` keyword (this is
        supplied internally).

    Returns
    -------
    str
        Text with HTML tables (if any) converted to markdown.
    """
    matches = _find_html_table_matches(text)
    if not matches:
        return text

    dfs = _find_dfs(text)
    if len(matches) != len(dfs):
        logger.error(
            "Found incompatible number of HTML (%d) and parsed (%d) tables! "
            "No replacement performed.",
            len(matches),
            len(dfs),
        )
        return text

    return _replace_tables_in_text(text, matches, dfs, **kwargs)


def _find_html_table_matches(text):
    """Find HTML table matches in the text"""
    return re.findall("<table>[\s\S]*?</table>", text)


def _find_dfs(text):
    """Load HTML tables from text into DataFrames"""
    return pd.read_html(io.StringIO(text))


def _replace_tables_in_text(text, matches, dfs, **kwargs):
    """Replace all items in the 'matches' input with MD tables"""
    for table_str, df in zip(matches, dfs):
        new_table_str = df.to_markdown(headers=df.columns, **kwargs)
        text = text.replace(table_str, new_table_str)
    return text


def clean_headers(
    pages,
    char_thresh=0.6,
    page_thresh=0.8,
    split_on="\n",
    iheaders=(0, 1, -2, -1),
):
    """Clean headers/footers that are duplicated across pages of a document.

    Note that this function will update the items within the `pages`
    input.

    Parameters
    ----------
    pages : list
        List of pages (as str) from document.
    char_thresh : float
        Fraction of characters in a given header that are similar
        between pages to be considered for removal
    page_thresh : float
        Fraction of pages that share the header to be considered for
        removal
    split_on : str
        Chars to split lines of a page on
    iheaders : list | tuple
        Integer indices to look for headers after splitting a page into
        lines based on split_on. This needs to go from the start of the
        page to the end.

    Returns
    -------
    out : str
        Clean text with all pages joined
    """
    logger.info("Cleaning headers")
    headers = _get_nominal_headers(pages, split_on, iheaders)
    tests = np.zeros((len(pages), len(headers)))

    for ip, page in enumerate(pages):
        for ih, header in zip(iheaders, headers):
            pheader = ""
            try:
                pheader = page.split(split_on)[ih]
            except IndexError:
                pass

            harr = header.replace(" ", "")
            parr = pheader.replace(" ", "")

            harr = harr.ljust(len(parr))
            parr = parr.ljust(len(harr))

            harr = np.array([*harr])
            parr = np.array([*parr])
            assert len(harr) == len(parr)

            test = harr == parr
            if len(test) == 0:
                test = 1.0
            else:
                test = test.sum() / len(test)

            tests[ip, ih] = test

    logger.debug("Header tests (page, iheader): \n{}".format(tests))
    tests = (tests > char_thresh).sum(axis=0) / len(pages)
    tests = tests > page_thresh
    logger.debug("Header tests (iheader,): \n{}".format(tests))

    header_inds_to_remove = {
        ind for is_header, ind in zip(tests, iheaders) if is_header
    }
    if not header_inds_to_remove:
        return pages

    for ip, page in enumerate(pages):
        page = page.split(split_on)
        if len(iheaders) >= len(page):
            continue
        pages[ip] = split_on.join(
            [
                line
                for line_ind, line in enumerate(page)
                if line_ind not in header_inds_to_remove
                and line_ind - len(page) not in header_inds_to_remove
            ]
        )

    return pages


def _get_nominal_headers(pages, split_on, iheaders):
    """Get nominal headers from a standard page.

    This function aims for a "typical" page that is likely to have a
    normal header, not the first or last.

    Parameters
    ----------
    pages : list
        List of pages (as str) from document.
    split_on : str
        Chars to split lines of a page on
    iheaders : list | tuple
        Integer indices to look for headers after splitting a page into
        lines based on split_on. This needs to go from the start of the
        page to the end.

    Returns
    -------
    headers : list
        List of headers where each entry is a string header
    """

    headers = [None] * len(iheaders)
    page_lens = np.array([len(p) for p in pages])
    median_len = np.median(page_lens)
    ipage = np.argmin(np.abs(page_lens - median_len))
    page = pages[ipage]
    for i, ih in enumerate(iheaders):
        headers[i] = page.split(split_on)[ih]

    return headers


def combine_pages(pages):
    """Combine pages of GPT cleaned text into a single string.

    Parameters
    ----------
    pages : list
        List of pages (as str) from document.

    Returns
    -------
    full : str
        Single multi-page string
    """
    return "\n".join(pages).replace("\n•", "-").replace("•", "-")


def replace_common_pdf_conversion_chars(text):
    """Re-format text to remove common pdf-converter chars.

    Chars affected include ``\r\n``, ``\r`` and ``\x0c``.

    Parameters
    ----------
    text : str
        Input text (presumably from pdf parser).

    Returns
    -------
    str
        Cleaned text.
    """
    return text.replace("\r\n", "\n").replace("\x0c", "").replace("\r", "\n")


def replace_multi_dot_lines(text):
    """Replace instances of three or more dots (.....) with just "..."

    Parameters
    ----------
    text : str
        Text possibly containing many repeated dots.

    Returns
    -------
    str
        Cleaned text with only three dots max in a row.
    """
    return re.sub(r"[.]{3,}", "...", text)


def remove_empty_lines_or_page_footers(text):
    """Replace empty lines (potentially with page numbers only) as newlines

    Parameters
    ----------
    text : str
        Text possibly containing empty lines and/or lines with only page
        numbers.

    Returns
    -------
    str
        Cleaned text with no empty lines.
    """
    return re.sub(r"[\n\r]+(?:\s*?\d*?\s*)[\n\r]+", "\n", text)
