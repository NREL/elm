# -*- coding: utf-8 -*-
"""ELM parsing utilities."""
import io
import re
import logging
from warnings import warn

import html2text
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def is_multi_col(text, separator="    ", threshold_ratio=0.35):
    """Does the text look like it has multiple vertical text columns?

    Parameters
    ----------
    text : str
        Input text, which may or may not contain multiple vertical
        columns.
    separator : str
        Heuristic split string to look for spaces between columns
    threshold_ratio : float
        Portion of lines containing the separator at which point
        the text should be classified as multi-column.

    Returns
    -------
    out : bool
        True if more than one vertical text column
    """
    lines = text.split("\n")
    total_lines = len(lines)

    gap_lines = [line for line in lines if separator in line.strip()]
    cols = len(gap_lines)

    ratio = cols / total_lines

    return ratio >= threshold_ratio


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
    return re.findall(r"<table>[\s\S]*?</table>", text)


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
        try:
            header = page.split(split_on)[ih]
        except IndexError:
            header = ""
        headers[i] = header

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

    Chars affected include ``\\r\\n``, ``\\r`` and ``\\x0c``.

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


def replace_excessive_newlines(text):
    """Replace instances of three or more newlines with ``\\n\\n``

    Parameters
    ----------
    text : str
        Text possibly containing many repeated newline characters.

    Returns
    -------
    str
        Cleaned text with only a maximum of two newlines in a row.
    """
    return re.sub(r"[\n]{3,}", "\n\n", text)


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


def read_pdf(pdf_bytes, verbose=True):
    """Read PDF contents from bytes.

    This method will automatically try to detect multi-column format
    and load the text without a physical layout in that case.

    Parameters
    ----------
    pdf_bytes : bytes
        Bytes corresponding to a PDF file.
    verbose : bool, optional
        Option to log errors during parsing. By default, ``True``.

    Returns
    -------
    iterable
        Iterable containing pages of the PDF document. This iterable
        may be empty if there was an error reading the PDF file.
    """
    import pdftotext

    try:
        pages = _load_pdf_possibly_multi_col(pdf_bytes)
    except pdftotext.Error as e:
        if verbose:
            logger.error("Failed to decode PDF content!")
            logger.exception(e)
        pages = []

    return pages


def _load_pdf_possibly_multi_col(pdf_bytes):
    """Load PDF, which may be multi-column"""
    import pdftotext

    pdf_bytes = io.BytesIO(pdf_bytes)
    pages = pdftotext.PDF(pdf_bytes, physical=True)
    if is_multi_col(combine_pages(pages)):
        pages = pdftotext.PDF(pdf_bytes, physical=False)
    return pages


def read_pdf_ocr(pdf_bytes, verbose=True):  # pragma: no cover
    """Read PDF contents from bytes using Optical Character recognition (OCR).

    This method attempt to read the PDF document using OCR. This is one
    of the only ways to parse a scanned PDF document. To use this
    function, you will need to install the `pytesseract` and `pdf2image`
    Modules. Installation guides here:

        - `pytesseract`:
         https://github.com/madmaze/pytesseract?tab=readme-ov-file#installation
        - `pdf2image`:
         https://github.com/Belval/pdf2image?tab=readme-ov-file#how-to-install

    Windows users may also need to apply the fix described in this
    answer before they can use pytesseract: http://tinyurl.com/v9xr4vrj

    Parameters
    ----------
    pdf_bytes : bytes
        Bytes corresponding to a PDF file.
    verbose : bool, optional
        Option to log errors during parsing. By default, ``True``.

    Returns
    -------
    iterable
        Iterable containing pages of the PDF document. This iterable
        may be empty if there was an error reading the PDF file.
    """
    try:
        pages = _load_pdf_with_pytesseract(pdf_bytes)
    except Exception as e:
        if verbose:
            logger.error("Failed to decode PDF content!")
            logger.exception(e)
        pages = []

    return pages


def _load_pdf_with_pytesseract(pdf_bytes):  # pragma: no cover
    """Load PDF bytes using Optical Character recognition (OCR)"""

    try:
        import pytesseract
    except ImportError:
        msg = (
            "Module `pytesseract` not found. Please follow these instructions "
            "to install: https://github.com/madmaze/pytesseract?"
            "tab=readme-ov-file#installation"
        )
        logger.warning(msg)
        warn(msg)
        return []

    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        msg = (
            "Module `pdf2image` not found. Please follow these instructions "
            "to install: https://github.com/Belval/pdf2image?"
            "tab=readme-ov-file#how-to-install"
        )
        logger.warning(msg)
        warn(msg)
        return []

    logger.debug(
        "Loading PDF with `tesseract_cmd` as %s",
        pytesseract.pytesseract.tesseract_cmd,
    )

    return [
        str(pytesseract.image_to_string(page_data).encode("utf-8"))
        for page_data in convert_from_bytes(bytes(pdf_bytes))
    ]
