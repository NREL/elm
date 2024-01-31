# -*- coding: utf-8 -*-
"""ELM parsing utilities."""
import io
import re
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def is_double_col(text, separator="    "):
    """Does the text look like it has multiple vertical text columns?

    Parameters
    ----------
    separator : str
        Heuristic split string to look for spaces between columns

    Returns
    -------
    out : bool
        True if more than one vertical text column
    """

    lines = text.split("\n")
    n_cols = np.zeros(len(lines))
    for i, line in enumerate(lines):
        columns = line.strip().split(separator)
        n_cols[i] = len(columns)
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
