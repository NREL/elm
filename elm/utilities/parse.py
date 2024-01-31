# -*- coding: utf-8 -*-
"""ELM parsing utilities."""
import numpy as np


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
