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
