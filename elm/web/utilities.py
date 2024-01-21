# -*- coding: utf-8 -*-
"""ELM Web Scraping utilities."""


def clean_search_query(query):
    """Check if the first character is a digit and remove it if so.

    Some search tools (e.g., Google) will fail to return results if the
    query has a leading digit: 1. "LangCh..."

    This function will take all teh text after the first double quote
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
    query = query[first_quote_pos + 1 : last_ind]

    return query.strip()
