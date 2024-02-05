# -*- coding: utf-8 -*-
"""ELM Web Scraping utilities."""
import uuid
import hashlib

from slugify import slugify


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
