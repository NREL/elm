# -*- coding: utf-8 -*-
"""ELM Web Document class definitions"""
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property

from elm.utilities.parse import (
    combine_pages,
    clean_headers,
    html_to_text,
    remove_blank_pages,
    format_html_tables,
    replace_common_pdf_conversion_chars,
    replace_multi_dot_lines,
    remove_empty_lines_or_page_footers,
)


class BaseDocument(ABC):
    """Base ELM web document representation."""

    def __init__(self, pages, source=None):
        """

        Parameters
        ----------
        pages : iterable
            Iterable of strings, where each string is a page of a
            document.
        source : str, optional
            Optional string to identify the source for the document.
            Has no effect on parsing - for documentation purposes only.
            By default, ``None``.
        """
        self.pages = remove_blank_pages(pages)
        self.source = source

    @cached_property
    def raw_pages(self):
        """list: List of (a limited count of) raw pages"""
        if not self.pages:
            return []

        return self._raw_pages()

    @cached_property
    def text(self):
        """str: Cleaned text from document"""
        if not self.pages:
            return ""

        return self._cleaned_text()

    @abstractmethod
    def _raw_pages(self):
        """Get raw pages from document"""
        raise NotImplementedError(
            "This document does not implement a raw pages extraction function"
        )

    @abstractmethod
    def _cleaned_text(self):
        """Compute cleaned text from document"""
        raise NotImplementedError(
            "This document does not implement a pages cleaning function"
        )


class PDFDocument(BaseDocument):
    """ELM web PDF document"""

    CLEAN_HEADER_KWARGS = {
        "char_thresh": 0.6,
        "page_thresh": 0.8,
        "split_on": "\n",
        "iheaders": [0, 1, 3, -3, -2, -1],
    }
    """Default :func:`~elm.utilities.parse.clean_headers` arguments"""

    def __init__(
        self,
        pages,
        source=None,
        percent_raw_pages_to_keep=25,
        max_raw_pages=18,
        num_end_pages_to_keep=2,
        clean_header_kwargs=None,
    ):
        """

        Parameters
        ----------
        pages : iterable
            Iterable of strings, where each string is a page of a
            document.
        source : str, optional
            Optional string to identify the source for the document.
            Has no effect on parsing - for documentation purposes only.
            By default, ``None``.
        percent_raw_pages_to_keep : int, optional
            Percent of "raw" pages to keep. Useful for extracting info
            from headers/footers of a doc, which are normally stripped
            to form the "clean" text. By default, ``25``.
        max_raw_pages : int, optional
            The max number of raw pages to keep. The number of raw pages
            will never exceed the total of this value +
            `num_end_pages_to_keep`. By default, ``18``.
        num_end_pages_to_keep : int, optional
            Number of additional pages to keep from the end of the
            document. This can be useful to extract more meta info.
            The number of raw pages will never exceed the total of this
            value + `max_raw_pages`. By default, ``2``.
        clean_header_kwargs : dict, optional
            Optional dictionary of keyword-value pair arguments to pass
            to the :func:`~elm.utilities.parse.clean_headers`
            function. By default, ``None``.
        """
        super().__init__(pages, source)
        self.percent_raw_pages_to_keep = percent_raw_pages_to_keep
        self.max_raw_pages = min(len(self.pages), max_raw_pages)
        self.num_end_pages_to_keep = num_end_pages_to_keep
        self.clean_header_kwargs = deepcopy(self.CLEAN_HEADER_KWARGS)
        self.clean_header_kwargs.update(clean_header_kwargs or {})

    @cached_property
    def num_raw_pages_to_keep(self):
        """int: Number of raw pages to keep from PDF document"""
        num_to_keep = self.percent_raw_pages_to_keep / 100 * len(self.pages)
        return min(self.max_raw_pages, max(1, int(num_to_keep)))

    @cached_property
    def _last_page_index(self):
        """int: last page index (determines how many end pages to include)"""
        neg_num_extra_pages = self.num_raw_pages_to_keep - len(self.pages)
        neg_num_last_pages = max(
            -self.num_end_pages_to_keep, neg_num_extra_pages
        )
        return min(0, neg_num_last_pages)

    def _cleaned_text(self):
        """Compute cleaned text from document"""
        pages = clean_headers(deepcopy(self.pages), **self.clean_header_kwargs)
        text = combine_pages(pages)
        text = replace_common_pdf_conversion_chars(text)
        text = replace_multi_dot_lines(text)
        text = remove_empty_lines_or_page_footers(text)
        return text

    def _raw_pages(self):
        """Get raw pages from document"""
        raw_pages = [page for page in self.pages[: self.num_raw_pages_to_keep]]
        if self._last_page_index:
            raw_pages += [page for page in self.pages[self._last_page_index :]]
        return raw_pages


class HTMLDocument(BaseDocument):
    """ELM web HTML document"""

    HTML_TABLE_TO_MARKDOWN_KWARGS = {
        "floatfmt": ".5f",
        "index": True,
        "tablefmt": "psql",
    }
    """Default :func:`~elm.utilities.parse.format_html_tables` arguments"""

    def __init__(
        self,
        pages,
        source=None,
        html_table_to_markdown_kwargs=None,
        ignore_html_links=True,
    ):
        """

        Parameters
        ----------
        pages : iterable
            Iterable of strings, where each string is a page of a
            document.
        source : str, optional
            Optional string to identify the source for the document.
            Has no effect on parsing - for documentation purposes only.
            By default, ``None``.
        html_table_to_markdown_kwargs : dict, optional
            Optional dictionary of keyword-value pair arguments to pass
            to the :func:`~elm.utilities.parse.format_html_tables`
            function. By default, ``None``.
        ignore_html_links : bool, optional
            Option to ignore link in HTML text during parsing.
            By default, ``True``.
        """
        super().__init__(pages, source)
        self.html_table_to_markdown_kwargs = deepcopy(
            self.HTML_TABLE_TO_MARKDOWN_KWARGS
        )
        self.html_table_to_markdown_kwargs.update(
            html_table_to_markdown_kwargs or {}
        )
        self.ignore_html_links = ignore_html_links

    def _cleaned_text(self):
        """Compute cleaned text from document"""
        text = combine_pages(self.pages)
        text = html_to_text(text, self.ignore_html_links)
        text = format_html_tables(text, **self.html_table_to_markdown_kwargs)
        return text

    def _raw_pages(self):
        """Get raw pages from document"""
        return self.pages
