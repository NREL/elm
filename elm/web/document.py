# -*- coding: utf-8 -*-
"""ELM Web Document class definitions"""
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property
import logging

import pandas as pd

from elm.utilities.parse import (
    combine_pages,
    clean_headers,
    html_to_text,
    remove_blank_pages,
    format_html_tables,
    read_pdf,
    read_pdf_ocr,
    replace_common_pdf_conversion_chars,
    replace_multi_dot_lines,
    remove_empty_lines_or_page_footers,
)


logger = logging.getLogger(__name__)


class BaseDocument(ABC):
    """Base ELM web document representation

    Purpose:
        Track document content and perform minor processing on it.
    Responsibilities:
        1. Store "raw" document text.
        2. Compute "cleaned" text, which combines pages, strips HTML,
           and formats tables.
        3. Track pages and other document metadata.
    Key Relationships:
        Created by :class:`~elm.web.file_loader.AsyncFileLoader` and
        used all over ordinance code.

    .. end desc
    """

    def __init__(self, pages, attrs=None):
        """

        Parameters
        ----------
        pages : iterable
            Iterable of strings, where each string is a page of a
            document.
        attrs : dict, optional
            Optional dict containing metadata for the document.
            By default, ``None``.
        """
        self.pages = remove_blank_pages(pages)
        self.attrs = attrs or {}

    def __repr__(self):
        header = (f"{self.__class__.__name__} with {len(self.pages):,} "
                  "pages\nAttrs:")
        if not self.attrs:
            return f"{header} None"

        attrs = {}
        for k, v in self.attrs.items():
            if isinstance(v, pd.DataFrame):
                v = f"DataFrame with {len(v):,} rows"
            attrs[k] = v

        indent = max(len(k) for k in attrs) + 2
        attrs = "\n".join([f"{k:>{indent}}:\t{v}"
                           for k, v in attrs.items()])
        return f"{header}\n{attrs}"

    @property
    def empty(self):
        """bool: ``True`` if the document contains no pages."""
        return not any(_non_empty_pages(self.text.split("\n")))

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

    @property
    @abstractmethod
    def WRITE_KWARGS(self):
        """dict: Dict of kwargs to pass to `open` when writing this doc."""
        raise NotImplementedError

    @property
    @abstractmethod
    def FILE_EXTENSION(self):
        """str: Cleaned document file extension."""
        raise NotImplementedError


class PDFDocument(BaseDocument):
    """ELM web PDF document"""

    CLEAN_HEADER_KWARGS = {
        "char_thresh": 0.6,
        "page_thresh": 0.8,
        "split_on": "\n",
        "iheaders": [0, 1, 3, -3, -2, -1],
    }
    """Default :func:`~elm.utilities.parse.clean_headers` arguments"""
    WRITE_KWARGS = {"mode": "wb"}
    FILE_EXTENSION = "pdf"

    def __init__(
        self,
        pages,
        attrs=None,
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
        attrs : str, optional
            Optional dict containing metadata for the document.
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
        super().__init__(pages, attrs=attrs)
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

    # pylint: disable=unnecessary-comprehension
    # fmt: off
    def _raw_pages(self):
        """Get raw pages from document"""
        raw_pages = [page for page in self.pages[:self.num_raw_pages_to_keep]]
        if self._last_page_index:
            raw_pages += [page for page in self.pages[self._last_page_index:]]
        return raw_pages

    @classmethod
    def from_file(cls, fp, **init_kwargs):
        """Initialize a PDFDocument object from a .pdf file on disk. This
        method will try to use pdftotext (a poppler utility) and then
        OCR with pytesseract.

        Parameters
        ----------
        fp : str
            filepath to .pdf on disk
        init_kwargs : dict
            Optional kwargs for PDFDocument Initialization

        Returns
        -------
        out : PDFDocument
            Initialized PDFDocument class from input fp
        """

        with open(fp, 'rb') as f:
            pages = read_pdf(f.read())

        pages = list(_non_empty_pages(pages))
        if pages:
            return cls(pages, **init_kwargs)

        # fallback to OCR with pytesseract if no pages have more than 10
        # chars. Typical scanned document only has weird ascii per page.
        with open(fp, 'rb') as f:
            pages = read_pdf_ocr(f.read())

        pages = list(_non_empty_pages(pages))
        if not any(pages):
            msg = f'Could not get text from pdf: {fp}'
            logger.error(msg)
            raise RuntimeError(msg)

        return cls(pages, **init_kwargs)


class HTMLDocument(BaseDocument):
    """ELM web HTML document"""

    HTML_TABLE_TO_MARKDOWN_KWARGS = {
        "floatfmt": ".5f",
        "index": True,
        "tablefmt": "psql",
    }
    """Default :func:`~elm.utilities.parse.format_html_tables` arguments"""
    WRITE_KWARGS = {"mode": "w", "encoding": "utf-8"}
    FILE_EXTENSION = "txt"

    def __init__(
        self,
        pages,
        attrs=None,
        html_table_to_markdown_kwargs=None,
        ignore_html_links=True,
        text_splitter=None,
    ):
        """

        Parameters
        ----------
        pages : iterable
            Iterable of strings, where each string is a page of a
            document.
        attrs : dict, optional
            Optional dict containing metadata for the document.
            By default, ``None``.
        html_table_to_markdown_kwargs : dict, optional
            Optional dictionary of keyword-value pair arguments to pass
            to the :func:`~elm.utilities.parse.format_html_tables`
            function. By default, ``None``.
        ignore_html_links : bool, optional
            Option to ignore link in HTML text during parsing.
            By default, ``True``.
        text_splitter : obj, optional
            Instance of an object that implements a `split_text` method.
            The method should take text as input (str) and return a list
            of text chunks. The raw pages will be passed through this
            splitter to create raw pages for this document. Langchain's
            text splitters should work for this input.
            By default, ``None``, which means the original pages input
            becomes the raw pages attribute.
        """
        super().__init__(pages, attrs=attrs)
        self.html_table_to_markdown_kwargs = deepcopy(
            self.HTML_TABLE_TO_MARKDOWN_KWARGS
        )
        self.html_table_to_markdown_kwargs.update(
            html_table_to_markdown_kwargs or {}
        )
        self.ignore_html_links = ignore_html_links
        self.text_splitter = text_splitter

    def _cleaned_text(self):
        """Compute cleaned text from document"""
        text = combine_pages(self.pages)
        text = html_to_text(text, self.ignore_html_links)
        text = format_html_tables(text, **self.html_table_to_markdown_kwargs)
        return text

    def _raw_pages(self):
        """Get raw pages from document"""
        if self.text_splitter is None:
            return self.pages
        return self.text_splitter.split_text("\n\n".join(self.pages))


def _non_empty_pages(pages):
    """Return all pages with more than 10 chars"""
    return filter(
        lambda page: re.search('[a-zA-Z]', page) and len(page) > 10, pages
    )
