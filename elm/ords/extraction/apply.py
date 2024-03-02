# -*- coding: utf-8 -*-
"""ELM Ordinance function to apply ordinance extraction on a document """
import logging
from warnings import warn

from elm.ords.llm import LLMCaller, StructuredLLMCaller
from elm.ords.extraction.date import DateExtractor
from elm.ords.extraction.ordinance import (
    OrdinanceValidator,
    OrdinanceExtractor,
)
from elm.ords.extraction.parse import StructuredOrdinanceParser


logger = logging.getLogger(__name__)


async def check_for_ordinance_info(doc, text_splitter, **kwargs):
    """Parse a single document for ordinance information.

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document potentially containing ordinance information. Note
        that if the document's metadata contains the
        ``"contains_ord_info"`` key, it will not be processed. To force
        a document to be processed by this function, remove that key
        from the documents metadata.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    **kwargs
        Keyword-value pairs used to initialize an
        `elm.ords.llm.LLMCaller` instance.

    Returns
    -------
    elm.web.document.BaseDocument
        Document that has been parsed for ordinance text. The results of
        the parsing are stored in the documents metadata. In particular,
        the metadata will contain a ``"contains_ord_info"`` key that
        will be set to ``True`` if ordinance info was found in the text,
        and ``False`` otherwise. If ``True``, the metadata will also
        contain a ``"date"`` key containing the most recent date that
        the ordinance was enacted (or a tuple of `None` if not found),
        and an ``"ordinance_text"`` key containing the ordinance text
        snippet. Note that the snippet may contain other info as well,
        but should encapsulate all of the ordinance text.
    """
    if "contains_ord_info" in doc.metadata:
        return doc

    llm_caller = StructuredLLMCaller(**kwargs)
    chunks = text_splitter.split_text(doc.text)
    validator = OrdinanceValidator(llm_caller, chunks)
    doc.metadata["contains_ord_info"] = await validator.parse()
    if doc.metadata["contains_ord_info"]:
        doc.metadata["date"] = await DateExtractor(llm_caller).parse(doc)
        doc.metadata["ordinance_text"] = validator.ordinance_text

    return doc


async def extract_ordinance_text(doc, text_splitter, **kwargs):
    """Extract ordinance text for a single document with known ord info.

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document known to contain ordinance information. This means it
        must contain an ``"ordinance_text"`` key in the metadata. You
        can run
        :func:`~elm.ords.extraction.apply.check_for_ordinance_info`
        to have this attribute populated automatically for documents
        that are found to contain ordinance data. Note that if the
        document's metadata does not contain the ``"ordinance_text"``
        key, it will not be processed.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    **kwargs
        Keyword-value pairs used to initialize an
        `elm.ords.llm.LLMCaller` instance.

    Returns
    -------
    elm.web.document.BaseDocument
        Document that has been parsed for ordinance text. The results of
        the extraction are stored in the document's metadata. In
        particular, the metadata will contain a
        ``"cleaned_ordinance_text"`` key that will contain the cleaned
        ordinance text.
    """
    if "ordinance_text" not in doc.metadata:
        msg = (
            "Input document has no 'ordinance_text' key in metadata. "
            "Please run `check_for_ordinance_info` prior to calling this "
            "method."
        )
        logger.warning(msg)
        warn(msg, UserWarning)
        return doc

    llm_caller = LLMCaller(**kwargs)
    extractor = OrdinanceExtractor(llm_caller)

    text_chunks = text_splitter.split_text(doc.metadata["ordinance_text"])
    ordinance_text = await extractor.check_for_restrictions(text_chunks)
    doc.metadata["restrictions_ordinance_text"] = ordinance_text

    text_chunks = text_splitter.split_text(ordinance_text)
    ordinance_text = await extractor.check_for_correct_size(text_chunks)
    doc.metadata["cleaned_ordinance_text"] = ordinance_text

    return doc


async def extract_ordinance_values(doc, **kwargs):
    """Extract ordinance values for a single document with known ord text.

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document known to contain ordinance text. This means it must
        contain an ``"cleaned_ordinance_text"`` key in the metadata. You
        can run
        :func:`~elm.ords.extraction.apply.extract_ordinance_text`
        to have this attribute populated automatically for documents
        that are found to contain ordinance data. Note that if the
        document's metadata does not contain the
        ``"cleaned_ordinance_text"`` key, it will not be processed.
    **kwargs
        Keyword-value pairs used to initialize an
        `elm.ords.llm.LLMCaller` instance.

    Returns
    -------
    elm.web.document.BaseDocument
        Document that has been parsed for ordinance values. The results
        of the extraction are stored in the document's metadata. In
        particular, the metadata will contain an ``"ordinance_values"``
        key that will contain the DataFame with ordinance values.
    """
    if "cleaned_ordinance_text" not in doc.metadata:
        msg = (
            "Input document has no 'cleaned_ordinance_text' key in metadata. "
            "Please run `extract_ordinance_text` prior to calling this "
            "method."
        )
        logger.warning(msg)
        warn(msg, UserWarning)
        return doc

    parser = StructuredOrdinanceParser(**kwargs)
    text = doc.metadata["cleaned_ordinance_text"]
    doc.metadata["ordinance_values"] = await parser.parse(text)
    return doc
