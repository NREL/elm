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
    text_splitter : obj
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


async def extract_ordinance_text_with_llm(doc, text_splitter, extractor):
    """Extract ordinance text from document using LLM.

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
        key, you will get an error.
    text_splitter : obj
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    extractor : elm.ords.extraction.ordinance.OrdinanceExtractor
        Instance of `~elm.ords.extraction.ordinance.OrdinanceExtractor`
        used for ordinance text extraction.

    Returns
    -------
    elm.web.document.BaseDocument
        Document that has been parsed for ordinance text. The results of
        the extraction are stored in the document's metadata. In
        particular, the metadata will contain a
        ``"cleaned_ordinance_text"`` key that will contain the cleaned
        ordinance text.
    """
    text_chunks = text_splitter.split_text(doc.metadata["ordinance_text"])
    ordinance_text = await extractor.check_for_restrictions(text_chunks)
    doc.metadata["restrictions_ordinance_text"] = ordinance_text

    text_chunks = text_splitter.split_text(ordinance_text)
    ordinance_text = await extractor.check_for_correct_size(text_chunks)
    doc.metadata["cleaned_ordinance_text"] = ordinance_text

    return doc


async def extract_ordinance_text_with_ngram_validation(
    doc,
    text_splitter,
    n=4,
    num_extraction_attempts=3,
    ngram_fraction_threshold=0.95,
    **kwargs,
):
    """Extract ordinance text for a single document with known ord info.

    This extraction includes an "ngram" check, which attempts to detect
    wether or not the cleaned text was extracted from the original
    ordinance text. The processing will attempt to re-extract the text
    if the validation does not pass a certain threshold until the
    maximum number of attempts is reached. If the text still does not
    pass validation at this point, there is a good chance that the LLM
    hallucinated parts of the output text, so caution should be taken.

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
    text_splitter : obj
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    n : int, optional
        Number of words to include per ngram for the ngram validation,
        which helps ensure that the LLM did not hallucinate.
        By default, ``4``.
    num_extraction_attempts : int, optional
        Number of extraction attempts before returning text that did not
        pass the ngram check. If the processing exceeds this value,
        there is a good chance that the LLM hallucinated parts of the
        output text. Cannot be negative or 0. By default, ``3``.
    ngram_fraction_threshold : float, optional
        Fraction of ngrams in the cleaned text that are also found in
        the original ordinance text for the extraction to be considered
        successful. Should be a value between 0 and 1 (inclusive).
        By default, ``0.95``.
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
    if not doc.metadata.get("ordinance_text"):
        msg = (
            "Input document has no 'ordinance_text' key or string does not "
            "contain information. Please run `check_for_ordinance_info` "
            "prior to calling this method."
        )
        logger.warning(msg)
        warn(msg, UserWarning)
        return doc

    llm_caller = LLMCaller(**kwargs)
    extractor = OrdinanceExtractor(llm_caller)

    doc = await _extract_with_ngram_check(
        doc,
        text_splitter,
        extractor,
        n=max(1, n),
        num_tries=max(1, num_extraction_attempts),
        ngram_fraction_threshold=max(0, min(1, ngram_fraction_threshold)),
    )

    return doc


async def _extract_with_ngram_check(
    doc,
    text_splitter,
    extractor,
    n=4,
    num_tries=3,
    ngram_fraction_threshold=0.95,
):
    """Extract ordinance info from doc and validate using ngrams."""
    from elm.ords.extraction.ngrams import sentence_ngram_containment

    source = doc.metadata.get("source", "Unknown")
    og_text = doc.metadata["ordinance_text"]
    if not og_text:
        msg = (
            "Document missing original ordinance text! No extraction "
            "performed (Document source: %s)",
            source,
        )
        logger.warning(msg)
        warn(msg, UserWarning)
        return doc

    best_score = 0
    best_summary = ""
    for attempt in range(num_tries):
        doc = await extract_ordinance_text_with_llm(
            doc, text_splitter, extractor
        )
        cleaned_text = doc.metadata["cleaned_ordinance_text"]
        if not cleaned_text:
            logger.debug(
                "No cleaned text found after extraction on attempt %d "
                "for document with source %s. Retrying...",
                attempt,
                source,
            )
            continue

        ngram_frac = sentence_ngram_containment(
            original=og_text, test=cleaned_text, n=n
        )
        if ngram_frac >= ngram_fraction_threshold:
            logger.debug(
                "Document extraction passed ngram check on attempt %d "
                "with score %.2f (Document source: %s)",
                attempt + 1,
                ngram_frac,
                source,
            )
            break

        if ngram_frac > best_score:
            best_score = ngram_frac
            best_summary = cleaned_text

        logger.debug(
            "Document extraction failed ngram check on attempt %d "
            "with score %.2f (Document source: %s). Retrying...",
            attempt + 1,
            ngram_frac,
            source,
        )
    else:
        doc.metadata["cleaned_ordinance_text"] = best_summary
        msg = (
            f"Ngram check failed after {num_tries}. LLM hallucination in "
            "cleaned ordinance text is extremely likely! Proceed with "
            f"caution!! (Document source: {best_score})"
        )
        logger.warning(msg)
        warn(msg, UserWarning)

    return doc


async def extract_ordinance_values(doc, **kwargs):
    """Extract ordinance values for a single document with known ord text.

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document known to contain ordinance text. This means it must
        contain an ``"cleaned_ordinance_text"`` key in the metadata. You
        can run
        :func:`~elm.ords.extraction.apply.extract_ordinance_text_with_llm`
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
    if not doc.metadata.get("cleaned_ordinance_text"):
        msg = (
            "Input document has no 'cleaned_ordinance_text' key or string "
            "does not contain info. Please run "
            "`extract_ordinance_text_with_llm` prior to calling this method."
        )
        logger.warning(msg)
        warn(msg, UserWarning)
        return doc

    parser = StructuredOrdinanceParser(**kwargs)
    text = doc.metadata["cleaned_ordinance_text"]
    doc.metadata["ordinance_values"] = await parser.parse(text)
    return doc
