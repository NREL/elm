# -*- coding: utf-8 -*-
"""ELM Ordinance function to apply ordinance extraction on a document """
import logging

from elm.ords.llm import StructuredLLMCaller
from elm.ords.extraction.date import DateExtractor
from elm.water_rights.extraction.ordinance import OrdinanceValidator

from elm.water_rights.extraction.parse import StructuredOrdinanceParser


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
    if "contains_ord_info" in doc.attrs:
        return doc

    llm_caller = StructuredLLMCaller(**kwargs)
    chunks = text_splitter.split_text(doc.text)
    validator = OrdinanceValidator(llm_caller, chunks)
    doc.attrs["contains_ord_info"] = await validator.parse()
    if doc.attrs["contains_ord_info"]:
        doc.attrs["date"] = await DateExtractor(llm_caller).parse(doc)
        doc.attrs["ordinance_text"] = validator.ordinance_text

    return doc

async def extract_ordinance_values(wizard, location, **kwargs):
    """Extract ordinance values from a temporary vector store.

    Parameters
    ----------
    wizard : elm.wizard.EnergyWizard
        Instance of the EnergyWizard class used for RAG.
    location : str
        Name of the groundwater conservation district or county.
    **kwargs
        Keyword-value pairs used to initialize an
        `elm.ords.llm.LLMCaller` instance.

    Returns
    -------
    values : dict
        Dictionary of values extracted from the vector store.
    """

    parser = StructuredOrdinanceParser(wizard=wizard, location=location,
                                       **kwargs)
    values  = await parser.parse(location=location)

    return values
