# -*- coding: utf-8 -*-
"""ELM Ordinance function to apply ordinance extraction on a document """
from elm.ords.extraction.date import DateExtractor
from elm.ords.extraction.ordinance import OrdinanceValidator


async def check_for_ordinance_info(doc, llm_caller, text_splitter):
    """Parse a single document for ordinance information.

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document potentially containing ordinance information. Note
        that if the documents metadata contains the
        ``"contains_ord_info"`` key, it will not be processed. To force
        a document to be processed by this function, remove that key
        from the documents metadata.
    llm_caller : elm.ords.llm.StructuredLLMCaller
        StructuredLLMCaller instance used for validation and checking
        the contents of the document.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.

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

    chunks = text_splitter.split_text(doc.text)
    extractor = OrdinanceValidator(llm_caller, chunks)
    doc.metadata["contains_ord_info"] = await extractor.parse()
    if doc.metadata["contains_ord_info"]:
        doc.metadata["date"] = await DateExtractor(llm_caller).parse(doc)
        doc.metadata["ordinance_text"] = extractor.ordinance_text

    return doc
