# -*- coding: utf-8 -*-
"""ELM Ordinance function to apply ordinance extraction on a document """
import logging
from warnings import warn

from elm.water_rights.extraction.parse import StructuredOrdinanceParser


logger = logging.getLogger(__name__)


async def extract_ordinance_values(wizard, location, **kwargs):
    """Extract ordinance values for a district from a temporary vector store.

    Parameters
    ----------
    vector_store : # TODO: specify type, finish docstring
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

    parser = StructuredOrdinanceParser(**kwargs)
    values  = await parser.parse(wizard=wizard, location=location)

    return values 
