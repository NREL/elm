# -*- coding: utf-8 -*-
"""ELM Ordinance Location Validation logic

These are primarily used to validate that a legal document applies to a
particular location.
"""
import asyncio
import logging

from elm.ords.extraction.ngrams import convert_text_to_sentence_ngrams
from elm.ords.validation.location import FixedMessageValidator


logger = logging.getLogger(__name__)


class CountyJurisdictionValidator(FixedMessageValidator):
    """Validator that checks wether text applies at the county level."""

    SYSTEM_MESSAGE = (
        "You extract structured data from legal text. Return "
        "your answer in JSON format. Your JSON file must include exactly "
        "three keys. The first key is 'x', which is a boolean that is set to "
        "`True` if the text excerpt explicitly mentions that the regulations "
        "within apply to a jurisdiction scope other than {county} "
        "(i.e. they apply to a subdivision like a township or a city, or "
        "they apply more broadly, like to a state, the full country, or an "
        "aquifer management zone). `False` if the regulations in the text "
        "apply at the {county} level or if there is not enough information to "
        "determine the answer. The second key is 'y', which is a boolean "
        "that is set to `True` if the text excerpt explicitly mentions that "
        "the regulations within apply to more than one groundwater conservation "
        "district. `False` if the regulations in the text excerpt apply to a "
        "single groundwater conservation district only or if there is not enough "
        "information to determine the answer. The third key is 'explanation', "
        "which is a string that contains a short explanation if you chose `True` "
        "for any answers above."
    )

    def _parse_output(self, props):
        """Parse LLM response and return `True` if the document passes."""
        logger.debug(
            "Parsing county jurisdiction validation output:\n\t%s", props
        )
        check_vars = ("x", "y")
        return not any(props.get(var) for var in check_vars)


class CountyNameValidator(FixedMessageValidator):
    """Validator that checks whether text applies to a particular county."""
    # TODO: there are only some minor differences here ('gcd' instead of 'county')
    # minor enough to just import?
    SYSTEM_MESSAGE = (
        "You extract structured data from legal text. Return "
        "your answer in JSON format. Your JSON file must include exactly "
        "three keys. The first key is 'wrong_county', which is a boolean that "
        "is set to `True` if the legal text is not for {county}. Do "
        "not infer based on any information about any US state, city, "
        "township, or otherwise and keep in mind that aquifer management zones "
        "should not be considered groundwater conservation districts. "
        "`False` if the text applies to {county} or if there is not enough "
        "information to determine the answer. The second key is 'wrong_state', "
        "which is a boolean that is set to `True` if the legal text is not for "
        "a conservation district in the state of {state}. Do not infer based "
        "on any information about any US county, city, township, or otherwise. "
        "`False` if the text applies to a conservation district in the state of "
        "{state} or if there is not enough information to determine the answer. "
        "The third key is 'explanation', which is a string that contains a short "
        "explanation if you chose `True` for any answers above."
    )

    def _parse_output(self, props):
        """Parse LLM response and return `True` if the document passes."""
        logger.debug("Parsing county validation output:\n\t%s", props)
        check_vars = ("wrong_county", "wrong_state")
        return not any(props.get(var) for var in check_vars)


class CountyValidator:
    """ELM Ords County validator.

    Combines the logic of several validators into a single class.

    Purpose:
        Determine wether a document pertains to a specific county.
    Responsibilities:
        1. Use a combination of heuristics and LLM queries to determine
           wether or not a document pertains to a particular county.
    Key Relationships:
        Uses a :class:`~elm.ords.llm.calling.StructuredLLMCaller` for
        LLM queries and delegates sub-validation to
        :class:`~elm.ords.validation.location.CountyNameValidator`,
        :class:`~elm.ords.validation.location.CountyJurisdictionValidator`,
        and :class:`~elm.ords.validation.location.URLValidator`.

    .. end desc
    """

    def __init__(self, structured_llm_caller, score_thresh=0.8):
        """

        Parameters
        ----------
        structured_llm_caller : :class:`elm.ords.llm.StructuredLLMCaller`
            StructuredLLMCaller instance. Used for structured validation
            queries.
        score_thresh : float, optional
            Score threshold to exceed when voting on content from raw
            pages. By default, ``0.8``.
        """
        self.score_thresh = score_thresh
        self.cn_validator = CountyNameValidator(structured_llm_caller)
        self.cj_validator = CountyJurisdictionValidator(structured_llm_caller)

    async def check(self, doc, county, county_acronym, state):
        """Check if the document belongs to the county.

        Parameters
        ----------
        doc : :class:`elm.web.document.BaseDocument`
            Document instance. Should contain a "source" key in the
            metadata that contains a URL (used for the URL validation
            check). Raw content will be parsed for county name and
            correct jurisdiction.
        county : str
            County that document should belong to.
        state : str
            State corresponding to `county` input.

        Returns
        -------
        bool
            `True` if the doc contents pertain to the input county.
            `False` otherwise.
        """
        source = doc.attrs.get("source")
        logger.debug(
            "Validating document from source: %s", source or "Unknown"
        )
        logger.debug("Checking for correct for jurisdiction...")
        jurisdiction_is_county = await _validator_check_for_doc(
            validator=self.cj_validator,
            doc=doc,
            score_thresh=self.score_thresh,
            county=county,
        )
        if not jurisdiction_is_county:
            return False

        logger.debug(
            "Checking text for county name (heuristic; URL: %s)...",
            source or "Unknown",
        )
        correct_county_heuristic = _heuristic_check_for_county_and_state(
            doc, county, county_acronym, state
        )
        logger.debug(
            "Found county name in text (heuristic): %s",
            correct_county_heuristic,
        )
        if correct_county_heuristic:
            return True

        logger.debug(
            "Checking text for county name (LLM; URL: %s)...",
            source or "Unknown",
        )
        return await _validator_check_for_doc(
            validator=self.cn_validator,
            doc=doc,
            score_thresh=self.score_thresh,
            county=county,
            state=state,
        )


def _heuristic_check_for_county_and_state(doc, county, county_acronym, state):
    """Check if county and state names are in doc"""
    county = county.lower().replace(" county", "")
    return any(
        any(
            ((county.lower() in fg or county_acronym.lower() in fg) and
             state.lower() in fg)
            for fg in convert_text_to_sentence_ngrams(t.lower(), 5)
        )
        for t in doc.pages
    )


async def _validator_check_for_doc(validator, doc, score_thresh=0.8, **kwargs):
    """Apply a validator check to a doc's raw pages."""
    outer_task_name = asyncio.current_task().get_name()
    validation_checks = [
        asyncio.create_task(
            validator.check(text, **kwargs), name=outer_task_name
        )
        for text in doc.raw_pages
    ]
    out = await asyncio.gather(*validation_checks)
    score = _weighted_vote(out, doc)
    logger.debug(
        "%s score is %.2f for doc from source %s (Pass: %s)",
        validator.__class__.__name__,
        score,
        doc.attrs.get("source", "Unknown"),
        str(score > score_thresh),
    )
    return score > score_thresh


def _weighted_vote(out, doc):
    """Compute weighted average of responses based on text length."""
    if not doc.raw_pages:
        return 0
    weights = [len(text) for text in doc.raw_pages]
    total = sum(verdict * weight for verdict, weight in zip(out, weights))
    return total / sum(weights)
