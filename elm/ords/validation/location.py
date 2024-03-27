# -*- coding: utf-8 -*-
"""ELM Ordinance Location Validation logic

These are primarily used to validate that a legal document applies to a
particular location.
"""
import asyncio
import logging
from abc import ABC, abstractmethod

from elm.ords.extraction.ngrams import convert_text_to_sentence_ngrams


logger = logging.getLogger(__name__)


class FixedMessageValidator(ABC):
    """Validation base class using a static system prompt."""

    SYSTEM_MESSAGE = None
    """LLM system message describing validation task. """

    def __init__(self, structured_llm_caller):
        """

        Parameters
        ----------
        structured_llm_caller : :class:`elm.ords.llm.StructuredLLMCaller`
            StructuredLLMCaller instance. Used for structured validation
            queries.
        """
        self.slc = structured_llm_caller

    async def check(self, content, **fmt_kwargs):
        """Check if the content passes the validation.

        The exact validation is outlined in the class `SYSTEM_MESSAGE`.

        Parameters
        ----------
        content : str
            Document content to validate.
        **fmt_kwargs
            Keyword arguments to be passed to `SYSTEM_MESSAGE.format()`.

        Returns
        -------
        bool
            ``True`` if the content passes the validation check,
            ``False`` otherwise.
        """
        if not content:
            return False
        sys_msg = self.SYSTEM_MESSAGE.format(**fmt_kwargs)
        out = await self.slc.call(
            sys_msg, content, usage_sub_label="document_location_validation"
        )
        return self._parse_output(out)

    @abstractmethod
    def _parse_output(self, props):
        """Parse LLM response and return `True` if the document passes."""
        raise NotImplementedError


class URLValidator(FixedMessageValidator):
    """Validator that checks wether a URL matches a county."""

    SYSTEM_MESSAGE = (
        "You extract structured data from a URL. Return your "
        "answer in JSON format. Your JSON file must include exactly two keys. "
        "The first key is 'correct_county', which is a boolean that is set to "
        "`True` if the URL mentions {county} County in some way. DO NOT infer "
        "based on information in the URL about any US state, city, township, "
        "or otherwise. `False` if not sure. The second key is "
        "'correct_state', which is a boolean that is set to `True` if the URL "
        "mentions {state} State in some way. DO NOT infer based on "
        "information in the URL about any US county, city, township, or "
        "otherwise. `False` if not sure."
    )

    def _parse_output(self, props):
        """Parse LLM response and return `True` if the document passes."""
        logger.debug("Parsing URL validation output:\n\t%s", props)
        check_vars = ("correct_county", "correct_state")
        return all(props.get(var) for var in check_vars)


class CountyJurisdictionValidator(FixedMessageValidator):
    """Validator that checks wether text applies at the county level."""

    SYSTEM_MESSAGE = (
        "You extract structured data from legal text. Return "
        "your answer in JSON format. Your JSON file must include exactly "
        "three keys. The first key is 'x', which is a boolean that is set to "
        "`True` if the text excerpt explicitly mentions that the regulations "
        "within apply to a jurisdiction scope other than {county} County "
        "(i.e. they apply to a subdivision like a township or a city, or "
        "they apply more broadly, like to a state or the full country). "
        "`False` if the regulations in the text apply at the {county} County "
        "level, if the regulations in the text apply to all unincorporated "
        "areas of {county} County, or if there is not enough information to "
        "determine the answer. The second key is 'y', which is a boolean "
        "that is set to `True` if the text excerpt explicitly mentions that "
        "the regulations within apply to more than one county. `False` if "
        "the regulations in the text excerpt apply to a single county only "
        "or if there is not enough information to determine the answer. The "
        "third key is 'explanation', which is a string that contains a short "
        "explanation if you chose `True` for any answers above."
    )

    def _parse_output(self, props):
        """Parse LLM response and return `True` if the document passes."""
        logger.debug(
            "Parsing county jurisdiction validation output:\n\t%s", props
        )
        check_vars = ("x", "y")
        return not any(props.get(var) for var in check_vars)


class CountyNameValidator(FixedMessageValidator):
    """Validator that checks wether text applies to a particular county."""

    SYSTEM_MESSAGE = (
        "You extract structured data from legal text. Return "
        "your answer in JSON format. Your JSON file must include exactly "
        "three keys. The first key is 'wrong_county', which is a boolean that "
        "is set to `True` if the legal text is not for {county} County. Do "
        "not infer based on any information about any US state, city, "
        "township, or otherwise. `False` if the text applies to {county} "
        "County or if there is not enough information to determine the "
        "answer. The second key is 'wrong_state', which is a boolean that is "
        "set to `True` if the legal text is not for a county in {state} "
        "State. Do not infer based on any information about any US county, "
        "city, township, or otherwise. `False` if the text applies to "
        "a county in {state} State or if there is not enough information to "
        "determine the answer. The third key is 'explanation', which is a "
        "string that contains a short explanation if you chose `True` for "
        "any answers above."
    )

    def _parse_output(self, props):
        """Parse LLM response and return `True` if the document passes."""
        logger.debug("Parsing county validation output:\n\t%s", props)
        check_vars = ("wrong_county", "wrong_state")
        return not any(props.get(var) for var in check_vars)


class CountyValidator:
    """ELM Ords County validator.

    Combines the logic of several validators into a single class.
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
        self.url_validator = URLValidator(structured_llm_caller)

    async def check(self, doc, county, state):
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
        source = doc.metadata.get("source")
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
            "Checking URL (%s) for county name...", source or "Unknown"
        )
        url_is_county = await self.url_validator.check(
            source, county=county, state=state
        )
        if url_is_county:
            return True

        logger.debug(
            "Checking text for county name (heuristic; URL: %s)...",
            source or "Unknown",
        )
        correct_county_heuristic = _heuristic_check_for_county_and_state(
            doc, county, state
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


def _heuristic_check_for_county_and_state(doc, county, state):
    """Check if county and state names are in doc"""
    return any(
        any(
            (county.lower() in fg and state.lower() in fg)
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
        doc.metadata.get("source", "Unknown"),
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
