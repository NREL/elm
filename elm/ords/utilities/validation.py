# -*- coding: utf-8 -*-
"""ELM Ordinance Validation utilities

These are primarily used to validate that a legal document applies to a
particular location.
"""
import asyncio
import logging
from abc import ABC, abstractmethod

from elm.ords.utilities.parsing import llm_response_as_json


logger = logging.getLogger(__name__)

_JSON_INSTRUCTIONS = "Return your answer in JSON format"


class Validator(ABC):
    """Document validation base class."""

    SYSTEM_MESSAGE = None
    """LLM system message describing validation task. """

    def __init__(self, llm_service, usage_tracker=None, **kwargs):
        """

        Parameters
        ----------
        llm_service : elm.ords.services.base.Service
            LLM service used for validation queries.
        usage_tracker : elm.ords.services.usage.UsageTracker, optional
            Optional tracker instance to monitor token usage during
            validation. By default, ``None``.
        **kwargs
            Keyword arguments to be passed to the underlying service
            processing function (i.e. `llm_service.call(**kwargs)`).
            Should *not* contain the following keys:

                - usage_tracker
                - usage_sub_label
                - messages

            These arguments are provided by the validation object.
        """
        self.llm_service = llm_service
        self.usage_tracker = usage_tracker
        self.kwargs = kwargs
        self._add_json_instructions_if_needed()

    def _add_json_instructions_if_needed(self):
        """Add JSON instruction to system message if needed."""
        if _JSON_INSTRUCTIONS.casefold() not in self.SYSTEM_MESSAGE.casefold():
            logger.debug(
                "JSON instructions not found in system message. Adding..."
            )
            self.SYSTEM_MESSAGE = (
                f"{self.SYSTEM_MESSAGE} {_JSON_INSTRUCTIONS}."
            )
            logger.debug("New system message:\n%s", self.SYSTEM_MESSAGE)

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
        out = await self._call_llm(sys_msg, content)
        return self._parse_output(out)

    async def _call_llm(self, sys_msg, content):
        """Call LLM service for validation."""
        logger.debug("Submitting API call for validation")
        response = await self.llm_service.call(
            usage_tracker=self.usage_tracker,
            usage_sub_label="location_validation",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": content},
            ],
            **self.kwargs,
        )
        return llm_response_as_json(response) if response else {}

    @abstractmethod
    def _parse_output(self, props):
        """Parse LLM response and return `True` if the document passes."""
        raise NotImplementedError


class URLValidator(Validator):

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


class CountyJurisdictionValidator(Validator):

    SYSTEM_MESSAGE = (
        "You extract structured data from legal text. Return "
        "your answer in JSON format. Your JSON file must include exactly "
        "three keys. The first key is 'x', which is a boolean that is set to "
        "`True` if the text excerpt explicitly mentions that the regulations "
        "within apply to a jurisdiction other than {county} County (i.e. they "
        "apply to a subdivision like a township or a city, or they apply more "
        "broadly, like to a state or the full country). `False` if the "
        "regulations in the text apply to {county} County or if there is not "
        "enough information to determine the answer. The second key is 'y', "
        "which is a boolean that is set to `True` if the text excerpt "
        "explicitly mentions that the regulations within apply to more than "
        "one county. `False` if the regulations in the text excerpt apply to "
        "a single county only or if there is not enough information to "
        "determine the answer. The third key is 'explanation', which is a "
        "string that contains a short explanation if you chose `True` for any "
        "answers above."
    )

    def _parse_output(self, props):
        """Parse LLM response and return `True` if the document passes."""
        logger.debug(
            "Parsing county jurisdiction validation output:\n\t%s", props
        )
        check_vars = ("x", "y")
        return not any(props.get(var) for var in check_vars)


class CountyNameValidator(Validator):

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

    def __init__(self, score_thresh=0.8, **kwargs):
        """

        Parameters
        ----------
        score_thresh : float, optional
            Score threshold to exceed when voting on content from raw
            pages. By default, ``0.8``.
        **kwargs
            Keyword arguments to pass to the internal `Validator`
            instances.

        """
        self.score_thresh = score_thresh
        self.cn_validator = CountyNameValidator(**kwargs)
        self.cj_validator = CountyJurisdictionValidator(**kwargs)
        self.url_validator = URLValidator(**kwargs)

    async def check(self, doc, county, state):
        """Check if the document belongs to the county.

        Parameters
        ----------
        doc : elm.web.document.BaseDocument
            Document instance. Should contain a "source" key in the
            metadata that contains a URL (used for the URL validation
            check). Raw content will be parsed for county name and
            correct jurisdiction.
        county : str
            County that document should belong to.
        state : _type_
            State corresponding to `county` input.

        Returns
        -------
        bool
            `True` if the doc contents pertain to the input county.
            `False` otherwise.
        """
        logger.debug("Checking for correct for jurisdiction...")
        jurisdiction_is_county = await _validator_check_for_doc(
            validator=self.cj_validator,
            doc=doc,
            score_thresh=self.score_thresh,
            county=county,
        )
        if not jurisdiction_is_county:
            return False

        logger.debug("Checking URL for county name...")
        url_is_county = await self.url_validator.check(
            doc.metadata.get("source"), county=county, state=state
        )
        if url_is_county:
            return True

        logger.debug("Checking text for county name (heuristic)...")
        correct_county_heuristic = _heuristic_check_for_county_and_state(
            doc, county, state
        )
        if correct_county_heuristic:
            return True

        logger.debug("Checking text for county name (LLM)...")
        return await _validator_check_for_doc(
            validator=self.cn_validator,
            doc=doc,
            score_thresh=self.score_thresh,
            county=county,
            state=state,
        )


def _heuristic_check_for_county_and_state(doc, county, state):
    """Check if county and state names are in doc"""
    if not any(county.lower() in t.lower() for t in doc.pages):
        logger.debug(f"    - False, did not find {county!r} in text")
        return False

    if not any(state.lower() in t.lower() for t in doc.pages):
        logger.debug(f"    - False, did not find {state!r} in text")
        return False

    logger.debug(f"    - True, found {county}, {state} in text")
    return True


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
    return _weighted_vote(out, doc) > score_thresh


def _weighted_vote(out, doc):
    """Compute weighted average of responses based on text length."""
    if not doc.raw_pages:
        return 0
    weights = [len(text) for text in doc.raw_pages]
    total = sum(verdict * weight for verdict, weight in zip(out, weights))
    return total / sum(weights)
