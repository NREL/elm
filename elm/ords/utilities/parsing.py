# -*- coding: utf-8 -*-
"""ELM Ordinances parsing utilities."""
import json
import logging


logger = logging.getLogger(__name__)
_JSON_INSTRUCTIONS = "Return your answer in JSON format"


def llm_response_as_json(content):
    """LLM response to JSON.

    Parameters
    ----------
    content : str
        LLM response that contains a string representation of
        a JSON file.

    Returns
    -------
    dict
        Response parsed into dictionary. This dictionary will be empty
        if the response cannot be parsed by JSON.
    """
    content = content.lstrip().rstrip()
    content = content.lstrip("```").lstrip("json").lstrip("\n")
    content = content.rstrip("```")
    content = content.replace("True", "true").replace("False", "false")
    try:
        content = json.loads(content)
    except json.decoder.JSONDecodeError as e:
        logger.error(
            "LLM returned improperly formatted JSON. "
            "This is likely due to the completion running out of tokens. "
            "Setting a higher token limit may fix this error. "
            "Also ensure you are requesting JSON output in your prompt. "
            "JSON returned:\n%s",
            content,
        )
        content = {}
    return content


class StructuredLLMCaller:
    """Class to support structured (JSON) LLM calling functionality."""

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

    async def call(self, sys_msg, content):
        """Call LLM service for validation."""
        sys_msg = _add_json_instructions_if_needed(sys_msg)

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


def _add_json_instructions_if_needed(system_message):
    """Add JSON instruction to system message if needed."""
    if _JSON_INSTRUCTIONS.casefold() not in system_message.casefold():
        logger.debug(
            "JSON instructions not found in system message. Adding..."
        )
        system_message = f"{system_message} {_JSON_INSTRUCTIONS}."
        logger.debug("New system message:\n%s", system_message)
    return system_message
