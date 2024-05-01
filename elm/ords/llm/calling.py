# -*- coding: utf-8 -*-
"""ELM Ordinances LLM Calling classes."""
import logging

from elm.ords.utilities import llm_response_as_json


logger = logging.getLogger(__name__)
_JSON_INSTRUCTIONS = "Return your answer in JSON format"


class BaseLLMCaller:
    """Class to support LLM calling functionality."""

    def __init__(self, llm_service, usage_tracker=None, **kwargs):
        """

        Parameters
        ----------
        llm_service : elm.ords.services.base.Service
            LLM service used for queries.
        usage_tracker : elm.ords.services.usage.UsageTracker, optional
            Optional tracker instance to monitor token usage during
            LLM calls. By default, ``None``.
        **kwargs
            Keyword arguments to be passed to the underlying service
            processing function (i.e. `llm_service.call(**kwargs)`).
            Should *not* contain the following keys:

                - usage_tracker
                - usage_sub_label
                - messages

            These arguments are provided by this caller object.
        """
        self.llm_service = llm_service
        self.usage_tracker = usage_tracker
        self.kwargs = kwargs


class LLMCaller(BaseLLMCaller):
    """Simple LLM caller, with no memory and no parsing utilities."""

    async def call(self, sys_msg, content, usage_sub_label="default"):
        """Call LLM.

        Parameters
        ----------
        sys_msg : str
            The LLM system message.
        content : str
            Your chat message for the LLM.
        usage_sub_label : str, optional
            Label to store token usage under. By default, ``"default"``.

        Returns
        -------
        str | None
            The LLM response, as a string, or ``None`` if something went
            wrong during the call.
        """
        response = await self.llm_service.call(
            usage_tracker=self.usage_tracker,
            usage_sub_label=usage_sub_label,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": content},
            ],
            **self.kwargs,
        )
        return response


class ChatLLMCaller(BaseLLMCaller):
    """Class to support chat-like LLM calling functionality."""

    def __init__(
        self, llm_service, system_message, usage_tracker=None, **kwargs
    ):
        """

        Parameters
        ----------
        llm_service : elm.ords.services.base.Service
            LLM service used for queries.
        system_message : str
            System message to use for chat with LLM.
        usage_tracker : elm.ords.services.usage.UsageTracker, optional
            Optional tracker instance to monitor token usage during
            LLM calls. By default, ``None``.
        **kwargs
            Keyword arguments to be passed to the underlying service
            processing function (i.e. `llm_service.call(**kwargs)`).
            Should *not* contain the following keys:

                - usage_tracker
                - usage_sub_label
                - messages

            These arguments are provided by this caller object.
        """
        super().__init__(llm_service, usage_tracker, **kwargs)
        self.messages = [{"role": "system", "content": system_message}]

    async def call(self, content, usage_sub_label="chat"):
        """Chat with the LLM.

        Parameters
        ----------
        content : str
            Your chat message for the LLM.
        usage_sub_label : str, optional
            Label to store token usage under. By default, ``"chat"``.

        Returns
        -------
        str | None
            The LLM response, as a string, or ``None`` if something went
            wrong during the call.
        """
        self.messages.append({"role": "user", "content": content})

        response = await self.llm_service.call(
            usage_tracker=self.usage_tracker,
            usage_sub_label=usage_sub_label,
            messages=self.messages,
            **self.kwargs,
        )
        if response is None:
            self.messages = self.messages[:-1]
            return None

        self.messages.append({"role": "assistant", "content": response})
        return response


class StructuredLLMCaller(BaseLLMCaller):
    """Class to support structured (JSON) LLM calling functionality."""

    async def call(self, sys_msg, content, usage_sub_label="default"):
        """Call LLM for structured data retrieval.

        Parameters
        ----------
        sys_msg : str
            The LLM system message. If this text does not contain the
            instruction text "Return your answer in JSON format", it
            will be added.
        content : str
            LLM call content (typically some text to extract info from).
        usage_sub_label : str, optional
            Label to store token usage under. By default, ``"default"``.

        Returns
        -------
        dict
            Dictionary containing the LLM-extracted features. Dictionary
            may be empty if there was an error during the LLM call.
        """
        sys_msg = _add_json_instructions_if_needed(sys_msg)

        response = await self.llm_service.call(
            usage_tracker=self.usage_tracker,
            usage_sub_label=usage_sub_label,
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
