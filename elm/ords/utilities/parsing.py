# -*- coding: utf-8 -*-
"""ELM Ordinances parsing utilities."""
import json
import logging


logger = logging.getLogger(__name__)


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
