# -*- coding: utf-8 -*-
"""ELM Ordinances OpenAI service amd utils."""
import openai

from elm.base import ApiBase


def usage_from_response(current_usage, response):
    """OpenAI usage parser.

    Parameters
    ----------
    current_usage : dict
        Dictionary containing current usage information. For OpenAI
        trackers, this may contain the keys ``"requests"``,
        ``"prompt_tokens"``, and ``"response_tokens"`` if there is
        already existing tracking information. Empty dictionaries are
        allowed, in which case the three keys above will be added to
        this input.
    response : openai.Completion
        OpenAI Completion object. Must contain a ``usage`` attribute
        that

    Returns
    -------
    dict
        Dictionary with updated usage statistics.
    """
    current_usage["requests"] = current_usage.get("requests", 0) + 1
    current_usage["prompt_tokens"] = (
        current_usage.get("prompt_tokens", 0) + response.usage.prompt_tokens
    )
    current_usage["response_tokens"] = (
        current_usage.get("response_tokens", 0)
        + response.usage.completion_tokens
    )
    return current_usage


def count_tokens(messages, model):
    """Count the number of tokens in an outgoing set of messages.

    Parameters
    ----------
    messages : list
        A list of message objects, where the latter is represented
        using a dictionary. Each message dictionary must have a
        "content" key containing the string to count tokens for.
    model : str
        The OpenAI model being used. This input will be passed to
        :func:`tiktoken.encoding_for_model`.

    Returns
    -------
    int
        Total number of tokens in the set of messages outgoing to
        OpenAI.

    References
    ----------
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    message_total = sum(
        ApiBase.count_tokens(message["content"], model=model) + 4
        for message in messages
    )
    return message_total + 3
