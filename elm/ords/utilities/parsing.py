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
    except json.decoder.JSONDecodeError:
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


# fmt: off
def merge_overlapping_texts(text_chunks, n=300):
    """Merge chunks fo text by removing any overlap.

    Parameters
    ----------
    text_chunks : iterable of str
        Iterable containing text chunks which may or may not contain
        consecutive overlapping portions.
    n : int, optional
        Number of characters to check at the beginning of each message
        for overlap with the previous message. By default, ``100``.

    Returns
    -------
    str
        Merged text.
    """
    if not text_chunks:
        return ""

    out_text = text_chunks[0]
    for next_text in text_chunks[1:]:
        start_ind = out_text[-2 * n:].find(next_text[:n])
        if start_ind == -1:
            out_text = "\n".join([out_text, next_text])
            continue
        start_ind = 2 * n - start_ind
        out_text = "".join([out_text, next_text[start_ind:]])
    return out_text
