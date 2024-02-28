from .parsing import llm_response_as_json, merge_overlapping_texts

RTS_SEPARATORS = [
    "Setbacks",
    "CHAPTER ",
    "SECTION ",
    "\r\n\r\n",
    "\r\n",
    "Chapter ",
    "Section ",
    "\n\n",
    "\n",
    "section ",
    "chapter ",
    " ",
    "",
]
