"""ELM Ordinance utilities. """

from elm.ords.utilities.counties import load_all_county_info, load_counties_from_fp
from elm.ords.utilities.parsing import llm_response_as_json, merge_overlapping_texts


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
