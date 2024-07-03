# -*- coding: utf-8 -*-
"""ELM Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Large Wind Energy Conversion Systems).
"""
import logging


logger = logging.getLogger(__name__)
NOT_WIND_WORDS = [
    "windy",
    "winds",
    "window",
    "windiest",
    "windbreak",
    "windshield",
    "wind blow",
    "wind erosion",
    "rewind",
    "mini wecs",
    "swecs",
    "private wecs",
    "pwecs",
    "wind direction",
    "wind movement",
    "wind attribute",
    "wind runway",
    "wind load",
    "wind orient",
    "wind damage",
]
GOOD_WIND_KEYWORDS = ["wind", "setback"]
GOOD_WIND_ACRONYMS = ["wecs", "wes", "lwet", "uwet", "wef"]
_GOOD_ACRONYM_CONTEXTS = [
    " {acronym} ",
    " {acronym}\n",
    " {acronym}.",
    "\n{acronym} ",
    "\n{acronym}.",
    "\n{acronym}\n",
    "({acronym} ",
    " {acronym})",
]
GOOD_WIND_PHRASES = ["wind energy conversion", "wind turbine", "wind tower"]


class ValidationWithMemory:
    """Validate a set of text chunks by sometimes looking at previous chunks"""

    def __init__(self, structured_llm_caller, text_chunks, num_to_recall=2):
        """

        Parameters
        ----------
        structured_llm_caller : elm.ords.llm.StructuredLLMCaller
            StructuredLLMCaller instance. Used for structured validation
            queries.
        text_chunks : list of str
            List of strings, each of which represent a chunk of text.
            The order of the strings should be the order of the text
            chunks. This validator may refer to previous text chunks to
            answer validation questions.
        num_to_recall : int, optional
            Number of chunks to check for each validation call. This
            includes the original chunk! For example, if
            `num_to_recall=2`, the validator will first check the chunk
            at the requested index, and then the previous chunk as well.
            By default, ``2``.
        """
        self.slc = structured_llm_caller
        self.text_chunks = text_chunks
        self.num_to_recall = num_to_recall
        self.memory = [{} for _ in text_chunks]

    # fmt: off
    def _inverted_mem(self, starting_ind):
        """Inverted memory."""
        inverted_mem = self.memory[:starting_ind + 1:][::-1]
        yield from inverted_mem[:self.num_to_recall]

    # fmt: off
    def _inverted_text(self, starting_ind):
        """Inverted text chunks"""
        inverted_text = self.text_chunks[:starting_ind + 1:][::-1]
        yield from inverted_text[:self.num_to_recall]

    async def parse_from_ind(self, ind, prompt, key):
        """Validate a chunk of text.

        Validation occurs by querying the LLM using the input prompt and
        parsing the `key` from the response JSON. The prompt should
        request that the key be a boolean output. If the key retrieved
        from the LLM response is False, a number of previous text chunks
        are checked as well, using the same prompt. This can be helpful
        in cases where the answer to the validation prompt (e.g. does
        this text pertain to a large WECS?) is only found in a previous
        text chunk.

        Parameters
        ----------
        ind : int
            Positive integer corresponding to the chunk index.
            Must be less than `len(text_chunks)`.
        prompt : str
            Input LLM system prompt that describes the validation
            question. This should request a JSON output from the LLM.
            It should also take `key` as a formatting input.
        key : str
            A key expected in the JSON output of the LLM containing the
            response for the validation question. This string will also
            be used to format the system prompt before it is passed to
            the LLM.

        Returns
        -------
        bool
            ``True`` if the LLM returned ``True`` for this text chunk or
            `num_to_recall-1` text chunks before it.
            ``False`` otherwise.
        """
        logger.debug("Checking %r for ind %d", key, ind)
        mem_text = zip(self._inverted_mem(ind), self._inverted_text(ind))
        for step, (mem, text) in enumerate(mem_text):
            logger.debug("Mem at ind %d is %s", step, mem)
            check = mem.get(key)
            if check is None:
                content = await self.slc.call(
                    sys_msg=prompt.format(key=key),
                    content=text,
                    usage_sub_label="document_content_validation",
                )
                check = mem[key] = content.get(key, False)
            if check:
                return check
        return False


def possibly_mentions_wind(text, match_count_threshold=1):
    """Perform a heuristic check for mention of wind energy in text.

    This check first strips the text of any wind "look-alike" words
    (e.g. "window", "windshield", etc). Then, it checks for particular
    keywords, acronyms, and phrases that pertain to wind in the text.
    If enough keywords are mentions (as dictated by
    `match_count_threshold`), this check returns ``True``.

    Parameters
    ----------
    text : str
        Input text that may or may not mention win in relation to wind
        energy.
    match_count_threshold : int, optional
        Number of keywords that must match for the text to pass this
        heuristic check. Count must be strictly greater than this value.
        By default, ``1``.

    Returns
    -------
    bool
        ``True`` if the number of keywords/acronyms/phrases detected
        exceeds the `match_count_threshold`.
    """
    heuristics_text = _convert_to_heuristics_text(text)
    total_keyword_matches = _count_single_keyword_matches(heuristics_text)
    total_keyword_matches += _count_acronym_matches(heuristics_text)
    total_keyword_matches += _count_phrase_matches(heuristics_text)
    return total_keyword_matches > match_count_threshold


def _convert_to_heuristics_text(text):
    """Convert text for heuristic wind content parsing"""
    heuristics_text = text.casefold()
    for word in NOT_WIND_WORDS:
        heuristics_text = heuristics_text.replace(word, "")
    return heuristics_text


def _count_single_keyword_matches(heuristics_text):
    """Count number of good wind energy keywords that appear in text."""
    return sum(keyword in heuristics_text for keyword in GOOD_WIND_KEYWORDS)


def _count_acronym_matches(heuristics_text):
    """Count number of good wind energy acronyms that appear in text."""
    acronym_matches = 0
    for context in _GOOD_ACRONYM_CONTEXTS:
        acronym_keywords = {
            context.format(acronym=acronym) for acronym in GOOD_WIND_ACRONYMS
        }
        acronym_matches = sum(
            keyword in heuristics_text for keyword in acronym_keywords
        )
        if acronym_matches > 0:
            break
    return acronym_matches


def _count_phrase_matches(heuristics_text):
    """Count number of good wind energy phrases that appear in text."""
    return sum(
        all(keyword in heuristics_text for keyword in phrase.split(" "))
        for phrase in GOOD_WIND_PHRASES
    )
