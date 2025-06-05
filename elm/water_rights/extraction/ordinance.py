# -*- coding: utf-8 -*-
"""ELM water rights document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Large Wind Energy Conversion Systems).
"""
import asyncio
import logging

from elm import ApiBase
from elm.ords.validation.content import (
    ValidationWithMemory,
    possibly_mentions_wind,
)
from elm.ords.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)


RESTRICTIONS = """- buildings / structures / residences
- property lines / parcels / subdivisions
- roads / rights-of-way
- railroads
- overhead electrical transmission wires
- bodies of water including wetlands, lakes, reservoirs, streams, and rivers
- natural, wildlife, and environmental conservation areas
- noise restrictions
- shadow flicker restrictions
- density restrictions
- turbine height restrictions
- minimum/maximum lot size
"""


class OrdinanceValidator(ValidationWithMemory):
    """Check document text for wind ordinances."""

    IS_LEGAL_TEXT_PROMPT = (
        "You extract structured data from text. Return your answer in JSON "
        "format (not markdown). Your JSON file must include exactly three "
        "keys. The first key is 'summary', which is a string that provides a "
        "short summary of the text. The second key is 'type', which is a "
        "string that best represent the type of document this text belongs "
        "to. The third key is '{key}', which is a boolean that is set to "
        "True if the type of the text (as you previously determined) is a "
        "legally-binding statute or code and False if the text is an excerpt "
        "from other non-legal text such as a news article, survey, summary, "
        "application, public notice, etc."
    )

    CONTAINS_DEF_PROMPT = (
        "You extract structured data from text. Return your answer in JSON "
        "format (not markdown). Your JSON file must include exactly two "
        "keys. The first key is 'resource_def', which is a string that "
        "summarizes the definition of a geothermal resource mentioned in the "
        "text. The "
        "last key is '{key}', which is a boolean that is set to True if the "
        "text excerpt provides enough info to describe geothermal resources "
        "and False otherwise."
    )

    WELL_PERMITS_PROMPT = (
        "You extract structured data from text. Return your answer in JSON "
        "format (not markdown). Your JSON file must include exactly three "
        "keys. The first key is 'district_rules' which is a string summarizes "
        "the rules associated with the ground water conservation district. "
        "The second key is 'well_requirements', which is a string that "
        "summarizes the requirements for drilling a groundwater well. The "
        "last key is '{key}', which is a boolean that is set to True if the "
        "text excerpt provides enough info to determine what is required to "
        "drill a water well and False otherwise. "
    )

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
        super().__init__(
            structured_llm_caller=structured_llm_caller,
            text_chunks=text_chunks,
            num_to_recall=num_to_recall,
        )
        self._legal_text_mem = []
        self._wind_mention_mem = []
        self._ordinance_chunks = []

    @property
    def is_legal_text(self):
        """bool: ``True`` if text was found to be from a legal source."""
        if not self._legal_text_mem:
            return False
        return sum(self._legal_text_mem) >= 0.5 * len(self._legal_text_mem)

    @property
    def ordinance_text(self):
        """str: Combined ordinance text from the individual chunks."""
        inds_to_grab = set()
        for info in self._ordinance_chunks:
            inds_to_grab |= {
                info["ind"] + x for x in range(1 - self.num_to_recall, 2)
            }

        text = [
            self.text_chunks[ind]
            for ind in sorted(inds_to_grab)
            if 0 <= ind < len(self.text_chunks)
        ]
        return merge_overlapping_texts(text)

    async def parse(self, min_chunks_to_process=3):
        """Parse text chunks and look for ordinance text.

        Parameters
        ----------
        min_chunks_to_process : int, optional
            Minimum number of chunks to process before checking if
            document resembles legal text and ignoring chunks that don't
            pass the wind heuristic. By default, ``3``.

        Returns
        -------
        bool
            ``True`` if any ordinance text was found in the chunks.
        """
        for ind, text in enumerate(self.text_chunks):
            self._wind_mention_mem.append(possibly_mentions_wind(text))
            if ind >= min_chunks_to_process:
                # TODO: find another method to validate and/or bypass this, some info is not being recognized as legal text
                # if not self.is_legal_text:
                #     return False

                # fmt: off
                if not any(self._wind_mention_mem[-self.num_to_recall:]):
                    continue

            logger.debug("Processing text at ind %d", ind)
            logger.debug("Text:\n%s", text)

            # if ind < min_chunks_to_process:
            #     is_legal_text = await self.parse_from_ind(
            #         ind, self.IS_LEGAL_TEXT_PROMPT, key="legal_text"
            #     )
            #     self._legal_text_mem.append(is_legal_text)
            #     if not is_legal_text:
            #         logger.debug("Text at ind %d is not legal text", ind)
            #         continue

            #     logger.debug("Text at ind %d is legal text", ind)

            contains_ord_info = await self.parse_from_ind(
                # ind, self.CONTAINS_DEF_PROMPT, key="contains_ord_info"
                ind, self.WELL_PERMITS_PROMPT, key="contains_ord_info"
            )
            if not contains_ord_info:
                logger.debug(
                    "Text at ind %d does not contain ordinance info", ind
                )
                continue

            logger.debug("Text at ind %d does contain ordinance info", ind)

            self._ordinance_chunks.append({"text": text, "ind": ind})
            logger.debug("Added text at ind %d to ordinances", ind)
            # mask, since we got a good result
            self._wind_mention_mem[-1] = False

        return bool(self._ordinance_chunks)


class OrdinanceExtractor:
    """Extract succinct ordinance text from input"""

    SYSTEM_MESSAGE = (
        "You extract one or more direct excerpts from a given text based on "
        "the user's request. Maintain all original formatting and characters "
        "without any paraphrasing. If the relevant text is inside of a "
        "space-delimited table, return the entire table with the original "
        "space-delimited formatting. Never paraphrase! Only return portions "
        "of the original text directly."
    )

    MODEL_INSTRUCTIONS_DEFINITION = (
        "Extract one or more direct text excerpts related to the definition "
        "of a geothermal resource. Include section headers (if any) for the "
        "text excerpts. Also include any supplementary text that might be "
        "referenced in the definition or that defines a by-product. "
        "Additionally, any text referring to a 'Geothermal Area' should be "
        "included. If there is no text related to geothermal definitions "
        'simply say: "No relevant text."'
    )

    MODEL_INSTRUCTIONS_PERMITS = (
        "Extract one or more direct text excerpts related to the requirements "
        "needed to drill a water well. The text should include any information "
        "related to the permit application process and requirements specified "
        "in order to obtain permission to drill a water well. Include section "
        "headers (if any) for the "
        "text excerpts. If there is no text related to well permits "
        'simply say: "No relevant text."'
    )

    def __init__(self, llm_caller):
        """

        Parameters
        ----------
        llm_caller : elm.ords.llm.LLMCaller
            LLM Caller instance used to extract ordinance info with.
        """
        self.llm_caller = llm_caller

    async def _process(self, text_chunks, instructions, valid_chunk):
        """Perform extraction processing."""
        logger.info(
            "Extracting ordinance text from %d text chunks asynchronously...",
            len(text_chunks),
        )
        outer_task_name = asyncio.current_task().get_name()
        summaries = [
            asyncio.create_task(
                self.llm_caller.call(
                    sys_msg=self.SYSTEM_MESSAGE,
                    content=f"Text:\n{chunk}\n{instructions}",
                    usage_sub_label="document_ordinance_summary",
                ),
                name=outer_task_name,
            )
            for chunk in text_chunks
        ]
        summary_chunks = await asyncio.gather(*summaries)
        summary_chunks = [
            chunk for chunk in summary_chunks if valid_chunk(chunk)
        ]

        text_summary = "\n".join(summary_chunks)
        logger.debug(
            "Final summary contains %d tokens",
            ApiBase.count_tokens(
                text_summary,
                model=self.llm_caller.kwargs.get("model", "gpt-4"),
            ),
        )
        return text_summary

    async def check_for_definition(self, text_chunks):
        """Extract definition ordinance text from input text chunks.

        Parameters
        ----------
        text_chunks : list of str
            List of strings, each of which represent a chunk of text.
            The order of the strings should be the order of the text
            chunks.

        Returns
        -------
        str
            Ordinance text extracted from text chunks.
        """
        return await self._process(
            text_chunks=text_chunks,
            # instructions=self.MODEL_INSTRUCTIONS_DEFINITION,
            instructions=self.MODEL_INSTRUCTIONS_PERMITS,
            valid_chunk=_valid_chunk_not_short,
        )


def _valid_chunk(chunk):
    """True if chunk has content."""
    return chunk and "no relevant text" not in chunk.lower()


def _valid_chunk_not_short(chunk):
    """True if chunk has content and is not too short."""
    return _valid_chunk(chunk) and len(chunk) > 20