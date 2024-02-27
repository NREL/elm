# -*- coding: utf-8 -*-
"""ELM Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Large Wind Energy Conversion Systems).
"""
import logging

from elm.ords.validation.content import (
    ValidationWithMemory,
    possibly_mentions_wind,
)
from elm.ords.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)


class OrdinanceExtractor(ValidationWithMemory):
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

    CONTAINS_ORD_PROMPT = (
        "You extract structured data from text. Return your answer in JSON "
        "format (not markdown). Your JSON file must include exactly three "
        "keys. The first key is 'wind_reqs', which is a string that "
        "summarizes the setbacks or other geospatial siting requirements (if "
        "any) given in the text for a wind turbine. The second key is 'reqs', "
        "which lists the quantitative values from the text excerpt that can "
        "be used to compute setbacks or other geospatial siting requirements "
        "for a wind turbine/tower (empty list if none exist in the text). The "
        "last key is '{key}', which is a boolean that is set to True if the "
        "text excerpt provides enough quantitative info to compute setbacks "
        "or other geospatial siting requirements for a wind turbine/tower "
        "and False otherwise. Geospatial siting is impacted by any of the "
        "following:\n"
        "- buildings / structures / residences\n"
        "- property lines / parcels / subdivisions\n"
        "- roads / rights-of-way\n"
        "- railroads\n"
        "- overhead electrical transmission wires\n"
        "- bodies of water including wetlands, lakes, reservoirs, streams, "
        "and rivers\n"
        "- natural, wildlife, and environmental conservation areas\n"
        "- noise restrictions\n"
        "- shadow flicker restrictions\n"
        "- density restrictions\n"
        "- turbine height restrictions\n"
        "- minimum/maximum lot size"
    )

    IS_UTILITY_SCALE_PROMPT = (
        "You are a legal scholar that reads ordinance text and determines "
        "wether it applies to large wind energy systems. Wind energy systems "
        "(WES) may also be referred to as wind turbines, wind energy "
        "conversion systems (WECS), wind energy facilities (WEF), wind energy "
        "turbines (WET), large wind energy turbines (LWET), utility-scale "
        "wind energy turbines (UWET), commercial wind energy systems, or "
        "similar. Your client is a wind developer that does not care about "
        "ordinances related to private, micro, small, or medium sized wind "
        "energy systems. Return your answer in JSON format (not markdown). "
        "Your JSON file must include exactly two keys. The first key is "
        "'summary' which contains a string that summarizes the types of wind "
        "energy systems the text applies to (if any). The second key is "
        "'{key}', which is a boolean that is set to True if any part of the "
        "text excerpt is applicable to the type of wind energy conversion "
        "systems that the client is interested in and False otherwise."
    )

    def __init__(self, structured_llm_caller, text_chunks, num_to_recall=2):
        """

        Parameters
        ----------
        structured_llm_caller : StructuredLLMCaller
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
            if ind >= 0 and ind < len(self.text_chunks)
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
                if not self.is_legal_text:
                    return False

                if not any(self._wind_mention_mem[-self.num_to_recall :]):
                    continue

            logger.debug("Processing text at ind %d", ind)
            logger.debug("Text:\n%s", text)

            if ind < min_chunks_to_process:
                is_legal_text = await self.parse_from_ind(
                    ind, self.IS_LEGAL_TEXT_PROMPT, key="legal_text"
                )
                self._legal_text_mem.append(is_legal_text)
                if not is_legal_text:
                    logger.debug("Text at ind %d is not legal text", ind)
                    continue

            contains_ord_info = await self.parse_from_ind(
                ind, self.CONTAINS_ORD_PROMPT, key="contains_ord_info"
            )
            if not contains_ord_info:
                logger.debug(
                    "Text at ind %d does not contain ordinance info", ind
                )
                continue

            is_utility_scale = await self.parse_from_ind(
                ind, self.IS_UTILITY_SCALE_PROMPT, key="x"
            )
            if not is_utility_scale:
                logger.debug(
                    "Text at ind %d is not for utility-scale WECS", ind
                )
                continue

            self._ordinance_chunks.append({"text": text, "ind": ind})
            logger.debug("Added text at ind %d to ordinances", ind)
            # mask, since we got a good result
            self._wind_mention_mem[-1] = False

        return bool(self._ordinance_chunks)
