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
)
from elm.ords.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)


class OrdinanceValidator(ValidationWithMemory):
    """Check document text for wind ordinances."""

    WELL_PERMITS_PROMPT = (
        "You extract structured data from text. Return your answer in JSON "
        "format (not markdown). Your JSON file must include exactly three "
        "keys. The first key is 'district_rules' which is a string summarizes "
        "the rules associated with the groundwater conservation district. "
        "The second key is 'well_requirements', which is a string that "
        "summarizes the requirements for drilling a groundwater well. The "
        "last key is '{key}', which is a boolean that is set to True if the "
        "text excerpt provides substantive information related to the "
        "groundwater conservation district's rules or plans. "
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
        # self._wind_mention_mem = []
        self._ordinance_chunks = []

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
            # TODO: I got good results without a similar test for water but
            # is it worth including for the sake of being thorough?

            # self._wind_mention_mem.append(possibly_mentions_wind(text))
            # if ind >= min_chunks_to_process:
            #     # fmt: off
            #     if not any(self._wind_mention_mem[-self.num_to_recall:]):
            #         continue

            logger.debug("Processing text at ind %d", ind)
            logger.debug("Text:\n%s", text)

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
            # self._wind_mention_mem[-1] = False

        return bool(self._ordinance_chunks)
