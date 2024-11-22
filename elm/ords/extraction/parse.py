# -*- coding: utf-8 -*-
"""ELM Ordinance structured parsing class."""
import asyncio
import logging
from copy import deepcopy
from itertools import chain

import pandas as pd

from elm.ords.llm.calling import BaseLLMCaller, ChatLLMCaller
from elm.ords.utilities import llm_response_as_json
from elm.ords.extraction.tree import AsyncDecisionTree
from elm.ords.extraction.features import SetbackFeatures
from elm.ords.extraction.graphs import (
    EXTRACT_ORIGINAL_TEXT_PROMPT,
    setup_graph_wes_types,
    setup_base_graph,
    setup_multiplier,
    setup_conditional,
    setup_participating_owner,
    setup_graph_extra_restriction,
    llm_response_starts_with_yes,
)


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a legal scholar explaining legal ordinances to a wind "
    "energy developer."
)
SETBACKS_SYSTEM_MESSAGE = (
    f"{DEFAULT_SYSTEM_MESSAGE} "
    "For the duration of this conversation, only focus on "
    "ordinances relating to setbacks from {feature} for {wes_type}. Ignore "
    "all text that pertains to private, micro, small, or medium sized wind "
    "energy systems."
)
RESTRICTIONS_SYSTEM_MESSAGE = (
    f"{DEFAULT_SYSTEM_MESSAGE} "
    "For the duration of this conversation, only focus on "
    "ordinances relating to {restriction} for {wes_type}. Ignore "
    "all text that pertains to private, micro, small, or medium sized wind "
    "energy systems."
)
EXTRA_RESTRICTIONS_TO_CHECK = {
    "noise": "maximum noise level",
    "max height": "maximum turbine height",
    "min lot size": "minimum lot size",
    "shadow flicker": "maximum shadow flicker",
    "density": "maximum turbine spacing",
}


def _setup_async_decision_tree(graph_setup_func, **kwargs):
    """Setup Async Decision tree dor ordinance extraction."""
    G = graph_setup_func(**kwargs)
    tree = AsyncDecisionTree(G)
    assert len(tree.chat_llm_caller.messages) == 1
    return tree


def _found_ord(messages):
    """Check if ordinance was found based on messages from the base graph.
    IMPORTANT: This function may break if the base graph structure changes.
    Always update the hardcoded values to match the base graph message
    containing the LLM response about ordinance content.
    """
    if len(messages) < 3:
        return False
    return llm_response_starts_with_yes(messages[2].get("content", ""))


async def _run_async_tree(tree, response_as_json=True):
    """Run Async Decision Tree and return output as dict."""
    try:
        response = await tree.async_run()
    except RuntimeError:
        logger.error(
            "    - NOTE: This is not necessarily an error and may just mean "
            "that the text does not have the requested data."
        )
        response = None

    if response_as_json:
        return llm_response_as_json(response) if response else {}

    return response


async def _run_async_tree_with_bm(tree, base_messages):
    """Run Async Decision Tree from base messages and return dict output."""
    tree.chat_llm_caller.messages = base_messages
    assert len(tree.chat_llm_caller.messages) == len(base_messages)
    return await _run_async_tree(tree)


def _empty_output(feature):
    """Empty output for a feature (not found in text)."""
    if feature in {"struct", "pline"}:
        return [
            {"feature": f"{feature} (participating)"},
            {"feature": f"{feature} (non-participating)"},
        ]
    return [{"feature": feature}]


class StructuredOrdinanceParser(BaseLLMCaller):
    """LLM ordinance document structured data scraping utility

    Purpose:
        Extract structured ordinance data from text.
    Responsibilities:
        1. Extract ordinance values into structured format by executing
           a decision-tree-based chain-of-thought prompt on the text for
           each value to be extracted.
    Key Relationships:
        Uses a :class:`~elm.ords.llm.calling.StructuredLLMCaller` for
        LLM queries and multiple
        :class:`~elm.ords.extraction.tree.AsyncDecisionTree` instances
        to guide the extraction of individual values.

    .. end desc
    """

    def _init_chat_llm_caller(self, system_message):
        """Initialize a ChatLLMCaller instance for the DecisionTree"""
        return ChatLLMCaller(
            self.llm_service,
            system_message=system_message,
            usage_tracker=self.usage_tracker,
            **self.kwargs,
        )

    async def parse(self, text):
        """Parse text and extract structure ordinance data.

        Parameters
        ----------
        text : str
            Ordinance text which may or may not contain setbacks for one
            or more features (property lines, structure, roads, etc.).
            Text can also contain other supported regulations (noise,
            shadow-flicker, etc,) which will be extracted as well.

        Returns
        -------
        pd.DataFrame
            DataFrame containing parsed-out ordinance values.
        """
        largest_wes_type = await self._check_wind_turbine_type(text)
        logger.info("Largest WES type found in text: %s", largest_wes_type)

        outer_task_name = asyncio.current_task().get_name()
        feature_parsers = [
            asyncio.create_task(
                self._parse_setback_feature(
                    text, feature_kwargs, largest_wes_type
                ),
                name=outer_task_name,
            )
            for feature_kwargs in SetbackFeatures()
        ]
        extras_parsers = [
            asyncio.create_task(
                self._parse_extra_restriction(
                    text, feature, r_text, largest_wes_type
                ),
                name=outer_task_name,
            )
            for feature, r_text in EXTRA_RESTRICTIONS_TO_CHECK.items()
        ]
        outputs = await asyncio.gather(*(feature_parsers + extras_parsers))

        return pd.DataFrame(chain.from_iterable(outputs))

    async def _check_wind_turbine_type(self, text):
        """Get the largest turbine size mentioned in the text."""
        logger.debug("Checking turbine_types")
        tree = _setup_async_decision_tree(
            setup_graph_wes_types,
            text=text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )
        dtree_wes_types_out = await _run_async_tree(tree)

        largest_wes_type = (
            dtree_wes_types_out.get("largest_wes_type")
            or "large wind energy systems"
        )
        return largest_wes_type

    async def _parse_extra_restriction(
        self, text, feature, restriction_text, largest_wes_type
    ):
        """Parse a non-setback restriction from the text."""
        logger.debug("Parsing extra feature %r", feature)
        system_message = RESTRICTIONS_SYSTEM_MESSAGE.format(
            restriction=restriction_text, wes_type=largest_wes_type
        )
        tree = _setup_async_decision_tree(
            setup_graph_extra_restriction,
            wes_type=largest_wes_type,
            restriction=restriction_text,
            text=text,
            chat_llm_caller=self._init_chat_llm_caller(system_message),
        )
        info = await _run_async_tree(tree)
        info.update({"feature": feature})
        return [info]

    async def _parse_setback_feature(
        self, text, feature_kwargs, largest_wes_type
    ):
        """Parse values for a setback feature."""
        feature = feature_kwargs["feature_id"]
        feature_kwargs["wes_type"] = largest_wes_type
        logger.debug("Parsing feature %r", feature)

        base_messages = await self._base_messages(text, **feature_kwargs)
        if not _found_ord(base_messages):
            logger.debug("Failed `_found_ord` check for feature %r", feature)
            return _empty_output(feature)

        if feature not in {"struct", "pline"}:
            output = {"feature": feature}
            output.update(
                await self._extract_setback_values(
                    text,
                    base_messages=base_messages,
                    **feature_kwargs,
                )
            )
            return [output]

        return await self._extract_setback_values_for_p_or_np(
            text, base_messages, **feature_kwargs
        )

    async def _base_messages(self, text, **feature_kwargs):
        """Get base messages for setback feature parsing."""
        system_message = SETBACKS_SYSTEM_MESSAGE.format(
            feature=feature_kwargs["feature"],
            wes_type=feature_kwargs["wes_type"],
        )
        tree = _setup_async_decision_tree(
            setup_base_graph,
            text=text,
            chat_llm_caller=self._init_chat_llm_caller(system_message),
            **feature_kwargs,
        )
        await _run_async_tree(tree, response_as_json=False)
        return deepcopy(tree.chat_llm_caller.messages)

    async def _extract_setback_values_for_p_or_np(
        self, text, base_messages, **feature_kwargs
    ):
        """Extract setback values for participating/non-participating ords."""
        logger.debug("Checking participating vs non-participating")
        dtree_participating_out = await self._run_setback_graph(
            setup_participating_owner,
            text,
            base_messages=deepcopy(base_messages),
            **feature_kwargs,
        )
        outer_task_name = asyncio.current_task().get_name()
        p_or_np_parsers = [
            asyncio.create_task(
                self._parse_p_or_np_text(
                    key, sub_text, base_messages, **feature_kwargs
                ),
                name=outer_task_name,
            )
            for key, sub_text in dtree_participating_out.items()
        ]
        return await asyncio.gather(*p_or_np_parsers)

    async def _parse_p_or_np_text(
        self, key, sub_text, base_messages, **feature_kwargs
    ):
        """Parse participating/non-participating sub-text for ord values."""
        feature = feature_kwargs["feature_id"]
        out_feat_name = f"{feature} ({key})"
        output = {"feature": out_feat_name}
        if not sub_text:
            return output

        feature = feature_kwargs["feature"]
        feature = f"{key} {feature}"
        feature_kwargs["feature"] = feature

        base_messages = deepcopy(base_messages)
        base_messages[-2]["content"] = EXTRACT_ORIGINAL_TEXT_PROMPT.format(
            feature=feature, wes_type=feature_kwargs["wes_type"]
        )
        base_messages[-1]["content"] = sub_text

        values = await self._extract_setback_values(
            sub_text,
            base_messages=base_messages,
            **feature_kwargs,
        )
        output.update(values)
        return output

    async def _extract_setback_values(self, text, **kwargs):
        """Extract setback values for a particular feature from input text."""
        dtree_out = await self._run_setback_graph(
            setup_multiplier, text, **kwargs
        )

        if dtree_out.get("mult_value") is None:
            return dtree_out

        dtree_con_out = await self._run_setback_graph(
            setup_conditional, text, **kwargs
        )
        dtree_out.update(dtree_con_out)
        return dtree_out

    async def _run_setback_graph(
        self,
        graphs_setup_func,
        text,
        feature,
        wes_type,
        base_messages=None,
        **kwargs,
    ):
        """Generic function to run async tree for ordinance extraction."""
        system_message = SETBACKS_SYSTEM_MESSAGE.format(
            feature=feature, wes_type=wes_type
        )
        tree = _setup_async_decision_tree(
            graphs_setup_func,
            feature=feature,
            text=text,
            chat_llm_caller=self._init_chat_llm_caller(system_message),
            **kwargs,
        )
        if base_messages:
            return await _run_async_tree_with_bm(tree, base_messages)
        return await _run_async_tree(tree)
