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
#from elm.ords.extraction.features import SetbackFeatures
from elm.water_rights.extraction.graphs import (
    EXTRACT_ORIGINAL_TEXT_PROMPT,
    setup_graph_fluids,
    setup_graph_temperature,
    setup_graph_permits,
    llm_response_starts_with_yes,
)

## extra imports for the wizard
from elm import EnergyWizard
import openai
import os
from glob import glob

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a legal scholar explaining legal ordinances to a geothermal "
    "energy developer."
)
# SETBACKS_SYSTEM_MESSAGE = (
#     f"{DEFAULT_SYSTEM_MESSAGE} "
#     "For the duration of this conversation, only focus on "
#     "ordinances relating to setbacks from {feature} for {wes_type}. Ignore "
#     "all text that pertains to private, micro, small, or medium sized wind "
#     "energy systems."
# )
# RESTRICTIONS_SYSTEM_MESSAGE = (
#     f"{DEFAULT_SYSTEM_MESSAGE} "
#     "For the duration of this conversation, only focus on "
#     "ordinances relating to {restriction} for {wes_type}. Ignore "
#     "all text that pertains to private, micro, small, or medium sized wind "
#     "energy systems."
# )
# EXTRA_RESTRICTIONS_TO_CHECK = {
#     "noise": "maximum noise level",
#     "max height": "maximum turbine height",
#     "min lot size": "minimum lot size",
#     "shadow flicker": "maximum shadow flicker",
#     "density": "maximum turbine spacing",
# }


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
    """LLM ordinance document structured data scraping utility."""

    def _init_chat_llm_caller(self, system_message):
        """Initialize a ChatLLMCaller instance for the DecisionTree"""
        return ChatLLMCaller(
            self.llm_service,
            system_message=system_message,
            usage_tracker=self.usage_tracker,
            **self.kwargs,
        )
    
    def _init_wizard(self):
        azure_client = openai.AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version = "2024-10-21",
            azure_endpoint = 'https://aoai-prod-eastus-egswaterord-001.openai.azure.com/'#os.getenv("AZURE_OPENAI_ENDPOINT") 
            )

        corpus = self.get_corpus()
        
        wizard = EnergyWizard(corpus, azure_client=azure_client, model='egswaterord-openai-embedding')

        return wizard

    def get_corpus(self):
        """Get the corpus of text data with embeddings."""
        # corpus = sorted(glob('./embed/*.json'))
        fp = '/Users/spodgorn/Library/CloudStorage/OneDrive-NREL/Desktop/ELM/elm/examples/water_rights/embed/*.json'
        corpus = sorted(glob(fp))
        corpus = [pd.read_json(fp) for fp in corpus]
        corpus = pd.concat(corpus, ignore_index=True)

        return corpus

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
        values = {}
        # fluid = await self._check_fluid_type(text)
        # logger.info("Definition type found in text: %s", fluid)
        # temp = await self._check_temperature(text)
        # logger.info("Definition type found in text: %s", temp)

        # values['fluid'] = fluid
        # values['temperature'] = temp
        
        reqs = await self._check_reqs(text)
        logger.info("Definition type found in text: %s", reqs)
        values['requirements'] = reqs

        breakpoint()

        return values

    async def _check_reqs(self, text):
        """Get the fluid type mentioned in the text."""
        logger.debug("Checking fluid types")

        prompt = setup_graph_permits().nodes['init'].get('db_query')
        wizard = self._init_wizard()
        response = wizard.query_vector_db(prompt)
        text = response[0].tolist()
        all_text = '\n'.join(text)
        
        breakpoint()

        tree = _setup_async_decision_tree(
            setup_graph_permits,
            text=all_text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )

        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out
    
    async def _check_fluid_type(self, text):
        """Get the fluid type mentioned in the text."""
        logger.debug("Checking fluid types")
        tree = _setup_async_decision_tree(
            setup_graph_fluids,
            text=text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )
        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out
    
    async def _check_temperature(self, text):
        """Get the largest turbine size mentioned in the text."""
        logger.debug("Checking turbine_types")
        tree = _setup_async_decision_tree(
            setup_graph_temperature,
            text=text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )
        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out
    
    