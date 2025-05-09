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
from elm.utilities import validate_azure_api_params
#from elm.ords.extraction.features import SetbackFeatures
from elm.water_rights.extraction.graphs import (
    EXTRACT_ORIGINAL_TEXT_PROMPT,
    setup_graph_permits,
    setup_graph_daily_limits,
    setup_graph_annual_limits,
    setup_graph_well_spacing,
    setup_graph_time,
    setup_graph_metering_device,
    setup_graph_drought,
    setup_graph_plugging_reqs,
    llm_response_starts_with_yes,
)

## extra imports for the wizard
from elm import EnergyWizard
import openai
from glob import glob

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a legal scholar explaining legal ordinances to a geothermal "
    "energy developer."
)

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
    
    def _init_wizard(self, vector_store):
        azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
        azure_client = openai.AzureOpenAI(
            api_key = azure_api_key,  
            api_version = azure_version,
            azure_endpoint = azure_endpoint
            )

        # TODO: would be ideal to improve this so the vector store doesn't need to be read in every time
        corpus = self.get_corpus(vector_store=vector_store)
        
        # wizard = EnergyWizard(corpus, azure_client=azure_client, model='egswaterord-openai-embedding')
        wizard = EnergyWizard(corpus, azure_client=azure_client, model='egswaterord-gpt4-mini')

        return wizard

    def get_corpus(self, vector_store):
        """Get the corpus of text data with embeddings."""
        # corpus = sorted(glob('./embed/*.json'))
        fp = vector_store
        corpus = sorted(glob(fp))
        corpus = [pd.read_json(fp) for fp in corpus]
        corpus = pd.concat(corpus, ignore_index=True)

        return corpus

    async def parse(self, vector_store, location):
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
        self.location = location
        
        # reqs = await self._check_reqs(vector_store)
        # logger.info("Requirements found in text: %s", reqs)
        # values['requirements'] = reqs


        daily_lims = await self._check_daily_limits(vector_store)
        logger.info("Definition type found in text: %s", daily_lims)
        values['daily_limits'] = daily_lims
        breakpoint()

        # annual_lims = await self._check_annual_limits(vector_store)
        # logger.info("Definition type found in text: %s", annual_lims)
        # values['annual_limits'] = annual_lims

        # well_spacing = await self._check_spacing(vector_store)
        # logger.info("Definition type found in text: %s", well_spacing)
        # values['well_spacing'] = well_spacing

        # time = await self._check_time(vector_store)
        # logger.info("Definition type found in text: %s", time)
        # values['drilling_window'] = time
        
        # metering_device = await self._check_metering_device(vector_store)
        # logger.info("Definition type found in text: %s", metering_device)
        # values['metering_device'] = metering_device

        # drought = await self._check_drought(vector_store)
        # logger.info("Definition type found in text: %s", drought)
        # values['drought_mgmt_plan'] = drought
        
        # plugging = await self._check_plugging(vector_store)
        # logger.info("Definition type found in text: %s", plugging)
        # values['plugging_requirements'] = plugging

        return values    

    async def _check_reqs(self, vector_store):
        """Get the requirements mentioned in the text."""
        logger.debug("Checking requirements")

        prompt = setup_graph_permits().nodes['init'].get('db_query')
        prompt = prompt.format(DISTRICT_NAME=self.location)
        wizard = self._init_wizard(vector_store)
        response, _, idx = wizard.query_vector_db(prompt)
        text = response.tolist()
        all_text = '\n'.join(text)

        tree = _setup_async_decision_tree(
            setup_graph_permits,
            text=all_text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )

        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out
    
    async def _check_daily_limits(self, vector_store):
        """Get the extraction limits mentioned in the text."""
        logger.debug("Checking daily extraction limits")

        prompt = setup_graph_daily_limits().nodes['init'].get('db_query')
        wizard = self._init_wizard(vector_store)
        response, _, idx = wizard.query_vector_db(prompt)
        text = response.tolist()
        all_text = '\n'.join(text)

        breakpoint()

        tree = _setup_async_decision_tree(
            setup_graph_daily_limits,
            text=all_text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )

        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out
    
    async def _check_annual_limits(self, vector_store):
        """Get the extraction limits mentioned in the text."""
        logger.debug("Checking annual extraction limits")

        prompt = setup_graph_annual_limits().nodes['init'].get('db_query')
        wizard = self._init_wizard(vector_store)
        response, _, idx = wizard.query_vector_db(prompt)
        text = response.tolist()
        all_text = '\n'.join(text)

        tree = _setup_async_decision_tree(
            setup_graph_annual_limits,
            text=all_text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )

        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out
    
    async def _check_spacing(self, vector_store):
        """Get the spacing requirements mentioned in the text."""
        logger.debug("Checking spacing requirements")

        prompt = setup_graph_well_spacing().nodes['init'].get('db_query')
        wizard = self._init_wizard(vector_store)
        response, _, idx = wizard.query_vector_db(prompt)
        text = response.tolist()
        all_text = '\n'.join(text)

        tree = _setup_async_decision_tree(
            setup_graph_well_spacing,
            text=all_text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )

        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out

    async def _check_time(self, vector_store):
        """Get the time requirements mentioned in the text."""
        logger.debug("Checking time limits")

        prompt = setup_graph_time().nodes['init'].get('db_query')
        wizard = self._init_wizard(vector_store)
        response, _, idx = wizard.query_vector_db(prompt)
        text = response.tolist()
        all_text = '\n'.join(text)

        tree = _setup_async_decision_tree(
            setup_graph_time,
            text=all_text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )

        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out
    
    async def _check_metering_device(self, vector_store):
        """Get the metering device mentioned in the text."""
        logger.debug("Checking metering device requirements")

        prompt = setup_graph_metering_device().nodes['init'].get('db_query')
        wizard = self._init_wizard(vector_store)
        response, _, idx = wizard.query_vector_db(prompt)
        text = response.tolist()
        all_text = '\n'.join(text)

        breakpoint()

        tree = _setup_async_decision_tree(
            setup_graph_metering_device,
            text=all_text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )

        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out
    
    async def _check_drought(self, vector_store):
        """Get the drought management plan mentioned in the text."""
        logger.debug("Checking drought management plan")

        prompt = setup_graph_drought().nodes['init'].get('db_query')
        wizard = self._init_wizard(vector_store)
        response, _, idx = wizard.query_vector_db(prompt)
        text = response.tolist()
        all_text = '\n'.join(text)

        tree = _setup_async_decision_tree(
            setup_graph_drought,
            text=all_text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )

        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out

    async def _check_plugging(self, vector_store):
        """Get the plugging requirements mentioned in the text."""
        logger.debug("Checking plugging requirements")

        prompt = setup_graph_plugging_reqs().nodes['init'].get('db_query')
        wizard = self._init_wizard(vector_store)
        response, _, idx = wizard.query_vector_db(prompt)
        text = response.tolist()
        all_text = '\n'.join(text)

        tree = _setup_async_decision_tree(
            setup_graph_plugging_reqs,
            text=all_text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )

        dtree_def_type_out = await _run_async_tree(tree)

        return dtree_def_type_out

    ########
    # TODO: below are questions left over from definition extraction task
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
    