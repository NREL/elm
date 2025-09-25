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
    setup_graph_extraction,
    setup_graph_geothermal,
    setup_graph_oil_and_gas,
    setup_graph_limits,
    setup_graph_well_spacing,
    setup_graph_time,
    setup_graph_metering_device,
    setup_graph_drought,
    setup_graph_contingency,
    setup_graph_plugging_reqs,
    setup_graph_external_transfer,
    setup_graph_production_reporting,
    setup_graph_production_cost,
    setup_graph_setback_features,
    setup_graph_redrilling,
    llm_response_starts_with_yes,
)

## extra imports for the wizard
from elm import EnergyWizard
import openai
from glob import glob

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_MESSAGE = (
    "You are a legal expert explaining water rights ordinances to a geothermal "
    "energy developer."
)

# TODO: should I import all of these from elm.ords.extraction.parse
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
    
    async def parse(self, wizard, location):
        """Parse text and extract structured ordinance data."""
        self.location = location
        values = {"location": location}

        check_map = {
            "requirements": self._check_reqs,
            "extraction_requirements": self._check_extraction,
            "well_spacing": self._check_spacing,
            "drilling_window": self._check_time,
            "metering_device": self._check_metering_device,
            "district_drought_mgmt_plan": self._check_district_drought,
            "well_drought_mgmt_plan": self._check_well_drought,
            "plugging_requirements": self._check_plugging,
            "transfer_requirements": self._check_transfer,
            "production_reporting": self._check_production_reporting,
            "production_cost": self._check_production_cost,
            "setbacks": self._check_setbacks,
            "redrilling": self._check_redrilling,
        }

        tasks = {name: func(wizard) for name, func in check_map.items()}

        limit_intervals = ["daily", "monthly", "annual"]
        for interval in limit_intervals:
            tasks[f"{interval}_limits"] = self._check_limits(wizard, interval)

        logger.debug("Starting value extraction.")

        results = await asyncio.gather(*tasks.values())

        for key, result in zip(tasks.keys(), results):
            values[key] = result

        logger.debug("Value extraction complete.")
        return values
    
    async def _check_with_graph(self, wizard, graph_setup_func,
                                check_name, limit=50, **format_kwargs):
        """Generic method to check requirements using a graph setup function.
        
        Parameters
        ----------
        wizard : elm.EnergyWizard # TODO confirm type
            Instance of the EnergyWizard class used for RAG.
        graph_setup_func : callable
            Function that returns a graph for the decision tree
        check_name : str
            Name of what's being checked (for logging)
        limit : int, optional
            Limit for vector DB query, by default 50
        **format_kwargs
            Additional keyword arguments for prompt formatting
        
        Returns
        -------
        dict
            Extracted data as JSON dict
        """
        logger.debug(f"Checking {check_name}")

        graph = graph_setup_func()
        prompt = graph.nodes['init'].get('db_query')
        
        format_dict = {'DISTRICT_NAME': self.location}
        format_dict.update(format_kwargs)
        prompt = prompt.format(**format_dict)

        response, _, _ = wizard.query_vector_db(prompt, limit=limit)
        text = response.tolist()
        all_text = '\n'.join(text)

        tree = _setup_async_decision_tree(
            graph_setup_func,
            text=all_text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
            **(format_kwargs or {})
        )

        return await _run_async_tree(tree)
    
    async def _check_reqs(self):
        """Get the requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_permits, 
            "requirements"
        )
    
    async def _check_extraction(self):
        """Get the extraction requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_extraction, 
            "extraction"
        )
    
    async def _check_geothermal(self):
        """Get the geothermal requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_geothermal, 
            "geothermal requirements"
        )

    async def _check_oil_and_gas(self):
        """Get the oil and gas requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_oil_and_gas, 
            "oil and gas requirements"
        )
    
    async def _check_limits(self, interval):
        """Get the extraction limits mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_limits, 
            f"{interval} extraction limits",
            interval=interval
        )
    
    async def _check_spacing(self):
        """Get the spacing requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_well_spacing, 
            "spacing requirements",
            limit=20
        )

    async def _check_time(self):
        """Get the time requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_time, 
            "time limits"
        )
    
    async def _check_metering_device(self):
        """Get the metering device mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_metering_device, 
            "metering device requirements"
        )
    
    async def _check_district_drought(self):
        """Get the drought management plan mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_drought, 
            "drought management plan"
        )
    
    async def _check_well_drought(self):
        """Get the well drought management plan mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_contingency, 
            "well drought management plan"
        )

    async def _check_plugging(self):
        """Get the plugging requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_plugging_reqs, 
            "plugging requirements"
        )

    async def _check_transfer(self):
        """Get the transfer requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_external_transfer, 
            "transfer requirements"
        )
    
    async def _check_production_reporting(self):
        """Get the reporting requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_production_reporting, 
            "production reporting requirements"
        )
    
    async def _check_production_cost(self):
        """Get the production cost requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_production_cost, 
            "production cost requirements"
        )

    async def _check_setbacks(self):
        """Get the setback requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_setback_features, 
            "setback requirements"
        )
    
    async def _check_redrilling(self):
        """Get the redrilling requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_redrilling, 
            "redrilling requirements"
        )
    