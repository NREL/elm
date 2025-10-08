# -*- coding: utf-8 -*-
"""ELM Ordinance structured parsing class."""
import asyncio
import logging

from elm.ords.llm.calling import BaseLLMCaller, ChatLLMCaller
from elm.ords.utilities import llm_response_as_json
from elm.ords.extraction.tree import AsyncDecisionTree
from elm.water_rights.extraction.graphs import (
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
)

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_MESSAGE = (
    "You are a legal expert explaining water rights ordinances to a geothermal "
    "energy developer."
)

def _setup_async_decision_tree(graph_setup_func, **kwargs):
    """Setup Async Decision tree dor ordinance extraction."""
    G = graph_setup_func(**kwargs)
    tree = AsyncDecisionTree(G)
    assert len(tree.chat_llm_caller.messages) == 1
    return tree

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

    def __init__(self, wizard, location, **kwargs):
        """
        Parameters
        ----------
        wizard : elm.wizard.EnergyWizard
            Instance of the EnergyWizard class used for RAG.
        location : str
            Name of the groundwater conservation district or county.
        """
        self.location = location
        self.wizard = wizard

        super().__init__(**kwargs)

    async def parse(self, location):
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

        tasks = {name: asyncio.create_task(func()) for name, func in check_map.items()}

        limit_intervals = ["daily", "monthly", "annual"]
        for interval in limit_intervals:
            task_name = f"{interval}_limits"
            tasks[task_name] = asyncio.create_task(self._check_limits(interval))

        logger.debug("Starting value extraction with %d tasks.", len(tasks))

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning("Task %s failed: %s", key, result)
                values[key] = None
            else:
                values[key] = result

        logger.debug("Value extraction complete.")

        return values

    async def _check_with_graph(self, graph_setup_func,
                                limit=50, **format_kwargs):
        """Generic method to check requirements using a graph setup function.
        
        Parameters
        ----------
        wizard : elm.wizard.EnergyWizard
            Instance of the EnergyWizard class used for RAG.
        graph_setup_func : callable
            Function that returns a graph for the decision tree
        limit : int, optional
            Limit for vector DB query, by default 50
        **format_kwargs
            Additional keyword arguments for prompt formatting
        
        Returns
        -------
        dict
            Extracted data as JSON dict
        """
        graph = graph_setup_func()
        prompt = graph.nodes['init'].get('db_query')

        format_dict = {'DISTRICT_NAME': self.location}
        format_dict.update(format_kwargs)
        prompt = prompt.format(**format_dict)

        response, _, _ = self.wizard.query_vector_db(prompt, limit=limit)
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
        )

    async def _check_extraction(self):
        """Get the extraction requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_extraction,
        )

    async def _check_geothermal(self):
        """Get the geothermal requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_geothermal,
        )

    async def _check_oil_and_gas(self):
        """Get the oil and gas requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_oil_and_gas,
        )

    async def _check_limits(self, interval):
        """Get the extraction limits mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_limits,
            interval=interval
        )

    async def _check_spacing(self):
        """Get the spacing requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_well_spacing,
        )

    async def _check_time(self):
        """Get the time requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_time,
        )

    async def _check_metering_device(self):
        """Get the metering device mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_metering_device,
        )

    async def _check_district_drought(self):
        """Get the drought management plan mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_drought,
        )

    async def _check_well_drought(self):
        """Get the well drought management plan mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_contingency,
        )

    async def _check_plugging(self):
        """Get the plugging requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_plugging_reqs,
        )

    async def _check_transfer(self):
        """Get the transfer requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_external_transfer,
        )

    async def _check_production_reporting(self):
        """Get the reporting requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_production_reporting,
        )

    async def _check_production_cost(self):
        """Get the production cost requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_production_cost,
        )

    async def _check_setbacks(self):
        """Get the setback requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_setback_features,
        )

    async def _check_redrilling(self):
        """Get the redrilling requirements mentioned in the text."""
        return await self._check_with_graph(
            setup_graph_redrilling,
        )
