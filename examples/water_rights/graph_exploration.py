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

import openai
from elm.utilities import validate_azure_api_params
from elm.ords.services.openai import OpenAIService

def _setup_async_decision_tree(graph_setup_func, **kwargs):
    """Setup Async Decision tree dor ordinance extraction."""
    G = graph_setup_func(**kwargs)
    # tree = AsyncDecisionTree(G)
    # assert len(tree.chat_llm_caller.messages) == 1
    # return tree
    return G

def init_chat_llm_caller():#system_message):
        """Initialize a ChatLLMCaller instance for the DecisionTree"""
        return ChatLLMCaller(
            # llm_service,
            # system_message=system_message,
            # system_message,
            usage_tracker=None,
            **kwargs,
        )

DEFAULT_SYSTEM_MESSAGE = (
    "You are a legal scholar explaining legal ordinances to a geothermal "
    "energy developer."
)

MODEL = "egswaterord-gpt4-mini"

if __name__ == "__main__":

    azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
    client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                     api_version=azure_version,
                                     azure_endpoint=azure_endpoint)
    llm_service = OpenAIService(client, rate_limit=1e9)
    services = [llm_service]
    kwargs = dict(llm_service=llm_service, model=MODEL,
                  temperature=0, system_message=DEFAULT_SYSTEM_MESSAGE)

    tree = _setup_async_decision_tree(setup_graph_permits,
                                      text='text',
                                      chat_llm_caller=init_chat_llm_caller()#DEFAULT_SYSTEM_MESSAGE),
                                      )

    # tree = setup_graph_permits()


    breakpoint()