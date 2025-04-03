"""This script launches a streamlit app for the Energy Wizard"""
import streamlit as st
import os
import openai
from glob import glob
import pandas as pd
import sys

import logging
from rex import init_logger
from elm.ords.services.openai import OpenAIService
from elm.utilities import validate_azure_api_params
from elm import EnergyWizard
from elm.ords.utilities import llm_response_as_json
from elm.ords.extraction.tree import AsyncDecisionTree
from elm.water_rights.extraction.graphs import (
    EXTRACT_ORIGINAL_TEXT_PROMPT,
    setup_graph_fluids,
    setup_graph_temperature,
    setup_graph_permits,
    llm_response_starts_with_yes,
)
from elm.water_rights.extraction.apply import extract_ordinance_values
from elm.ords.services.provider import RunningAsyncServices as ARun
from elm.water_rights.extraction.parse import StructuredOrdinanceParser
from elm.water_rights.extraction.apply import (check_for_ordinance_info,
                                       extract_ordinance_text_with_llm,
                                       extract_ordinance_text_with_ngram_validation,)

logger = logging.getLogger(__name__)
init_logger(__name__, log_level='DEBUG')
init_logger('elm', log_level='INFO')

MODEL = 'egswaterord-openai-embedding'

# NREL-Azure endpoint. You can also use just the openai endpoint.
# NOTE: embedding values are different between OpenAI and Azure models!
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_type = 'azure'
openai.api_version = os.getenv('AZURE_OPENAI_VERSION')

# EnergyWizard.EMBEDDING_MODEL = 'text-embedding-ada-002-2'
# EnergyWizard.EMBEDDING_URL = ('https://stratus-embeddings-south-central.'
#                               'openai.azure.com/openai/deployments/'
#                               'text-embedding-ada-002-2/embeddings?'
#                               f'api-version={openai.api_version}')
EnergyWizard.EMBEDDING_URL =('https://aoai-prod-eastus-egswaterord-001.'
                              'openai.azure.com/openai/deployments?api-'
                              f'version={openai.api_version}')

EnergyWizard.URL = ('https://stratus-embeddings-south-central.'
                    'openai.azure.com/openai/deployments/'
                    f'{MODEL}/chat/completions?'
                    f'api-version={openai.api_version}')
EnergyWizard.HEADERS = {"Content-Type": "application/json",
                        "Authorization": f"Bearer {openai.api_key}",
                        "api-key": f"{openai.api_key}"}

EnergyWizard.MODEL_ROLE = ('You are a water rights expert. Use the '
                           'information below to answer the question. If '
                           'documents do not provide enough information to '
                           'answer the question, say "I do not know."')
EnergyWizard.MODEL_INSTRUCTION = EnergyWizard.MODEL_ROLE



def get_corpus():
    """Get the corpus of text data with embeddings."""
    corpus = sorted(glob('./embed/*.json'))
    corpus = [pd.read_json(fp) for fp in corpus]
    corpus = pd.concat(corpus, ignore_index=True)
    # meta = pd.read_csv('./meta.csv')

    # corpus['osti_id'] = corpus['osti_id'].astype(str)
    # meta['osti_id'] = meta['osti_id'].astype(str)
    # corpus = corpus.set_index('osti_id')
    # meta = meta.set_index('osti_id')

    # corpus = corpus.join(meta, on='osti_id', rsuffix='_record', how='left')

    # ref = [f"{row['title']} ({row['doi']})" for _, row in corpus.iterrows()]
    # corpus['ref'] = ref

    return corpus

def get_wizard():
    """Get the energy wizard object."""

    # Getting Corpus of data. If no corpus throw error for user.
    try:
        corpus = get_corpus()
        azure_client = openai.AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version = "2023-05-15",
            azure_endpoint = 'https://aoai-prod-eastus-egswaterord-001.openai.azure.com/'#os.getenv("AZURE_OPENAI_ENDPOINT") 
            )
    except Exception:
        logger.info("Error: Have you run 'retrieve_docs.py'?")

        sys.exit(0)

    wizard = EnergyWizard(corpus, azure_client=azure_client, model=MODEL)
    return wizard 

azure_client = openai.AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2023-05-15",
  azure_endpoint = 'https://aoai-prod-eastus-egswaterord-001.openai.azure.com/'#os.getenv("AZURE_OPENAI_ENDPOINT") 
)


if __name__ == '__main__':

    # query = 'Does the following text mention restrictions related to well spacing?'
    query = ("Does the following text mention water well permit requirements? "
            "Requirements should specify whether or not an application is required "
            "in order to drill a groundwater well.")

    wizard = get_wizard()

    # relevant_info = wizard.query_vector_db(query)

    azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
    client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                     api_version=azure_version,
                                     azure_endpoint=azure_endpoint)
    
    breakpoint()
    llm_service = OpenAIService(client, rate_limit=1e9)
    services = [llm_service]
    kwargs = dict(llm_service=llm_service, model=MODEL,
                  temperature=0)
    
    # test = setup_graph_permits(**kwargs)
    
    # text = wizard.query_vector_db(query)

    values = ARun.run(services, extract_ordinance_values(**kwargs))

    breakpoint()

    # d_tree question -> query_vector_db -> return relevant text -> decision_tree -> repeat for each question