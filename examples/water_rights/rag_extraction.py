"""This script launches a streamlit app for the Energy Wizard"""
from glob import glob
import os
import openai
import logging
import json
import pandas as pd
from rex import init_logger
from elm.ords.services.openai import OpenAIService
from elm.utilities import validate_azure_api_params
from elm import EnergyWizard


from elm.water_rights.extraction.apply import extract_ordinance_values
from elm.ords.services.provider import RunningAsyncServices as ARun

logger = logging.getLogger(__name__)
init_logger(__name__, log_level='DEBUG')
init_logger('elm', log_level='DEBUG')

# NREL-Azure endpoint. You can also use just the openai endpoint.
# NOTE: embedding values are different between OpenAI and Azure models!
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_type = 'azure'
# openai.api_version = os.getenv('AZURE_OPENAI_VERSION')
openai.api_version = '2024-08-01'

EnergyWizard.EMBEDDING_URL =('https://aoai-prod-eastus-egswaterord-001.'
                              'openai.azure.com/openai/deployments?api-'
                              f'version={openai.api_version}')

EnergyWizard.URL = ('https://stratus-embeddings-south-central.'
                    'openai.azure.com/openai/deployments/'
                    f'gpt4/chat/completions?'
                    f'api-version={openai.api_version}')
EnergyWizard.HEADERS = {"Content-Type": "application/json",
                        "Authorization": f"Bearer {openai.api_key}",
                        "api-key": f"{openai.api_key}"}

EnergyWizard.MODEL_ROLE = ('You are a water rights expert helping geothermal '
                           'developers understand water permitting and access. Use the '
                           'information below to answer the question. If '
                           'documents do not provide enough information to '
                           'answer the question, say "I do not know."')
EnergyWizard.MODEL_INSTRUCTION = EnergyWizard.MODEL_ROLE
EnergyWizard.EMBEDDING_MODEL = 'egswaterord-openai-embedding'
EnergyWizard.EMBEDDING_TYPE = 'azure new'

# MODEL = 'egswaterord-openai-embedding'
# MODEL = 'egswaterord-gpt4-mini'
MODEL = 'egswaterord-gpt4.1-mini'
# GWCD_NAME = 'Lower Trinity'
# GWCD_NAME = 'Panola County'

def get_corpus(fp):
    """Get the corpus of text data with embeddings."""
    corpus = sorted(glob(fp))
    corpus = [pd.read_json(fp) for fp in corpus]
    corpus = pd.concat(corpus, ignore_index=True)

    return corpus

if __name__ == '__main__':
    districts = pd.read_csv('districts.csv')
    # names = [districts['district'].tolist()[74]]
    # names = [n for n in districts['district'].tolist() if 'panhandle' in n.lower()]
    names = districts['district'].tolist()[5:]
    # missing = ['Trinity Glen Rose Groundwater Conservation District',
    # 'Lone Wolf Groundwater Conservation District',
    # 'Cow Creek Groundwater Conservation District',
    # 'Hickory Underground Water Conservation District',
    # 'Comal Trinity Groundwater Conservation District']
    # names = [x for x in names if x in missing]
    azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
    azure_client = openai.AzureOpenAI(
            api_key = azure_api_key,  
            api_version = azure_version,
            azure_endpoint = azure_endpoint
            )
    
    for name in names:
        fp = name.lower().replace('/', '_').replace(' ', '_')
        vector_store = f"./embeddings/{fp}_embed/*.json"
        corpus = get_corpus(vector_store)
        
        wizard = EnergyWizard(corpus, azure_client=azure_client, model='egswaterord-gpt4-mini')


        azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
        client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                        api_version=azure_version,
                                        azure_endpoint=azure_endpoint)

        llm_service = OpenAIService(client, rate_limit=5e5)
        services = [llm_service]
        kwargs = dict(llm_service=llm_service, model=MODEL,
                      temperature=0)
        
        try:
            values = ARun.run(services, extract_ordinance_values(wizard=wizard,
                                                                location=name, **kwargs))
        except ValueError:
            continue

        # out_fp = f'./outputs/{fp}.json'
        # with open(out_fp, 'w') as f:
        #     json.dump(values, f, indent=2)
        breakpoint()
    breakpoint()
