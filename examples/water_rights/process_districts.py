"""RAG set up for water rights"""
from functools import partial
import asyncio
import os
import pandas as pd
import time
import logging
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rex import init_logger

from elm.base import ApiBase
from elm.ords.utilities import RTS_SEPARATORS
from elm.ords.services.openai import OpenAIService
from elm.ords.services.threaded import TempFileCache
from elm.ords.services.provider import RunningAsyncServices as ARun
from elm.utilities import validate_azure_api_params

from elm.water_rights.download import download_county_ordinance
from elm.water_rights.extraction.apply import extract_ordinance_values
from elm.water_rights.utilities.location import WaterDistrict

from elm.embed import ChunkAndEmbed
from elm.chunk import Chunker
from elm import EnergyWizard

logger = logging.getLogger(__name__)
init_logger(__name__, log_level='DEBUG')
init_logger('elm', log_level='DEBUG')

openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_type = 'azure'
openai.api_version = os.getenv("AZURE_OPENAI_VERSION")

ChunkAndEmbed.EMBEDDING_MODEL = 'egswaterord-openai-embedding'

ChunkAndEmbed.EMBEDDING_URL =('https://aoai-prod-eastus-egswaterord-001.'
                              'openai.azure.com/openai/deployments?api-'
                              f'version={openai.api_version}')

ChunkAndEmbed.URL= ('https://aoai-prod-eastus-egswaterord-001.'
                    'openai.azure.com/openai/deployments?api-'
                     f'version={openai.api_version}')
Chunker.URL= ('https://aoai-prod-eastus-egswaterord-001.'
              'openai.azure.com/openai/deployments?api-'
              f'version={openai.api_version}')

ChunkAndEmbed.EMBEDDING_TYPE ='azure new'

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
                           'developers understand water permitting and access. '
                           'Use the information below to answer the question.')
EnergyWizard.MODEL_INSTRUCTION = EnergyWizard.MODEL_ROLE
EnergyWizard.EMBEDDING_MODEL = 'egswaterord-openai-embedding'
EnergyWizard.EMBEDDING_TYPE = 'azure new'


EMBEDDING_MODEL = 'egswaterord-openai-embedding'
MODEL = 'egswaterord-gpt4.1-mini'

async def process(location, text_splitter, client, **kwargs):
    """Download water rights documents for a location."""

    llm_service_rate_limit =  500000
    td_kwargs = dict(dir=".")
    tpe_kwargs = dict(max_workers= 10)
    # azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
    # client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
    #                                  api_version=azure_version,
    #                                  azure_endpoint=azure_endpoint)

    services = [
        OpenAIService(client, rate_limit=llm_service_rate_limit),
        TempFileCache(td_kwargs=td_kwargs, tpe_kwargs=tpe_kwargs)
    ]

    async with ARun(services):
        docs = await download_county_ordinance(location,
                                              text_splitter,
                                              **kwargs)

    return docs

if __name__ == '__main__':
    districts = pd.read_csv('districts.csv')
    names = districts['district'].tolist()[:2]

    results = pd.DataFrame()
    for dis in names:
        gwcd_name = dis.strip()
        logger.info(f'Processing {gwcd_name}')
        text_splitter = RecursiveCharacterTextSplitter(
            RTS_SEPARATORS,
            chunk_size=3000,
            chunk_overlap=300,
            length_function=partial(ApiBase.count_tokens, model='gpt-4'),
        )
        azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
        client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                        api_version=azure_version,
                                        azure_endpoint=azure_endpoint)
        llm_service = OpenAIService(client, rate_limit=5e5)
        services = [llm_service]
        kwargs = dict(llm_service=llm_service, model=MODEL, temperature=0)
        location = WaterDistrict(name=gwcd_name, state='Texas')

        docs = asyncio.run(process(location, text_splitter, client, **kwargs))

        corpus = pd.DataFrame(columns=['text', 'embedding'])
        for i, d in enumerate(docs):
            url = d.attrs.get('source')
            logger.info(f'Embedding {url}')
            obj = ChunkAndEmbed(d.text, client, model=EMBEDDING_MODEL,
                                tokens_per_chunk=500, overlap=1, split_on='\n')
            try:
                embeddings = asyncio.run(obj.run_async(rate_limit=3e4))
                if any(e is None for e in embeddings):
                    raise RuntimeError('Embeddings are None!')
                else:
                    row = {'text': obj.text_chunks.chunks,
                           'embedding': embeddings,
                           }
                    corpus = pd.concat([corpus, pd.DataFrame(row)],
                                       ignore_index=True)
            except Exception as e:
                logger.info(f'could not embed {url} with error: {e}')
            time.sleep(5)
        
        if len(corpus) == 0:
            logger.info(f'No documents returned for {gwcd_name}, skipping')
            continue

        extraction_client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                        api_version=azure_version,
                                        azure_endpoint=azure_endpoint)
        llm_service = OpenAIService(extraction_client, rate_limit=5e5)
        services = [llm_service]
        kwargs = dict(llm_service=llm_service, model=MODEL, temperature=0)
        
        wizard = EnergyWizard(corpus, azure_client=extraction_client,
                              model='egswaterord-gpt4-mini')

        try:
            values = ARun.run(services,
                              extract_ordinance_values(wizard=wizard,
                                                       location=dis,
                                                       **kwargs))
            results = pd.concat([results, pd.DataFrame(values)],
                                ignore_index=True)
        except ValueError:
            continue

    results.to_csv('wr_results.csv', index=False)
    logger.info('finished')
