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
from elm.utilities import validate_azure_api_params
from elm.ords.services.openai import OpenAIService
from elm.embed import ChunkAndEmbed
from elm.chunk import Chunker
from elm.ords.llm import StructuredLLMCaller 
from elm.water_rights.download import download_county_ordinance
# from elm.water_rights.utilities.location import County
from elm.water_rights.utilities.location import WaterDistrict
from elm.ords.services.provider import RunningAsyncServices
from elm.ords.services.threaded import TempFileCache

logger = logging.getLogger(__name__)
init_logger(__name__, log_level='DEBUG')
init_logger('elm', log_level='DEBUG')

openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_type = 'azure'
# openai.api_version = os.getenv("AZURE_OPENAI_VERSION")
openai.api_version = '2024-08-01-preview'

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


MODEL = 'egswaterord-openai-embedding'
# EMBED_DIR = f"./{GWCD_NAME.lower().replace(' ', '_')}_embed/"

async def process(location, text_splitter, **kwargs):

    llm_service_rate_limit =  50000
    td_kwargs = dict(dir=".")
    tpe_kwargs = dict(max_workers= 10)
    azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
    client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                     api_version=azure_version,
                                     azure_endpoint=azure_endpoint)

    services = [
        OpenAIService(client, rate_limit=llm_service_rate_limit),
        TempFileCache(td_kwargs=td_kwargs, tpe_kwargs=tpe_kwargs)
    ]

    async with RunningAsyncServices(services):
        docs = await download_county_ordinance(location,
                                              text_splitter,
                                              **kwargs)
        
    return docs

if __name__ == '__main__':
    districts = pd.read_csv('districts.csv')
    # names = districts['district'].tolist()[76:]
    # import random
    # import time
    # names = random.sample(districts['district'].tolist(), 1)

    # names = ['Trinity Glen Rose Groundwater Conservation District',
    # 'Lone Wolf Groundwater Conservation District',
    # 'Cow Creek Groundwater Conservation District',
    # 'Hickory Underground Water Conservation District',
    # 'Comal Trinity Groundwater Conservation District']
    names = districts['district'].tolist()
    # names = [n for n in names if 'southeast' in n.lower()]

    breakpoint()
    times = []
    for dis in names:
        start = time.time()
        gwcd_name = dis.strip()
        logger.info(f'Processing {gwcd_name}')
        embed_dir = f"./embeddings/times/{gwcd_name.lower().replace('/', '_').replace(' ', '_')}_embed/"
        os.makedirs(embed_dir, exist_ok=True)
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
        llm_service = OpenAIService(client, rate_limit=1e9)
        services = [llm_service]
        kwargs = dict(llm_service=llm_service, model='egswaterord-gpt4.1-mini', temperature=0)
        location = WaterDistrict(name=gwcd_name, state='Texas')

        # loop = asyncio.get_event_loop()
        docs = asyncio.run(process(location, text_splitter, **kwargs))

        client = openai.AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"), 
            api_version = os.getenv('AZURE_OPENAI_VERSION'),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
            )
        
        breakpoint()

        for i, d in enumerate(docs):
            url = d.attrs.get('source')
            fn = os.path.basename(url)
            if not fn:
                fn =  url.replace('https://', '')
                fn = fn.replace('/', '-')
        
            fn = fn.replace('.pdf', '').split('.')[0]
            fn = f'{fn}_{i}'
            assert fn, f'no file name generated for {url}'
            embed_fp = os.path.join(embed_dir, fn)
            embed_fp = embed_fp + '.json'

            if not os.path.exists(embed_fp):
                logger.info(f'Embedding {url}')
                # obj = ChunkAndEmbed(d.text, model=MODEL, tokens_per_chunk=500, overlap=1)
                # obj = ChunkAndEmbed(d.text, client, model=MODEL, tokens_per_chunk=250, overlap=1)
                obj = ChunkAndEmbed(d.text, client, model=MODEL, tokens_per_chunk=500, overlap=1, split_on='\n')
                try:
                    embeddings = asyncio.run(obj.run_async(rate_limit=3e4))
                    if any(e is None for e in embeddings):
                        raise RuntimeError('Embeddings are None!')
                    else:
                        df = pd.DataFrame({'text': obj.text_chunks.chunks,
                                        'embedding': embeddings,
                                            'source': url,
                                        })
                        
                        df.to_json(embed_fp, indent=2)
                        logger.info(f'Saving {embed_fp}')
                except Exception as e:
                    logger.info(f'could not embed {fn} with error: {e}')

                time.sleep(5)
        finish = time.time()
        completion_time = finish - start
        times.append(completion_time)
    breakpoint()
    logger.info('finished')
