"""RAG set up for water rights"""
from elm.web.search import web_search_links_as_docs
import asyncio
import os
import pandas as pd
import time
import logging
import openai

from rex import init_logger

from elm.embed import ChunkAndEmbed
from elm.chunk import Chunker

logger = logging.getLogger(__name__)
init_logger(__name__, log_level='DEBUG')
init_logger('elm', log_level='INFO')

openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_type = 'azure'
# openai.api_version = os.getenv("AZURE_OPENAI_VERSION")
openai.api_version = '2024-08-01'

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


# GWCD_NAME = 'Panhandle'
GWCD_NAME = 'Trinity Glen Rose'
QUERIES = [f"{GWCD_NAME} groundwater conservation district",
           f"{GWCD_NAME} groundwater conservation district well permits",
           f"{GWCD_NAME} groundwater conservation district well requirements",
           f"{GWCD_NAME} groundwater conservation district rules"]

MODEL = 'egswaterord-openai-embedding'

EMBED_DIR = f"./{GWCD_NAME.lower().replace(' ', '_')}_embed_250/"

if __name__ == '__main__':
    init_logger('elm', log_level='DEBUG')
    os.makedirs(EMBED_DIR, exist_ok=True)
    loop = asyncio.get_event_loop()
    docs = asyncio.run(web_search_links_as_docs(
        QUERIES,
        pdf_read_kwargs={"verbose": False},
        ))
    breakpoint()
    client = openai.AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"), 
        api_version = os.getenv('AZURE_OPENAI_VERSION'),
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
        )

    for i, d in enumerate(docs):
        url = d.attrs.get('source')
        fn = os.path.basename(url)
        if not fn:
            fn =  url.replace('https://', '')
            fn = fn.replace('/', '-')
    
        fn = fn.replace('.pdf', '').split('.')[0]
        fn = f'{fn}_{i}'
        assert fn, f'no file name generated for {url}'
        embed_fp = os.path.join(EMBED_DIR, fn)
        embed_fp = embed_fp + '.json'

        if not os.path.exists(embed_fp):
            logger.info(f'Embedding {url}')
            # obj = ChunkAndEmbed(d.text, model=MODEL, tokens_per_chunk=500, overlap=1)
            obj = ChunkAndEmbed(d.text, client, model=MODEL, tokens_per_chunk=250, overlap=1)
            try:
                embeddings = asyncio.run(obj.run_async(rate_limit=3e4))
                if any(e is None for e in embeddings):
                    raise RuntimeError('Embeddings are None!')
                else:
                    df = pd.DataFrame({'text': obj.text_chunks.chunks,
                                    'embedding': embeddings,})
                    df.to_json(embed_fp, indent=2)
                    logger.info(f'Saving {embed_fp}')
            except Exception as e:
                logger.info(f'could not embed {fn} with error: {e}')

            time.sleep(5)

