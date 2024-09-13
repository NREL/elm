"""
Code to build Corpus from the researcher hub.
"""
import os
import asyncio
import pandas as pd
import logging
import openai
import time
from glob import glob
from rex import init_logger

from elm.pdf import PDFtoTXT
from elm.embed import ChunkAndEmbed
from elm.web.rhub import ProfilesList
from elm.web.rhub import PublicationsList


logger = logging.getLogger(__name__)
init_logger(__name__, log_level='INFO')
init_logger('elm', log_level='INFO')


# NREL-Azure endpoint. You can also use just the openai endpoint.
# NOTE: embedding values are different between OpenAI and Azure models!
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'

ChunkAndEmbed.EMBEDDING_MODEL = 'text-embedding-ada-002-2'
ChunkAndEmbed.EMBEDDING_URL = ('https://stratus-embeddings-south-central.'
                               'openai.azure.com/openai/deployments/'
                               'text-embedding-ada-002-2/embeddings?'
                               f'api-version={openai.api_version}')
ChunkAndEmbed.HEADERS = {"Content-Type": "application/json",
                         "Authorization": f"Bearer {openai.api_key}",
                         "api-key": f"{openai.api_key}"}

PDF_DIR = './pdfs/'
TXT_DIR = './txt/'
EMBED_DIR = './embed/'

rhub_api_key = os.getenv("RHUB_API_KEY")
PROFILES_URL = (f'https://research-hub.nrel.gov/ws/api'
                f'/524/persons?order=lastName'
                f'&pageSize=20&apiKey={rhub_api_key}')
PUBLICATIONS_URL = (f'https://research-hub.nrel.gov/ws/api'
                    f'/524/research-outputs?'
                    f'order=publicationYear&'
                    f'orderBy=descending&pageSize=20&'
                    f'apiKey={rhub_api_key}')


if __name__ == '__main__':
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TXT_DIR, exist_ok=True)
    os.makedirs(EMBED_DIR, exist_ok=True)

    profiles = ProfilesList(PROFILES_URL, n_pages=10)
    logger.info("Starting download for researcher profiles.")
    profiles.download(TXT_DIR)
    profiles_meta = profiles.meta()

    publications = PublicationsList(PUBLICATIONS_URL, n_pages=20)
    logger.info("Starting download for publications.")
    publications.download(PDF_DIR, TXT_DIR)
    pubs_meta = publications.meta()

    pdf_categories = ['Technical Report', 'Paper', 'Fact Sheet']

    pubs_meta['fn'] = pubs_meta.apply(lambda row:
                                      row['id'] + '.pdf'
                                      if row['category'] in pdf_categories
                                      and row['pdf_url'] is not None
                                      and row['pdf_url'].endswith('.pdf')
                                      else row['id'] + '.txt', axis=1)
    pubs_meta['fp'] = pubs_meta.apply(lambda row:
                                      PDF_DIR + row['id'] + '.pdf'
                                      if row['category'] in pdf_categories
                                      and row['pdf_url'] is not None
                                      and row['pdf_url'].endswith('.pdf')
                                      else TXT_DIR + row['fn'], axis=1)

    profiles_meta['fp'] = TXT_DIR + profiles_meta['fn']

    meta = pd.concat([profiles_meta, pubs_meta], axis=0, ignore_index=True)
    meta = meta.drop_duplicates(subset=['id'])
    meta.to_csv('./meta.csv', index=False)

    logger.info('Meta file saved to {}/meta.csv'.format(os.getcwd()))

    missing = ~meta['fp'].apply(os.path.exists)
    meta = meta[~missing]

    failed_count = 0

    for i, row in meta.iterrows():
        fp = row['fp']
        txt_fp = os.path.join(TXT_DIR, row['fn'].replace('.pdf', '.txt'))
        embed_fp = os.path.join(EMBED_DIR,
                                row['fn'].replace('.pdf', '.json')
                                .replace('.txt', '.json'))

        assert os.path.exists(fp), f'{fp} does not exist'

        if os.path.exists(txt_fp):
            logger.info(f'Opening:{txt_fp}')
            with open(txt_fp, 'r') as f:
                text = f.read()

        else:
            try:
                pdf_obj = PDFtoTXT(fp)
                text = pdf_obj.convert_to_txt(txt_fp)
            except Exception as e:
                failed_count += 1
                logger.info(f'Could not convert {fp} to pdf.')
                continue

        assert os.path.exists(txt_fp)

        if not os.path.exists(embed_fp):
            logger.info('Embedding {}/{}: "{}"'
                        .format(i + 1, len(meta), row['title']))
            tag = f"Title: {row['title']}\nAuthors: {row['authors']}"
            obj = ChunkAndEmbed(text, tag=tag, tokens_per_chunk=500, overlap=1)
            embeddings = asyncio.run(obj.run_async(rate_limit=3e4))
            if any(e is None for e in embeddings):
                raise RuntimeError('Embeddings are None!')
            else:
                df = pd.DataFrame({'text': obj.text_chunks.chunks,
                                   'embedding': embeddings,
                                   'id': row['id']})
                df.to_json(embed_fp, indent=2)
                logger.info('Saved: {}'.format(embed_fp))
            time.sleep(5)

    bad = []
    fps = glob(EMBED_DIR + '*.json')
    for fp in fps:
        data = pd.read_json(fp)
        if data['embedding'].isna().any():
            bad.append(fp)
    assert not any(bad), f'Bad output: {bad}'

    logger.info(f'Finished! Failed to process {failed_count} documents')
