"""
Code to build Corpus from the researcher hub.
"""
import os
import os.path
import asyncio
from glob import glob
import logging
import time
import pandas as pd
import openai
from rex import init_logger


from elm.pdf import PDFtoTXT
from elm.embed import ChunkAndEmbed
from elm.web_scraping.rhub import ResearcherProfiles
from elm.web_scraping.rhub import ResearchOutputs

# initialize logger
logger = logging.getLogger(__name__)
init_logger(__name__, log_level='DEBUG')
init_logger('elm', log_level='INFO')

# set openAI variables
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
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

if __name__ == '__main__':
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TXT_DIR, exist_ok=True)
    os.makedirs(EMBED_DIR, exist_ok=True)

    rp = ResearcherProfiles('https://research-hub.nrel.gov/en/persons/')
    pubs = ResearchOutputs('https://research-hub.nrel.gov/en/publications/')

    rp.scrape_profiles(TXT_DIR)
    pubs.scrape_publications(PDF_DIR, TXT_DIR)

    profiles_meta = rp.build_meta()
    pubs_meta = pubs.build_meta()

    pubs_meta['fn'] = pubs_meta.apply(lambda row: os.path.basename(row['pdf_url'])
                                      if row['category'] == 'Technical Report'
                                      and row['pdf_url'].endswith('.pdf')
                                      else os.path.basename(row['url']) +
                                      '_abstract.txt', axis=1)
    pubs_meta['fp'] = pubs_meta.apply(lambda row: PDF_DIR + row['fn']
                                      if row['category'] == 'Technical Report'
                                      and row['pdf_url'].endswith('.pdf')
                                      else TXT_DIR + row['fn'], axis=1)

    profiles_meta['fp'] = TXT_DIR + profiles_meta['fn']

    meta = pd.concat([profiles_meta, pubs_meta], axis=0, ignore_index=True)
    meta = meta.drop_duplicates(subset=['nrel_id'])
    meta.to_csv('./meta.csv', index=False)

    logger.info('Meta file saved to {}/meta.csv'.format(os.getcwd()))

    missing = []
    for i, row in meta.iterrows():
        if not os.path.exists(row['fp']):
            missing.append(i)
    meta = meta.drop(missing, axis=0)

    for i, row in meta.iterrows():
        fp = row['fp']
        txt_fp = os.path.join(TXT_DIR, row['fn'].replace('.pdf', '.txt'))
        embed_fp = os.path.join(EMBED_DIR,
                                row['fn'].replace('.pdf', '.json')
                                .replace('.txt', '.json'))

        assert os.path.exists(fp), f'{fp} does not exist'

        if os.path.exists(txt_fp):
            logger.info(f'Opening:{txt_fp}')
            try:
                with open(txt_fp, 'r') as f:
                    text = f.read()
            except UnicodeDecodeError as e:
                with open(txt_fp, 'r',  encoding='cp1252') as f:
                    text = f.read()
            except Exception as e:
                logger.info(f'Could not open {txt_fp}.')
        else:
            try:
                pdf_obj = PDFtoTXT(fp)
                text = pdf_obj.clean_poppler(layout=True)
                if pdf_obj.is_double_col():
                    text = pdf_obj.clean_poppler(layout=False)
                text = pdf_obj.clean_headers(char_thresh=0.6, page_thresh=0.8,
                                            split_on='\n',
                                            iheaders=[0, 1, 3, -3, -2, -1])
                with open(txt_fp, 'w') as f:
                    f.write(text)
                logger.info(f'Saved: {txt_fp}')
            except Exception as e:
                logger.info('Failed to convert {} to text.'\
                            'With error: {}.'.format(fp, e))
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
                                   'nrel_id': row['nrel_id']})
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

    logger.info('Finished!')
