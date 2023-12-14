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
from elm.osti import OstiList


logger = logging.getLogger(__name__)
init_logger(__name__, log_level='DEBUG')
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

URL = ('https://www.osti.gov/api/v1/records?'
       'research_org=NREL'
       '&sort=publication_date%20desc'
       '&product_type=Technical%20Report'
       '&has_fulltext=true'
       '&publication_date_start=01/01/2023'
       '&publication_date_end=12/31/2023')


if __name__ == '__main__':
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(TXT_DIR, exist_ok=True)
    os.makedirs(EMBED_DIR, exist_ok=True)

    osti = OstiList(URL, n_pages=1)
    osti.download(PDF_DIR)

    meta = osti.meta.copy()
    meta['osti_id'] = meta['osti_id'].astype(str)
    meta = meta.drop_duplicates(subset=['osti_id'])
    meta['fp'] = PDF_DIR + meta['fn']
    meta.to_csv('./meta.csv', index=False)

    missing = []
    for i, row in meta.iterrows():
        if not os.path.exists(row['fp']):
            missing.append(i)
    meta = meta.drop(missing, axis=0)

    for i, row in meta.iterrows():
        fp = os.path.join(PDF_DIR, row['fn'])
        txt_fp = os.path.join(TXT_DIR, row['fn'].replace('.pdf', '.txt'))
        embed_fp = os.path.join(EMBED_DIR, row['fn'].replace('.pdf', '.json'))

        assert fp.endswith('.pdf')
        assert os.path.exists(fp)

        if os.path.exists(txt_fp):
            with open(txt_fp, 'r') as f:
                text = f.read()
        else:
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

        if not os.path.exists(embed_fp):
            logger.info('Embedding {}/{}: "{}"'
                        .format(i+1, len(meta), row['title']))
            tag = f"Title: {row['title']}\nAuthors: {row['authors']}"
            obj = ChunkAndEmbed(text, tag=tag, tokens_per_chunk=500, overlap=1)
            embeddings = asyncio.run(obj.run_async(rate_limit=3e4))
            if any(e is None for e in embeddings):
                raise RuntimeError('Embeddings are None!')
            else:
                df = pd.DataFrame({'text': obj.text_chunks.chunks,
                                   'embedding': embeddings,
                                   'osti_id': row['osti_id']})
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
