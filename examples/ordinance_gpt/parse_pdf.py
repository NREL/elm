import asyncio
from elm.base import ApiBase
from elm.web.document import PDFDocument
from functools import partial
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elm.ords.services.queues import initialize_service_queue
from elm.ords.services.openai import OpenAIService
from elm.ords.utilities import RTS_SEPARATORS
from elm.ords.process import validate_api_params
from elm.ords.llm import LLMCaller
from elm.ords.extraction.ordinance import OrdinanceExtractor
from elm.ords.extraction.apply import extract_ordinance_values
from elm.ords.services.provider import RunningAsyncServices
from rex import init_logger

from elm.ords.extraction.apply import (check_for_ordinance_info,
                                       extract_ordinance_text_with_llm)


async def clean_text(services, doc, **kwargs):
    text_splitter = RecursiveCharacterTextSplitter(
        RTS_SEPARATORS,
        chunk_size=3000,
        chunk_overlap=300,
        length_function=partial(ApiBase.count_tokens, model='gpt-4'),
    )
    async with RunningAsyncServices(services):
        extractor = OrdinanceExtractor(LLMCaller(**kwargs))
        doc = await check_for_ordinance_info(doc, text_splitter, **kwargs)
        doc = await extract_ordinance_text_with_llm(doc, text_splitter,
                                                    extractor)
    return doc


async def extract_ordinances(services, doc, **kwargs):
    async with RunningAsyncServices(services):
        doc = await extract_ordinance_values(doc, **kwargs)
    return doc


if __name__ == '__main__':
    init_logger('elm', log_level='INFO')

    fp_pdf = './examples/ordinance_gpt/county_ord_files/Box Elder County, Utah.pdf'

    fp_txt_all = fp_pdf.replace('.pdf', '.txt')
    fp_txt_clean = fp_pdf.replace('.pdf', '_clean.txt')
    fp_ords = fp_pdf.replace('.pdf', '_ords.csv')

    doc = PDFDocument.from_file(fp_pdf)

    azure_api_key, azure_version, azure_endpoint = validate_api_params()
    client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                     api_version=azure_version,
                                     azure_endpoint=azure_endpoint)
    llm_service = OpenAIService(client, rate_limit=1e9)
    initialize_service_queue(llm_service.__class__.__name__)
    services = [llm_service]
    kwargs = dict(llm_service=llm_service, model='gpt-4', temperature=0)

    doc = asyncio.run(clean_text(services, doc, **kwargs))
    doc = asyncio.run(extract_ordinances(services, doc, **kwargs))

    doc.metadata['ordinance_values'].to_csv(fp_ords)
    with open(fp_txt_all, 'w') as f:
        f.write(doc.metadata["ordinance_text"])
    with open(fp_txt_clean, 'w') as f:
        f.write(doc.metadata["cleaned_ordinance_text"])
