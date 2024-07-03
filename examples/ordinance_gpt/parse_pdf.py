import threading
import asyncio
from elm.base import ApiBase
from elm.pdf import PDFtoTXT
from elm.chunk import Chunker
from elm.web.document import PDFDocument
from functools import partial
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elm.ords.services.queues import initialize_service_queue
from elm.ords.services.openai import OpenAIService
from elm.ords.utilities import RTS_SEPARATORS
from elm.ords.process import validate_api_params
from elm.utilities.parse import read_pdf
from elm.ords.services.provider import RunningAsyncServices
from rex import init_logger

from elm.ords.extraction.apply import check_for_ordinance_info


async def run(services, doc, text_splitter, **kwargs):
    async with RunningAsyncServices(services):
        doc = await check_for_ordinance_info(doc, text_splitter, **kwargs)
    return doc


if __name__ == '__main__':
    init_logger(__name__, log_level='DEBUG')
    init_logger('elm', log_level='DEBUG')

    fp = './county_ord_files/Box Elder County, Utah.pdf'

    text_splitter = RecursiveCharacterTextSplitter(
        RTS_SEPARATORS,
        chunk_size=3000,
        chunk_overlap=300,
        length_function=partial(ApiBase.count_tokens, model='gpt-4'),
    )

    doc = PDFDocument.from_file(fp)

    azure_api_key, azure_version, azure_endpoint = validate_api_params()
    client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                     api_version=azure_version,
                                     azure_endpoint=azure_endpoint)
    llm_service = OpenAIService(client, rate_limit=1e9)
    initialize_service_queue(llm_service.__class__.__name__)
    services = [llm_service]

    kwargs = dict(llm_service=llm_service, model='gpt-4', temperature=0,
                  max_tokens=1000)

    doc = asyncio.run(run(services, doc, text_splitter, **kwargs))

    breakpoint()
    raise
    #doc = asyncio.run(check_for_ordinance_info(doc, text_splitter, **kwargs))

    #print(doc.metadata["ordinance_text"])
    #breakpoint()
    #raise

#    kwargs = dict(model="gpt-4",
#                  usage_tracker=None,
#                  usage_sub_label='document_content_validation',
#                  messages=[
#                      {"role": "system", "content": "You are a helpful assistant."},
#                      {"role": "user", "content": "Hello!"}
#                    ],
#                  temperature=0,
#                  max_tokens=1000)
    #asyncio.run(llm_service(


#    kwargs = dict(model="gpt-4",
#                  messages=[
#                      {"role": "system", "content": "You are a helpful assistant."},
#                      {"role": "user", "content": "Hello!"}
#                    ],
#                  temperature=0,
#                  max_tokens=1000)
#    out = asyncio.run(llm_service._call_gpt(**kwargs))
#    print(out)
