from elm.water_rights.process import process_county
from elm.web.search import web_search_links_as_docs
import asyncio

from functools import partial

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rex import init_logger
from elm.base import ApiBase
from elm.web.document import PDFDocument
from elm.utilities import validate_azure_api_params
from elm.ords.llm import LLMCaller
from elm.ords.services.openai import OpenAIService
from elm.ords.utilities import RTS_SEPARATORS
from elm.water_rights.extraction.ordinance import OrdinanceExtractor
from elm.ords.extraction.apply import extract_ordinance_values
from elm.ords.services.provider import RunningAsyncServices as ARun
from elm.water_rights.extraction.apply import (check_for_ordinance_info,
                                       extract_ordinance_text_with_llm,
                                       extract_ordinance_text_with_ngram_validation,)


QUERIES = ['panola county groundwater conservation district',
           "panola county groundwater conservation district well permits",
           "panola county groundwater conservation district well requirements",
           "panola county groundwater conservation district rules"]

MODEL = "egswaterord-gpt4-mini"

if __name__ == '__main__':
    init_logger('elm', log_level='DEBUG')
    loop = asyncio.get_event_loop()
    docs = asyncio.run(web_search_links_as_docs(
        QUERIES,
        pdf_read_kwargs={"verbose": False},
        ))

    breakpoint()
    # doc = docs[0]

    text_splitter = RecursiveCharacterTextSplitter(
        RTS_SEPARATORS,
        chunk_size=3000,
        chunk_overlap=300,
        length_function=partial(ApiBase.count_tokens, model=MODEL),
    )

    azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
    client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                     api_version=azure_version,
                                     azure_endpoint=azure_endpoint)
    llm_service = OpenAIService(client, rate_limit=1e9)
    services = [llm_service]
    kwargs = dict(llm_service=llm_service, model=MODEL,
                  temperature=0)
    extractor = OrdinanceExtractor(LLMCaller(**kwargs))

    """The following three function calls present three (equivalent) ways to
    call ELM async ordinance functions. The three functions 1) check ordinance
    documents for relevant ordinance info, 2) extract the relevant text, and 3)
    run the decision tree to get structured ordinance data from the
    unstructured legal text."""

    # 1) call async func using a partial function (`run_async`)
    run_async = partial(ARun.run, services)
    doc = run_async(check_for_ordinance_info(doc, text_splitter, **kwargs))

    breakpoint()

    # 2) Build coroutine first the use it to call async func
    # (extract_ordinance_text_with_llm is an async function)
    #extract = extract_ordinance_text_with_llm(doc, text_splitter, extractor)
    # extract = extract_ordinance_text_with_ngram_validation(doc, text_splitter, extractor)
    doc = run_async(extract_ordinance_text_with_llm(doc, text_splitter, extractor))
    # doc = ARun.run(services, extract)

    breakpoint()

    # 3) Build coroutine and use it to call async func in one go
    doc = ARun.run(services, extract_ordinance_values(doc, **kwargs))

    breakpoint()