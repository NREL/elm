"""Example on parsing an existing PDF file for ordinances. """
from functools import partial

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rex import init_logger
from elm.base import ApiBase
from elm.web.document import PDFDocument
from elm.ords.llm import LLMCaller
from elm.ords.services.openai import OpenAIService
from elm.ords.utilities import RTS_SEPARATORS
from elm.ords.process import validate_api_params
from elm.ords.extraction.ordinance import OrdinanceExtractor
from elm.ords.extraction.apply import extract_ordinance_values
from elm.ords.services.provider import RunningAsyncServices
from elm.ords.extraction.apply import (check_for_ordinance_info,
                                       extract_ordinance_text_with_llm)


if __name__ == '__main__':
    init_logger('elm', log_level='INFO')

    fp_pdf = ('./examples/ordinance_gpt/county_ord_files/'
              'Box Elder County, Utah.pdf')

    fp_txt_all = fp_pdf.replace('.pdf', '.txt')
    fp_txt_clean = fp_pdf.replace('.pdf', '_clean.txt')
    fp_ords = fp_pdf.replace('.pdf', '_ords.csv')

    doc = PDFDocument.from_file(fp_pdf)

    text_splitter = RecursiveCharacterTextSplitter(
        RTS_SEPARATORS,
        chunk_size=3000,
        chunk_overlap=300,
        length_function=partial(ApiBase.count_tokens, model='gpt-4'),
    )

    azure_api_key, azure_version, azure_endpoint = validate_api_params()
    client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                     api_version=azure_version,
                                     azure_endpoint=azure_endpoint)
    llm_service = OpenAIService(client, rate_limit=1e9)
    services = [llm_service]
    run_async = partial(RunningAsyncServices.run, services)

    kwargs = dict(llm_service=llm_service, model='gpt-4', temperature=0)
    extractor = OrdinanceExtractor(LLMCaller(**kwargs))

    # Three (equivalent) ways to call async ordinance functions:

    # call async func using a partial function (`run_async`)
    doc = run_async(check_for_ordinance_info(doc, text_splitter, **kwargs))

    # Build coroutine first the use it to call async func
    extract_text = extract_ordinance_text_with_llm(doc, text_splitter,
                                                   extractor)
    doc = RunningAsyncServices.run(services, extract_text)

    # Build coroutine and use it to call async func in one go
    doc = RunningAsyncServices.run(services,
                                   extract_ordinance_values(doc, **kwargs))

    doc.metadata['ordinance_values'].to_csv(fp_ords)
    with open(fp_txt_all, 'w') as f:
        f.write(doc.metadata["ordinance_text"])
    with open(fp_txt_clean, 'w') as f:
        f.write(doc.metadata["cleaned_ordinance_text"])
