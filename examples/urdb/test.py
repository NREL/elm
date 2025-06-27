"""Example on parsing an existing PDF file on-disk for ordinances."""
from functools import partial
import json
import os

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rex import init_logger
from elm.base import ApiBase
from elm.web.document import PDFDocument
from elm.ords.llm import LLMCaller
from elm.ords.services.openai import OpenAIService
from elm.ords.utilities import RTS_SEPARATORS
from elm.ords.process import validate_api_params
from elm.urdb.extraction.rate import RateExtractor
from elm.urdb.extraction.apply import (extract_rate_values,
                                       extract_rate_info)
from elm.ords.services.provider import RunningAsyncServices as ARun
from elm.urdb.extraction.apply import (check_for_rate_info,
                                       extract_rate_text_with_llm)

OUT_DIR = './output/'

def apply_baseline(dict):
    for period in dict.keys():
        dict[period]['baseline_usage'] = round(dict[period]['total_usage'] -
                                            abs(dict[period]['baseline_credit']),
                                            3)
    return dict


if __name__ == '__main__':

    # def apply_baseline(dict):
    #     for period in dict.keys():
    #         dict[period]['baseline_usage'] = round(dict[period]['total_usage'] -
    #                                             abs(dict[period]['baseline_credit']),
    #                                             3)
    #     return dict

    init_logger('elm', log_level='INFO')
    os.makedirs(OUT_DIR, exist_ok=True)

    #fp_pdf = './rates/ELEC_SCHEDS_E-1.pdf'
    # fp_pdf = './rates/ELEC_SCHEDS_A-1.pdf'
    #fp_pdf = './rates/ELEC_SCHEDS_E-TOU-C.pdf'
    fp_pdf = './rates/ELEC_SCHEDS_E-ELEC.pdf'
    #fp_pdf = './rates/ELEC_SCHEDS_E-TOU-D.pdf'
    out_fn = os.path.basename(fp_pdf).replace('.pdf', '.json')
    out_fp = os.path.join(OUT_DIR,  out_fn)


    fp_txt_all = fp_pdf.replace('.pdf', '_all.txt')
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
    kwargs = dict(llm_service=llm_service, model='gpt-4', temperature=0)
    extractor = RateExtractor(LLMCaller(**kwargs))

    # 1) call async func using a partial function (`run_async`)
    run_async = partial(ARun.run, services)
    doc = run_async(check_for_rate_info(doc, text_splitter, **kwargs))

    # 2) Build coroutine first the use it to call async func
    # (extract_ordinance_text_with_llm is an async function)
    extrct = extract_rate_text_with_llm(doc, text_splitter, extractor)
    doc = ARun.run(services, extrct)


    # 3) Build coroutine and use it to call async func in one go
    doc = ARun.run(services, extract_rate_values(doc, **kwargs))
    doc = ARun.run(services, extract_rate_info(doc, **kwargs))

    #breakpoint()

    doc.metadata['rate_info']['effective_date'] = doc.metadata['date']
    all = doc.metadata['rate_info'] | doc.metadata['rate_values']

    breakpoint()

    if (doc.metadata['rate_info']['rate_structure'] == 'Time of Use'):

        periods = ['summer', 'winter']
        
        for p in periods:
            season = doc.metadata['rate_values']['electricity_costs'][p]
            if 'baseline_credit' in season.keys():
                doc.metadata['rate_values']['electricity_costs'][p] = apply_baseline(season)

    with open(out_fp, "w") as outfile: 
        json.dump(all, outfile, indent=2)