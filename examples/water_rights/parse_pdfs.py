"""Example on parsing an existing PDF file on-disk for ordinances."""
from functools import partial
import json
import pandas as pd
import re
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rex import init_logger

### remove this
# import sys
# import os
# sys.path[-1] = '/Users/spodgorn/Library/CloudStorage/OneDrive-NREL/Desktop/ELM/elm'

from elm.base import ApiBase
from elm.web.document import PDFDocument
from elm.ords.llm import LLMCaller
from elm.ords.services.openai import OpenAIService
from elm.ords.utilities import RTS_SEPARATORS
from elm.utilities import validate_azure_api_params
#######
from elm.water_rights.extraction.ordinance import OrdinanceExtractor
from elm.water_rights.extraction.apply import extract_ordinance_values
from elm.ords.services.provider import RunningAsyncServices as ARun
from elm.water_rights.extraction.apply import (check_for_ordinance_info,
                                               extract_ordinance_text_with_llm)

def clean_text(input_string):
    cleaned = re.sub(r'[^\w\s]', '', input_string)  
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


if __name__ == '__main__':
    init_logger('elm', log_level='DEBUG')

    records = []
    # download this from https://statutes.capitol.texas.gov/Docs/NR/pdf/NR.141.pdf
    #fp_pdf = './data/NR.141.pdf'
    #fp_pdf = './data/chapter-19-public-lands-2024.pdf'
    fps = os.listdir('./data')#['./data/NR.141.pdf', './data/chapter-19-public-lands-2024.pdf']

    breakpoint()

    for f in fps:
        fp_pdf = os.path.join('./data', f)
        fp_txt_all = fp_pdf.replace('.pdf', '_all.txt')
        fp_txt_clean = fp_pdf.replace('.pdf', '_clean.txt')
        fp_ords = fp_pdf.replace('.pdf', '_rights.json')

        doc = PDFDocument.from_file(fp_pdf)

        text_splitter = RecursiveCharacterTextSplitter(
            RTS_SEPARATORS,
            chunk_size=3000,
            chunk_overlap=300,
            length_function=partial(ApiBase.count_tokens, model='c2cchat-openai-gpt4'),
        )

        # setup LLM and Ordinance service/utility classes
        azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
        client = openai.AsyncAzureOpenAI(api_key=azure_api_key,
                                        api_version=azure_version,
                                        azure_endpoint=azure_endpoint)
        llm_service = OpenAIService(client, rate_limit=1e9)
        services = [llm_service]
        kwargs = dict(llm_service=llm_service, model='c2cchat-openai-gpt4', temperature=0)

        extractor = OrdinanceExtractor(LLMCaller(**kwargs))

        """The following three function calls present three (equivalent) ways to
        call ELM async ordinance functions. The three functions 1) check ordinance
        documents for relevant ordinance info, 2) extract the relevant text, and 3)
        run the decision tree to get structured ordinance data from the
        unstructured legal text."""

        # 1) call async func using a partial function (`run_async`)
        run_async = partial(ARun.run, services)
        doc = run_async(check_for_ordinance_info(doc, text_splitter, **kwargs))

        #breakpoint()

        # 2) Build coroutine first the use it to call async func
        # (extract_ordinance_text_with_llm is an async function)
        extrct = extract_ordinance_text_with_llm(doc, text_splitter, extractor)
        doc = ARun.run(services, extrct)

        #breakpoint()

        # 3) Build coroutine and use it to call async func in one go
        doc = ARun.run(services, extract_ordinance_values(doc, **kwargs))

        records.append({
            'state': doc.metadata['state'].pop() if doc.metadata['state'] else None,
            'definition': clean_text(doc.metadata.get('definition_ordinance_text', None)),
            'fluids': doc.metadata['ordinance_values']['fluid'].get('fluid', None),
            'temperature': doc.metadata['ordinance_values']['temperature'].get('temperature', None),
            'threshold_units': doc.metadata['ordinance_values']['temperature'].get('units', None)
        })

    breakpoint()

    df = pd.DataFrame(records)
    df.to_csv('defs_subset.csv', index=False)
        # with open(fp_ords, "w") as outfile: 
        #     json.dump(doc.metadata['ordinance_values'], outfile, indent=2)


        # # save outputs
        # doc.metadata['ordinance_values'].to_csv(fp_ords)
        # with open(fp_txt_all, 'w') as f:
        #     f.write(doc.metadata["ordinance_text"])
        # with open(fp_txt_clean, 'w') as f:
        #     f.write(doc.metadata["cleaned_ordinance_text"])
