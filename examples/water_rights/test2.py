from elm.web.search import web_search_links_as_docs
import asyncio
from functools import partial
from elm import ApiBase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elm.ords.utilities import RTS_SEPARATORS
import openai
from elm.utilities import validate_azure_api_params
from elm.ords.llm import LLMCaller
from elm.ords.services.openai import OpenAIService
from elm.ords.services.provider import RunningAsyncServices

from elm.water_rights.extraction.apply import (check_for_ordinance_info,
                                       extract_ordinance_text_with_llm,)

QUERIES = ["panola county groundwater conservation district well permits"]
MODEL = "egswaterord-gpt4-mini"

docs = asyncio.run(web_search_links_as_docs(
    QUERIES,
    pdf_read_kwargs={"verbose": False},
    ignore_url_parts={"openei.org"},
    # pw_launch_kwargs={"headless": False, "slow_mo": 1000}
))

breakpoint()

SYSTEM_MESSAGE = (
    "You extract one or more direct excerpts from a given text based on "
    "the user's request. Maintain all original formatting and characters "
    "without any paraphrasing. If the relevant text is inside of a "
    "space-delimited table, return the entire table with the original "
    "space-delimited formatting. Never paraphrase! Only return portions "
    "of the original text directly."
)

INSTRUCTIONS = (
    "Extract one or more direct text excerpts related to the "
    "groundwater conservation district rules. Include "
    "section headers (if any) for the text excerpts. If there is no text "
    "related to the conservation district, simply say: "
    '"No relevant text."'
)


async def extract_relevant_info(doc, text_splitter, llm):
    text_chunks = text_splitter.split_text(doc.text)
    summaries = [
        asyncio.create_task(
            llm.call(
                sys_msg=SYSTEM_MESSAGE,
                content=f"Text:\n{chunk}\n{INSTRUCTIONS}",
            ),
        )
        for chunk in text_chunks
    ]
    summary_chunks = await asyncio.gather(*summaries)
    summary_chunks = [
        chunk for chunk in summary_chunks
        if chunk  # chunk not empty string
        and "no relevant text" not in chunk.lower()  # LLM found relevant info
        and len(chunk) > 20  # chunk is long enough to contain relevant info
    ]
    relevant_text = "\n".join(summary_chunks)
    doc.attrs["relevant_text"] = relevant_text  # store in doc's metadata
    return doc

text_splitter = RecursiveCharacterTextSplitter(
    RTS_SEPARATORS,  # or your own custom set of separators
    chunk_size=3000,  # or your own custom chunk size
    chunk_overlap=300,  # or your own custom chunk overlap
    length_function=partial(ApiBase.count_tokens, model=MODEL),
)

# func below assumes you have API params set as ENV variables
azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
client = openai.AsyncAzureOpenAI(
    api_key=azure_api_key,
    api_version=azure_version,
    azure_endpoint=azure_endpoint,
)

async def main(docs):
    llm = LLMCaller(llm_service=OpenAIService, model=MODEL)
    services = [OpenAIService(client, rate_limit=10000)]
    async with RunningAsyncServices(services):
        tasks = [
            asyncio.create_task(extract_relevant_info(doc, text_splitter, llm))
            for doc in docs
        ]
        results = await asyncio.gather(*tasks)
        return results

if __name__ == "__main__":
    breakpoint()
    results = asyncio.run(main(docs))  # Run the async function

    breakpoint()
    print(results)

