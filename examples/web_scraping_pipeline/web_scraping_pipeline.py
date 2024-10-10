"""ELM web scraping and info extraction pipeline demo. """

import asyncio
from functools import partial

import openai
import pandas as pd
import networkx as nx
from langchain.text_splitter import RecursiveCharacterTextSplitter

from elm import ApiBase
from elm.utilities import validate_azure_api_params
from elm.web.google_search import google_results_as_docs, filter_documents
from elm.ords.llm import LLMCaller, ChatLLMCaller
from elm.ords.services.openai import OpenAIService
from elm.ords.services.provider import RunningAsyncServices
from elm.ords.extraction.tree import AsyncDecisionTree
from elm.ords.utilities import RTS_SEPARATORS, llm_response_as_json


QUERIES = ["NREL wiki", "National Renewable Energy Laboratory director"]
SYSTEM_MESSAGE = (
    "You extract one or more direct excerpts from a given text based on "
    "the user's request. Maintain all original formatting and characters "
    "without any paraphrasing. If the relevant text is inside of a "
    "space-delimited table, return the entire table with the original "
    "space-delimited formatting. Never paraphrase! Only return portions "
    "of the original text directly."
)
INSTRUCTIONS = (
    "Extract one or more direct text excerpts related to leadership at NREL. "
    "Be sure to include any relevant names and position titles. Include "
    "section headers (if any) for the text excerpts. If there is no text "
    "related to leadership at NREL, simply say: "
    '"No relevant text."'
)
CHAT_SYSTEM_MESSAGE = (
    "You are a researcher extracting information from wikipedia articles. "
    "Always answer based off of the given text, and never use prior knowledge."
)


async def url_is_wiki(doc):
    return "wiki" in doc.metadata.get("source", "")


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
    doc.metadata["relevant_text"] = relevant_text  # store in doc's metadata
    return doc


def setup_decision_tree_graph(text, chat_llm_caller):
    G = nx.DiGraph(text=text, chat_llm_caller=chat_llm_caller)
    G.add_node(
        "init",
        prompt=(
            "Does the following text mention the National Renewable Energy "
            "Laboratory (NREL)?  Begin your response with either 'Yes' or "
            "'No' and justify your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )
    G.add_edge(
        "init", "leadership", condition=lambda x: x.lower().startswith("yes")
    )
    # Can add a branch for the "No" response if we want, but not required
    # since we catch `RuntimeErrors` below.
    G.add_node(
        "leadership",
        prompt=(
            "Does the following text mention who the current director of "
            "the National Renewable Energy Laboratory (NREL) is? Begin "
            "your response with either 'Yes' or 'No' and justify your answer."
            '\n\n"""\n{text}\n"""'
        ),
    )
    G.add_edge(
        "leadership", "name", condition=lambda x: x.lower().startswith("yes")
    )

    G.add_node(
        "name",
        prompt=(
            "Based on the text, who is the current director of the National "
            "Renewable Energy Laboratory (NREL)?"
            '\n\n"""\n{text}\n"""'
        ),
    )
    G.add_edge("name", "final")  # no condition - always go to the end
    G.add_node(
        "final",
        prompt=(
            "Respond based on our entire conversation so far. Return your "
            "answer in JSON format (not markdown). Your JSON file must "
            'include exactly two keys. The keys are "director" and '
            '"explanation". The value of the "director" key should '
            "be a string containing the name of the current director of NREL "
            'as mentioned in the text. The value of the "explanation" '
            "key should be a string containing a short explanation for your "
            "answer."
        ),
    )
    return G


async def extract_final_values(doc, chat_llm):

    G = setup_decision_tree_graph(
        text=doc.metadata["relevant_text"], chat_llm_caller=chat_llm
    )
    tree = AsyncDecisionTree(G)

    try:
        response = await tree.async_run()
    except RuntimeError:  # raised if the tree "condition" is not met
        response = None
    response = llm_response_as_json(response) if response else {}
    response.update(doc.metadata)
    return response


async def run_pipeline():
    docs = await google_results_as_docs(QUERIES)
    docs = await filter_documents(docs, url_is_wiki)

    model = "lmev-gpt-4"
    text_splitter = RecursiveCharacterTextSplitter(
        RTS_SEPARATORS,  # or your own custom set of separators
        chunk_size=3000,  # or your own custom chunk size
        chunk_overlap=300,  # or your own custom chunk overlap
        length_function=partial(ApiBase.count_tokens, model=model),
    )

    azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
    client = openai.AsyncAzureOpenAI(
        api_key=azure_api_key,
        api_version=azure_version,
        azure_endpoint=azure_endpoint,
    )

    llm = LLMCaller(llm_service=OpenAIService, model=model)
    chat_llm = ChatLLMCaller(
        llm_service=OpenAIService,
        system_message=CHAT_SYSTEM_MESSAGE,
        model=model
    )
    services = [OpenAIService(client, rate_limit=4000)]

    async with RunningAsyncServices(services):
        tasks = [
            asyncio.create_task(extract_relevant_info(doc, text_splitter, llm))
            for doc in docs
        ]
        docs = await asyncio.gather(*tasks)

        tasks = [
            asyncio.create_task(extract_final_values(doc, chat_llm))
            for doc in docs
        ]
        info_dicts = await asyncio.gather(*tasks)

    return pd.DataFrame(info_dicts)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_pipeline())
