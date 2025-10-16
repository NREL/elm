"""
ELM mixture of experts
"""
import streamlit as st
import os
import openai
from glob import glob
import pandas as pd
import sys


from elm.base import ApiBase
from elm.wizard import EnergyWizard
from elm.db_wiz import DataBaseWizard

model = 'gpt-4'

# NREL-Azure endpoint. You can also use just the openai endpoint.
# NOTE: embedding values are different between OpenAI and Azure models!
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_type = 'azure'
openai.api_version = os.getenv('AZURE_OPENAI_VERSION')

EnergyWizard.EMBEDDING_MODEL = 'text-embedding-ada-002-2'
EnergyWizard.EMBEDDING_URL = ('https://stratus-embeddings-south-central.'
                              'openai.azure.com/openai/deployments/'
                              'text-embedding-ada-002-2/embeddings?'
                              f'api-version={openai.api_version}')
EnergyWizard.URL = ('https://stratus-embeddings-south-central.'
                    'openai.azure.com/openai/deployments/'
                    f'{model}/chat/completions?'
                    f'api-version={openai.api_version}')
EnergyWizard.HEADERS = {"Content-Type": "application/json",
                        "Authorization": f"Bearer {openai.api_key}",
                        "api-key": f"{openai.api_key}"}

EnergyWizard.MODEL_ROLE = ('You are a energy research assistant. Use the '
                           'articles below to answer the question. If '
                           'articles do not provide enough information to '
                           'answer the question, say "I do not know."')
EnergyWizard.MODEL_INSTRUCTION = EnergyWizard.MODEL_ROLE

DataBaseWizard.URL = (
    f'https://stratus-embeddings-south-central.openai.azure.com/'
    f'openai/deployments/{model}/chat/'
    f'completions?api-version={openai.api_version}')
DataBaseWizard.HEADERS = {"Content-Type": "application/json",
                          "Authorization": f"Bearer {openai.api_key}",
                          "api-key": f"{openai.api_key}"}

st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache_data
def get_corpus():
    """Get the corpus of text data with embeddings."""
    corpus = sorted(glob('./embed/*.json'))
    corpus = [pd.read_json(fp) for fp in corpus]
    corpus = pd.concat(corpus, ignore_index=True)

    return corpus


@st.cache_resource
def get_wizard(model=model):
    """Get the energy wizard object.

    Parameters
    ----------
    model : str
        State which model to use for the energy wizard.

    Returns
    -------
    response : str
        GPT output / answer.
    wizard : EnergyWizard
        Returns the energy wizard object for use in chat responses.
    """

    # Getting Corpus of data. If no corpus throw error for user.
    try:
        corpus = get_corpus()
    except Exception:
        print("Error: Have you run 'retrieve_docs.py'?")
        st.header("Error")
        st.write("Error: Have you run 'retrieve_docs.py'?")
        sys.exit(0)

    wizard = EnergyWizard(corpus, ref_col='ref', model=model)
    return wizard


class MixtureOfExperts(ApiBase):
    """Interface to ask OpenAI LLMs about energy
    research either from a database or report."""

    MODEL_ROLE = ("You are an expert given a query. Which of the "
                  "following best describes the query? Please "
                  "answer with just the number and nothing else."
                  "1. This is a query best answered by a text-based report."
                  "2. This is a query best answered by pulling data from "
                  "a database and creating a figure.")
    """High level model role, somewhat redundant to MODEL_INSTRUCTION"""

    def __init__(self, db_wiz, txt_wiz, model=None):
        self.wizard_db = db_wiz
        self.wizard_chat = txt_wiz
        super().__init__(model)

    def chat(self, query,
             stream=True,
             temperature=0):
        """Answers a query by doing a semantic search of relevant text with
        embeddings and then sending engineered query to the LLM.

        Parameters
        ----------
        query : str
            Question being asked of EnergyWizard
        stream : bool
            Flag to print subsequent chunks of the response in a streaming
            fashion
        temperature : float
            GPT model temperature, a measure of response entropy from 0 to 1. 0
            is more reliable and nearly deterministic; 1 will give the model
            more creative freedom and may not return as factual of results.

        Returns
        -------
        response : str
            GPT output / answer.
        """

        messages = [{"role": "system", "content": self.MODEL_ROLE},
                    {"role": "user", "content": query}]
        response_message = ''
        kwargs = dict(model=self.model,
                      messages=messages,
                      temperature=temperature,
                      stream=stream)

        response = self._client.chat.completions.create(**kwargs)

        print(response)

        if stream:
            for chunk in response:
                chunk_msg = chunk.choices[0].delta.content or ""
                response_message += chunk_msg
                print(chunk_msg, end='')

        else:
            response_message = response["choices"][0]["message"]["content"]

        message_placeholder = st.empty()
        full_response = ""

        if '1' in response_message:
            out = self.wizard_chat.chat(query,
                                        debug=True, stream=True,
                                        token_budget=6000, temperature=0.0,
                                        print_references=True, convo=False,
                                        return_chat_obj=True)

            for response in out[0]:
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")

        elif '2' in response_message:
            out = self.wizard_db.chat(query,
                                      debug=True, stream=True,
                                      token_budget=6000, temperature=0.0,
                                      print_references=True, convo=False,
                                      return_chat_obj=True)

            st.pyplot(fig=out, clear_figure=False)

        else:
            response_message = 'Error cannot find data in report or database.'

        return full_response
