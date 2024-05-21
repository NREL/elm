"""This script launches a streamlit app for the Energy Wizard"""
import streamlit as st
import os
import openai
from glob import glob
import pandas as pd
import sys

from elm import EnergyWizard


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


@st.cache_data
def get_corpus():
    """Get the corpus of text data with embeddings."""
    corpus = sorted(glob('./embed/*.json'))
    corpus = [pd.read_json(fp) for fp in corpus]
    corpus = pd.concat(corpus, ignore_index=True)
    meta = pd.read_csv('./meta.csv')

    corpus['id'] = corpus['id'].astype(str)
    meta['id'] = meta['id'].astype(str)
    corpus = corpus.set_index('id')
    meta = meta.set_index('id')

    corpus = corpus.join(meta, on='id', rsuffix='_record', how='left')

    ref = [f"{row['title']} ({row['url']})" for _, row in corpus.iterrows()]
    corpus['ref'] = ref

    return corpus


@st.cache_resource
def get_wizard():
    """Get the energy wizard object."""

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


if __name__ == '__main__':
    wizard = get_wizard()

    msg = """Hello!\nI am the Energy Wizard - Research Hub edition. I have
    access to the NREL Research Hub which includes researcher profiles as
    well as NREL Publications.Note that each question you ask is independent.
    I am not fully conversational yet like ChatGPT is. Here are some examples
    of questions you can ask me:
    \n - What is 'insert researcher name' position at NREL?
    \n - Which publication has 'researcher name' contributed to?
    \n - Can you summarize 'publication name'?
    \n - Who has experience researching grid resilience?
    \n - Who at NREL has experience with on techno-economic analysis?
    """

    st.title(msg)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    msg = "Type your question here"
    if prompt := st.chat_input(msg):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):

            message_placeholder = st.empty()
            full_response = ""

            out = wizard.chat(prompt,
                              debug=True, stream=True, token_budget=6000,
                              temperature=0.0, print_references=True,
                              convo=False, return_chat_obj=True)
            references = out[-1]

            for response in out[0]:
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")

            ref_msg = ('\n\nThe wizard was provided with the '
                       'following documents to support its answer:')
            ref_msg += '\n - ' + '\n - '.join(references)
            full_response += ref_msg

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant",
                                          "content": full_response})
