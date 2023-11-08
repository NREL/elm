"""This script launches a streamlit app for the Energy Wizard"""
import streamlit as st
import os
import openai
from glob import glob
import pandas as pd

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

    corpus['osti_id'] = corpus['osti_id'].astype(str)
    meta['osti_id'] = meta['osti_id'].astype(str)
    corpus = corpus.set_index('osti_id')
    meta = meta.set_index('osti_id')

    corpus = corpus.join(meta, on='osti_id', rsuffix='_record', how='left')

    ref = [f"{row['title']} ({row['doi']})" for _, row in corpus.iterrows()]
    corpus['ref'] = ref

    return corpus


@st.cache_resource
def get_wizard():
    """Get the energy wizard object."""
    corpus = get_corpus()
    wizard = EnergyWizard(corpus, ref_col='ref', model=model)
    return wizard


if __name__ == '__main__':
    wizard = get_wizard()

    msg = """Hello!\nI am the Energy Wizard. I have access to all NREL
    technical reports from 1-1-2022 to present. Note that each question you ask
    is independent. I am not fully conversational yet like ChatGPT is. Here
    are some examples of questions you can ask me:
    \n - What are some of the key takeaways from the LA100 study?
    \n - What kind of work does NREL do on energy security and resilience?
    \n - Who is working on the reV model?
    \n - Who at NREL has published on capacity expansion analysis?
    \n - Can you teach me the basics of grid inertia versus
        inverter based resources?
    \n - What are some of the unique cyber security challenges facing
    renewables?
    \n - Can you give me some ideas for follow-on research related to climate
    change adaptation with renewables?
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
