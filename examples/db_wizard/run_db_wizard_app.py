"""Run the db wizard example streamlit app"""
import streamlit as st
import os
import openai

from elm.db_wiz import DataBaseWizard

model = 'gpt-4'
conn_string = ('postgresql://la100_admin:laa5SSf6KOC6k9xl'
               '@gds-cluster-1.cluster-ccklrxkcenui'
               '.us-west-2.rds.amazonaws.com:5432/la100-stage')

openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'

DataBaseWizard.URL = (
    f'https://stratus-embeddings-south-central.openai.azure.com/'
    f'openai/deployments/{model}/chat/'
    f'completions?api-version={openai.api_version}')
DataBaseWizard.HEADERS = {"Content-Type": "application/json",
                          "Authorization": f"Bearer {openai.api_key}",
                          "api-key": f"{openai.api_key}"}

st.set_option('deprecation.showPyplotGlobalUse', False)


if __name__ == '__main__':
    wizard = DataBaseWizard(model=model, connection_string=conn_string)

    opening_message = '''Hello! \n I am the Database Wizard. I
    Have access to a single database. You can ask me questions
    about the data and ask me to produce visualizations of the data.
    Here are some examples of what you can ask me:
    \n - Plot a time series of the winter residential
        heating load for the moderate scenario
        in model year 2030 for geography 1.
    \n - Plot a time series of the winter
        residential heating load for the moderate scenario
        in model year 2030 for the first five load centers.
    '''

    st.title(opening_message)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        if st.button('Clear Chat'):
            # Clearing Messages
            st.session_state.messages = []
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Clearing Wizard
            wizard.clear()
            wizard = DataBaseWizard(model=model, connection_string=conn_string)

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

            st.pyplot(fig=out, clear_figure=False)
