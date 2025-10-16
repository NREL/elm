"""Run the mixture of experts streamlit app"""
import streamlit as st
from elm.experts import MixtureOfExperts

model = 'gpt-4'
# User defined connection string
conn_string = ''

if __name__ == '__main__':
    wizard = MixtureOfExperts(model=model, connection_string=conn_string)

    msg = ("""Multi-Modal Wizard Demonstration!\nI am a multi-modal AI
           demonstration. I have access to NREL technical reports regarding the
           LA100 study and access to several LA100 databases. If you ask me a
           question, I will attempt to answer it using the reports or the
           database. Below are some examples of queries that have been shown to
           work.
    \n - Describe chapter 2 of the LA100 report.
    \n - What are key findings of the LA100 report?
    \n - What enduse consumes the most electricity?
    \n - During the year 2020 which geographic regions consumed the
    most electricity?
    """)

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

            out = wizard.chat(query=prompt,
                              debug=True, stream=True, token_budget=6000,
                              temperature=0.0, print_references=True,
                              convo=False, return_chat_obj=True)

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant",
                                          "content": full_response})
