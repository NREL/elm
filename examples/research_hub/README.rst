********************************
The Energy Wizard - Research Hub
********************************

This example demonstrates how to scrape publication abstracts and researcher profiles,
chunk, embed, and then run a streamlit app that interfaces an LLM with the text
corpus. It is intended for use with the [NREL Research Hub](https://research-hub.nrel.gov/) only. 

Notes:

- Currently this example is only set up to include 10 researchers and 10 publications

- Streamlit is required to run this app, which is not an explicit requirement of this repo (``pip install streamlit``)

- You need to set up your own OpenAI or Azure-OpenAI API keys to run the scripts.

Scraping and Embedding
==============================

Run ``python ./retrieve_docs.py``to scrape research-hub.nrel.gov for both profiles and publications. The script then runs the
text through the OpenAI embedding model.

Running the Streamlit App
=========================

Run ``streamlit run ./run_app.py`` to start the streamlit app. You can now chat
with the Energy Wizard, which will interface with the downloaded text corpus to
answer your questions.
