*****************
The Energy Wizard
*****************

This example demonstrates how to download a set of PDFs, convert to text,
chunk, embed, and then run a streamlit app that interfaces an LLM with the text
corpus.

Notes:

- In this example, we use the optional `popper <https://poppler.freedesktop.org/>`_ PDF utility which you will have to install separately. You can also use the python-native ``PyPDF2`` package when calling using ``elm.pdf.PDFtoTXT`` but we have found that poppler works better.

- Streamlit is required to run this app, which is not an explicit requirement of this repo (``pip install streamlit``)

- You need to set up your own OpenAI or Azure-OpenAI API keys to run the scripts.

Downloading and Embedding PDFs
==============================

Run ``python ./retrieve_docs.py`` to retrieve 20 of the latest NREL technical
reports from OSTI. The script then converts the PDFs to text and then runs the
text through the OpenAI embedding model.

Running the Streamlit App
=========================

Run ``streamlit run ./run_app.py`` to start the streamlit app. You can now chat
with the Energy Wizard, which will interface with the downloaded text corpus to
answer your questions.
