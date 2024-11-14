***************************************
Web Scraping and information Extraction
***************************************

In this example, we demonstrate how to set up your own end-to-end web scraping and information extraction pipeline.
Specifically, we set up a procedure to extract the name of NREL's current director using only Wikipedia articles.


.. NOTE:: Due to the non-deterministic nature of several pipeline components (Google search, LLM), you may get
          slightly different results when you run this code locally. This is expected and is OK - one of the key
          components of developing your own version of this procedure is making the results as reproducible as
          possible.
