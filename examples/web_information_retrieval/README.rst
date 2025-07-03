*************************
Web Information Retrieval
*************************

In this directory, we provide a set of examples that demonstrate how to use ELM to perform web information retrieval tasks.

- [`Custom web information retrieval using search engines`](./example_search_scrape_wiki.ipynb):

    In this example, we demonstrate how to set up your own end-to-end web and information retrieval pipeline.
    Specifically, we set up a procedure to extract the name of NREL's current director using only Wikipedia articles.

    .. NOTE::
        Due to the non-deterministic nature of several pipeline components (Google search, LLM), you may get
        slightly different results when you run this code locally. This is expected and is OK - one of the key
        components of developing your own version of this procedure is making the results as reproducible as
        possible.
