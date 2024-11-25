###############################################
**OrdinanceGPT: Architectural Design Document**
###############################################

*******************
**1. Introduction**
*******************

**1.1 Purpose**
===============
This document describes the architectural design of the ordinance web scraping and extraction tool, focusing on its components, key classes, and their roles within the system.

**1.2 Audience**
================
- **Primary:** Model developers working on expanding the capabilities of ordinance extraction.
- **Secondary:** Model developers extending this functionality to other contexts.

**1.3 Scope**
=============
Covers the OrdinanceGPT design, including key classes, their responsibilities, and interactions.

--------------------------------------------------------------------------------------------------------------------------------------------------

******************************
**2. High-Level Architecture**
******************************

**2.1 System Context**
======================
Points of interaction for OrdinanceGPT:

- **End Users:** Users submit model executions via command-line using a configuration file. Users can select specific jurisdictions to focus on.
- **Internet via Web Browser:** The model searches the web for relevant legal documents. The most common search technique is Google Search.
- **LLMs:** The model relies on LLMs (typically ChatGPT) to analyze web scraping results and subsequently extract information from documents.
- **Filesystem:** Stores output files in organized sub-directories and compiles ordinance information into a CSV.

**Diagram:**

.. mermaid::

    architecture-beta
        group model[OrdinanceGPT]

        service input[User Input]
        service scraper(server)[Web Scraper] in model
        service llm(cloud)[LLM Service]
        service web(internet)[Web]
        service ds(disk)[Document Storage]
        service parser(server)[Document Parser] in model
        service out(database)[Ordinances]

        input:R --> L:scraper
        scraper:T --> L:llm
        scraper:T --> R:web
        scraper:B --> T:ds
        scraper:R --> L:parser
        parser:T --> B:llm
        parser:B --> T:out


--------------------------------------------------------------------------------------------------------------------------------------------------

**********************
**3. Detailed Design**
**********************

**3.1 Web Scraper**
===================
The OrdinanceGPT Web Scraper consists of:

- **Google Search:** Searches Google using pre-determined queries.
- **File Downloader:** Converts Google Search results into documents (PDF or text).
- **Document Validators:** Filters out irrelevant documents.

**Diagram:**

.. uncomment below when sphinxcontrib-mermaid supports mermaid 11.3+
.. .. mermaid::

..     flowchart LR

..         A -->|File Downloader| B
..         B -->|Document Validator| C
..         C --> D
..         A@{ shape: rounded, label: "Google Search" }
..         B@{ shape: docs, label: "Multiple Documents"}
..         C@{ shape: lined-document, label: "Ordinance Document"}
..         D@{ shape: lin-cyl, label: "Disk storage" }

.. mermaid::

    flowchart LR

        A[Google Search] -->|File Downloader| B[Multiple Documents]
        B -->|Document Validator| C[Ordinance Document]
        C --> D[Disk storage]


**3.2 Document Parser**
=======================
The OrdinanceGPT Document Parser consists of:

- **Text Cleaner:** Extract text from ordinance related to data of interest (i.e. wind turbine zoning).
- **Decision Tree:** One decision tree per ordinance value of interest to guide data extraction using LLMs.

**Diagram:**

.. uncomment below when sphinxcontrib-mermaid supports mermaid 11.3+
.. .. mermaid::

..     flowchart LR

..         A -->|Text Cleaner| B
..         B --> C
..         C --> D
..         A@{ shape: lined-document, label: "Ordinance Document"}
..         B@{ shape: hex, label: "Cleaned Text"}
..         C@{ shape: procs, label: "Ordinance value extraction via Decision Tree"}
..         D@{ shape: cyl, label: "Ordinance Database" }


.. mermaid::

    flowchart LR

        A[Ordinance Document] -->|Text Cleaner| B[Cleaned Text]
        B --> C[Ordinance value extraction via Decision Tree]
        C --> D[Ordinance Database]


--------------------------------------------------------------------------------------------------------------------------------------------------

******************************
**4 Key Concepts and Classes**
******************************

**4.1 Key Concept: Services**
=============================

Because OrdinanceGPT is so reliant on LLMs, one of the main design goals is to minimize the code
overhead incurred by querying the LLM API. In other words, we want to make it **as simple as possible**
to make an LLM query from *anywhere* in the model code. Let's look at the code required to do a single
OpenAI query using the ``openai`` python wrapper:

.. code-block:: python

    import os
    from openai import OpenAI

    def my_function():

        ...

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            version=os.environ.get("OPENAI_VERSION"),
            endpoint=os.environ.get("OPENAI_ENDPOINT"),
        )

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say this is a test"}],
            model="gpt-4o",
        )

        if response is None:
            response_str = ""
        else:
            response_str = response.choices[0].message.content

        ...

    if __name__ == "__main__":
        my_function()


Not bad! However, it's still *A LOT* of boilerplate code every time you want to make query.
Moreover, you may want to do extra processing on the response every time a call is made (i.e.
convert it to JSON, track the number of tokens used, etc). One option is to refactor away
some of the logic into a separate function:


.. code-block:: python

    import os
    from openai import OpenAI

    def count_token_use(response):
        ...

    def parse_response_to_str(response):
        if response is None:
            return ""
        return response.choices[0].message.content

    def call_openai(messages, model="gpt-4o"):
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            version=os.environ.get("OPENAI_VERSION"),
            endpoint=os.environ.get("OPENAI_ENDPOINT"),
        )
        chat_completion = client.chat.completions.create(
            messages=messages, model=model
        )
        count_token_use(response)
        return parse_response_to_str(response)

    def my_function():
        ...
        response_str = call_openai(
            messages=[{"role": "user", "content": "Say this is a test"}],
            model="gpt-4o"
        )
        ...

    if __name__ == "__main__":
        my_function()


This is a lot closer to what we are looking for. However, all LLM deployments (that we know
of anyways!) have quotas and rate limits. It can be frustrating to run into an unexpected rate
limit error deep within our model logic, so we'd like to add a tracker for usage that
staggers the submission of our queries to stay within the pre-imposed rate limits.

To achieve this without complicating the code we have to invoke every time we wish to submit
an LLM query, we opt to submit our queries to a *queue* instead of to the API directly. Then,
a separate worker can simultaneously monitor the queue and track rolling token usage. If the
worker finds an item in the queue, it will submit the LLM call to the API as long as the rate
limit has not been reached. Otherwise, it will wait until the limit has been reset before
submitting an additional call.

This is the main concept behind *services* in the ELM ordinance code. We call the worker a
``Service``, and it monitors a dedicated queue that we can submit to from *anywhere* in our code
without having to worry about setting up usage monitors or other utility functions related to
the API call. To use the service, we simply have to invoke the ``call`` (class)method with the
relevant arguments. The only price we have to pay is that the service has to be *running* (i.e.
actively monitoring a queue and tracking usage) when our function is called. In practice, the
code looks something like this (with ``async`` flavor now spread throughout):

.. code-block:: python

    import asyncio
    import openai
    from elm.ords.services.provider import RunningAsyncServices
    from elm.ords.services.openai import OpenAIService

    async def my_function():
        # This function can be anywhere -
        # in a separate module or even in external code
        ...
        response_str = await OpenAIService.call(
            messages=[{"role": "user", "content": "Say this is a test"}],
            model="gpt-4o"
        )
        ...

    async def main():
        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        openai_service = OpenAIService(
            client, rate_limit=1e4  # adjustable; counted in tokens per minute
        )
        async with RunningAsyncServices([openai_service]):
            await my_function()

    if __name__ == "__main__":
        asyncio.run(main())


The cool thing is that if there are other functions in the model that use ``OpenAIService.call``,
this method will track their usage as well (all calls are submitted to the same queue), so no
need to worry about exceeding limits when calling other methods!
:class:`~elm.ords.services.openai.OpenAIService` also provides
some additional features behind the scenes, such as automatic resubmission upon API failure and
ability to set up total token usage tracking.

**4.1.1 Threaded and Process Pool Services**
--------------------------------------------

The ELM ordinance code takes the ``Services`` idea one step further. When running an ``async`` pipeline,
it can be beneficial to run some work on separate threads or even CPU cores. Since these are limited
resources, we can use ``Services`` to monitor their use as well! Let's look at a few examples:

.. code-block:: python

    import asyncio
    import openai
    from elm.ords.services.provider import RunningAsyncServices
    from elm.ords.services.openai import OpenAIService
    from elm.ords.services.threaded import FileMover
    from elm.ords.services.cpu import PDFLoader

    async def read_pdf():
        # Loads a PDF file in a separate process (this can be time consuming if using OCR, for example)
        return PDFLoader.call(...)

    async def my_function():
        ...
        response_str = await OpenAIService.call(
            messages=[{"role": "user", "content": "Say this is a test"}],
            model="gpt-4o"
        )
        ...
        FileMover.call(...) # Moves files to "./my_folder" using separate thread
        ...

    async def main():
        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        services = [
            OpenAIService(client, rate_limit=1e4),  # OpenAI service, with rate monitoring as before
            FileMover(out_dir="./my_folder", max_workers=8),  # launches 8 threads, each of which can be used run individual jobs
            PDFLoader(max_workers=4),  # launches 4 processes, each of which can be used run individual jobs
        ]
        async with RunningAsyncServices(services):
            await read_pdf()
            await my_function()

    if __name__ == "__main__":
        asyncio.run(main())


There are several other services provided out of the box - see the
`documentation <https://nrel.github.io/elm/_autosummary/elm.ords.services.html>`_ for details
Alternatively, we provide two base classes that you can extend to get similar functionality:
`ThreadedService <https://nrel.github.io/elm/_autosummary/elm.ords.services.threaded.ThreadedService.html#elm.ords.services.threaded.ThreadedService>`_
for threaded tasks and
`ProcessPoolService <https://nrel.github.io/elm/_autosummary/elm.ords.services.cpu.ProcessPoolService.html#elm.ords.services.cpu.ProcessPoolService>`_
for multiprocessing tasks.

**4.2 Key Classes**
===================

**4.2.1** :class:`~elm.web.google_search.PlaywrightGoogleLinkSearch`
--------------------------------------------------------------------

.. literalinclude:: ../../../elm/web/google_search.py
    :dedent:
    :start-at: Purpose:
    :end-before: .. end desc

**Example Code:**

.. code-block:: python

    from elm.web.google_search import PlaywrightGoogleLinkSearch

    async def main():
        search_engine = PlaywrightGoogleLinkSearch()
        return await search_engine.results(
            "Wind energy zoning ordinance Decatur County, Indiana",
            num_results=10,
        )

    if __name__ == "__main__":
        asyncio.run(main())


--------------------------------------------------------------------------------------------------------------------------------------------------

**4.2.2** :class:`~elm.web.file_loader.AsyncFileLoader`
-------------------------------------------------------

.. literalinclude:: ../../../elm/web/file_loader.py
    :dedent:
    :start-at: Purpose:
    :end-before: .. end desc

**Example Code:**

.. code-block:: python

    import asyncio
    from elm.web.file_loader import AsyncFileLoader

    async def main():
        loader = AsyncFileLoader()
        doc = await loader.fetch(
            url="https://en.wikipedia.org/wiki/National_Renewable_Energy_Laboratory"
        )
        return doc

    if __name__ == "__main__":
        asyncio.run(main())


--------------------------------------------------------------------------------------------------------------------------------------------------

**4.2.3** :class:`~elm.web.document.PDFDocument` / :class:`~elm.web.document.HTMLDocument`
------------------------------------------------------------------------------------------

.. literalinclude:: ../../../elm/web/document.py
    :dedent:
    :start-at: Purpose:
    :end-before: .. end desc

**Example Code:**

.. code-block:: python

    from elm.web.document import HTMLDocument

    content = ...
    doc = HTMLDocument([content])
    doc.text, doc.raw_pages, doc.metadata

--------------------------------------------------------------------------------------------------------------------------------------------------

**4.2.4** :class:`~elm.ords.services.openai.OpenAIService`
----------------------------------------------------------

.. literalinclude:: ../../../elm/ords/services/openai.py
    :dedent:
    :start-at: Purpose:
    :end-before: .. end desc

**Example Code:**

.. code-block:: python

    import asyncio
    import openai
    from elm.ords.services.provider import RunningAsyncServices
    from elm.ords.services.openai import OpenAIService

    async def main():
        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        service = OpenAIService(client, rate_limit=1e4)
        async with RunningAsyncServices([service]):
            response_str = await OpenAIService.call(
                messages=[{"role": "user", "content": "Say this is a test"}],
                model="gpt-4o"
            )
        return response_str

    if __name__ == "__main__":
        asyncio.run(main())


--------------------------------------------------------------------------------------------------------------------------------------------------

**4.2.5** :class:`~elm.ords.llm.calling.LLMCaller` / :class:`~elm.ords.llm.calling.ChatLLMCaller` / :class:`~elm.ords.llm.calling.StructuredLLMCaller`
------------------------------------------------------------------------------------------------------------------------------------------------------

.. literalinclude:: ../../../elm/ords/llm/calling.py
    :dedent:
    :start-at: Purpose:
    :end-before: .. end desc

**Example Code:**

.. code-block:: python

    import asyncio
    import openai
    from elm.ords.services.provider import RunningAsyncServices
    from elm.ords.services.openai import OpenAIService
    from elm.ords.llm import StructuredLLMCaller

    CALLER = StructuredLLMCaller(
        llm_service=OpenAIService,
        model="gpt-4o",
        temperature=0,
        seed=42,
        timeout=30,
    )

    async def main():
        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        service = OpenAIService(client, rate_limit=1e4)

        async with RunningAsyncServices([service]):
            response_str = await CALLER.call(
                sys_msg="You are a helpful assistant",
                content="Say this is a test",
            )
        return response_str

    if __name__ == "__main__":
        asyncio.run(main())

--------------------------------------------------------------------------------------------------------------------------------------------------

**4.2.6** :class:`~elm.ords.validation.location.CountyValidator`
----------------------------------------------------------------

.. literalinclude:: ../../../elm/ords/validation/location.py
    :dedent:
    :start-at: Purpose:
    :end-before: .. end desc

**Example Code:**

.. code-block:: python

    import asyncio
    import openai
    from elm.ords.services.provider import RunningAsyncServices
    from elm.ords.services.openai import OpenAIService
    from elm.ords.llm import StructuredLLMCaller
    from elm.ords.validation.location import CountyValidator
    from elm.web.document import HTMLDocument

    CALLER = StructuredLLMCaller(
        llm_service=OpenAIService,
        model="gpt-4o",
        temperature=0,
        seed=42,
        timeout=30,
    )

    async def main():
        content = ...
        doc = HTMLDocument([content])

        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        service = OpenAIService(client, rate_limit=1e4)
        validator =  CountyValidator(CALLER)

        async with RunningAsyncServices([service]):
            is_valid = await validator.check(
                doc, county="Decatur", state="Indiana"
            )

        return is_valid

    if __name__ == "__main__":
        asyncio.run(main())

--------------------------------------------------------------------------------------------------------------------------------------------------


**4.2.7** :class:`~elm.ords.extraction.ordinance.OrdinanceValidator`
--------------------------------------------------------------------

.. literalinclude:: ../../../elm/ords/extraction/ordinance.py
    :dedent:
    :start-after: .. start desc ov
    :end-before: .. end desc

**Example Code:**

.. code-block:: python

    import asyncio
    import openai
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from elm.ords.extraction.ordinance import OrdinanceValidator
    from elm.ords.services.provider import RunningAsyncServices
    from elm.ords.services.openai import OpenAIService
    from elm.ords.llm import StructuredLLMCaller
    from elm.web.document import HTMLDocument

    CALLER = StructuredLLMCaller(
        llm_service=OpenAIService,
        model="gpt-4o",
        temperature=0,
        seed=42,
        timeout=30,
    )
    TEXT_SPLITTER = RecursiveCharacterTextSplitter(...)

    async def main():
        content = ...
        doc = HTMLDocument([content])

        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        service = OpenAIService(client, rate_limit=1e4)
        chunks = TEXT_SPLITTER.split_text(doc.text)
        validator = OrdinanceValidator(CALLER, chunks)

        async with RunningAsyncServices([service]):
            contains_ordinances = await validator.parse()
            text = validator.ordinance_text

        return contains_ordinances, text

    if __name__ == "__main__":
        asyncio.run(main())

--------------------------------------------------------------------------------------------------------------------------------------------------


**4.2.8** :class:`~elm.ords.extraction.ordinance.OrdinanceExtractor`
--------------------------------------------------------------------

.. literalinclude:: ../../../elm/ords/extraction/ordinance.py
    :dedent:
    :start-after: .. start desc oe
    :end-before: .. end desc

**Example Code:**

.. code-block:: python

    import asyncio
    import openai
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from elm.ords.extraction.ordinance import OrdinanceExtractor
    from elm.ords.services.provider import RunningAsyncServices
    from elm.ords.services.openai import OpenAIService
    from elm.ords.llm import StructuredLLMCaller

    CALLER = StructuredLLMCaller(
        llm_service=OpenAIService,
        model="gpt-4o",
        temperature=0,
        seed=42,
        timeout=30,
    )
    TEXT_SPLITTER = RecursiveCharacterTextSplitter(...)

    async def main():
        content = ...

        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        service = OpenAIService(client, rate_limit=1e4)
        validator = OrdinanceExtractor(CALLER)

        async with RunningAsyncServices([service]):
            text_chunks = TEXT_SPLITTER.split_text(content)
            ordinance_text = await extractor.check_for_restrictions(text_chunks)

            text_chunks = text_splitter.split_text(ordinance_text)
            ordinance_text = await TEXT_SPLITTER extractor.check_for_correct_size(text_chunks)

        return ordinance_text

    if __name__ == "__main__":
        asyncio.run(main())


--------------------------------------------------------------------------------------------------------------------------------------------------


**4.2.9** :class:`~elm.ords.extraction.tree.AsyncDecisionTree`
--------------------------------------------------------------

.. literalinclude:: ../../../elm/ords/extraction/tree.py
    :dedent:
    :start-at: Purpose:
    :end-before: .. end desc

**Example Code:**

.. code-block:: python

    import asyncio
    import openai
    from elm.ords.services.provider import RunningAsyncServices
    from elm.ords.services.openai import OpenAIService
    from elm.ords.extraction.tree import AsyncDecisionTree


    async def main():
        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        service = OpenAIService(client, rate_limit=1e4)

        G = ... # graph with prompts and a `ChatLLMCaller` instance embedded
        tree = AsyncDecisionTree(G)

        async with RunningAsyncServices([service]):
            response = await tree.async_run()

        return response

    if __name__ == "__main__":
        asyncio.run(main())


--------------------------------------------------------------------------------------------------------------------------------------------------


**4.2.10** :class:`~elm.ords.extraction.parse.StructuredOrdinanceParser`
------------------------------------------------------------------------

.. literalinclude:: ../../../elm/ords/extraction/parse.py
    :dedent:
    :start-at: Purpose:
    :end-before: .. end desc

**Example Code:**

.. code-block:: python

    import asyncio
    import openai
    from elm.ords.extraction.parse import StructuredOrdinanceParser
    from elm.ords.services.provider import RunningAsyncServices
    from elm.ords.services.openai import OpenAIService
    from elm.ords.llm import StructuredLLMCaller

    CALLER = StructuredLLMCaller(
        llm_service=OpenAIService,
        model="gpt-4o",
        temperature=0,
        seed=42,
        timeout=30,
    )

    async def main():
        content = ...

        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        service = OpenAIService(client, rate_limit=1e4)
        parser = StructuredOrdinanceParser(CALLER)

        async with RunningAsyncServices([service]):
            ordinance_values = await parser.parse(content)

        return ordinance_values

    if __name__ == "__main__":
        asyncio.run(main())


--------------------------------------------------------------------------------------------------------------------------------------------------

****************
**5. Workflows**
****************

**5.1 Downloading documents from Google**
=========================================
We give a rough breakdown of the following call:

.. code-block:: python

    import asyncio
    from elm.web.google_search import google_results_as_docs

    QUERIES = [
        "NREL wiki",
        "National Renewable Energy Laboratory director",
        "NREL leadership wikipedia",
    ]

    async def main():
        docs = await google_results_as_docs(QUERIES, num_urls=4)
        return docs

    if __name__ == "__main__":
        asyncio.run(main())


**Step-by-Step:**

1. :func:`~elm.web.google_search.google_results_as_docs()` is invoked with 3 queries and ``num_urls=4``.
2. Each of the three queries are processed asynchronously, creating a :class:`~elm.web.google_search.PlaywrightGoogleLinkSearch` instance and retrieving the top URL results.
3. Internal code reduces the URL lists returned from each of the queries into the top 4 URLs.
4. :class:`~elm.web.file_loader.AsyncFileLoader` asynchronously downloads the content for reach of the top 4 URLs, determines the document type the content should be stored
   in (:class:`~elm.web.document.HTMLDocument` or :class:`~elm.web.document.PDFDocument`), creates and populates the document instances, and returns the document to the caller.

**Sequence Diagram:**

.. mermaid::

    sequenceDiagram
        participant A as google_results_as_docs()
        participant B as PlaywrightGoogleLinkSearch
        participant D as AsyncFileLoader
        participant E as HTMLDocument
        participant F as PDFDocument

        A ->> B: Query 1
        activate B
        A ->> B: Query 2
        B ->> A: Top-URL List 1
        A ->> B: Query 3
        B ->> A: Top-URL List 2
        B ->> A: Top-URL List 3
        deactivate B

        A ->> A: URL lists reduced to top 4 URLs

        A ->> D: URL 1
        activate D
        A ->> D: URL 2
        D ->> E: Content 1
        activate E
        A ->> D: URL 3
        E ->> A: Document 1
        deactivate E
        D ->> F: Content 2
        activate F
        D ->> E: Content 3
        activate E
        F ->> A: Document 2
        deactivate F
        E ->> A: Document 3
        deactivate E
        A ->> D: URL 4
        D ->> F: Content 4
        activate F
        F ->> A: Document 4
        deactivate F
        deactivate D


Note that the interleaved call-and-response pairs are meant to exhibit the `async` nature of the process and do not reflect a deterministic execution order.

--------------------------------------------------------------------------------------------------------------------------------------------------


**5.2 Querying OpenAI**
=======================
We give a rough breakdown of the following call:

.. code-block:: python

    import asyncio
    import openai
    from elm.ords.services.provider import RunningAsyncServices
    from elm.ords.services.openai import OpenAIService
    from elm.ords.llm import LLMCaller

    async def main():
        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        service = OpenAIService(client, rate_limit=1e4)
        llm_caller = LLMCaller(
            llm_service=OpenAIService,
            model="gpt-4o",
            temperature=0,
            seed=42,
            timeout=30,
        )
        async with RunningAsyncServices([service]):
            tasks = [
                asyncio.create_task(
                    llm_caller.call(
                        sys_msg="You are a helpful assistant",
                        content=f"Say this is a test: {i}"
                    )
                )
                for i in range(1, 4)
            ]
            responses = await asyncio.gather(*tasks)
        return responses

    if __name__ == "__main__":
        asyncio.run(main())

**Step-by-Step:**

1. ``main()`` initializes `openai.AsyncAzureOpenAI` client, :class:`~elm.ords.services.openai.OpenAIService` (not running), and :class:`~elm.ords.llm.calling.LLMCaller`.
2. ``main()`` enters the :class:`~elm.ords.services.provider.RunningAsyncServices` context, which starts the ``service``.
3. The now running :class:`~elm.ords.services.openai.OpenAIService` initializes it's own queue and begins monitoring it.
4. ``main()`` now submits three LLM queries using the :class:`~elm.ords.llm.calling.LLMCaller` instance.
5. :class:`~elm.ords.llm.calling.LLMCaller` puts the queries onto the queue initialized by :class:`~elm.ords.services.openai.OpenAIService`.
6. :class:`~elm.ords.services.openai.OpenAIService` detects that the queue is not empty, checks that the rate limit has not been exceeded, and submits the first query to the LLM.
7. :class:`~elm.ords.services.openai.OpenAIService` detects that the queue is still not empty. It checks the updated rate limit. Seeing that it has not been exceeded, it submits the second query to the LLM.
8. The LLM send back the first response to ``main()``.
9. Once again, :class:`~elm.ords.services.openai.OpenAIService` detects that the queue is not empty. It checks the updated rate limit, but it has now been exceeded. It does *not* submit the third query,
   and instead continues to monitor the running rate limit.
10. :class:`~elm.ords.services.openai.OpenAIService` detects that it has waited long enough, and that the rate limit has still not been exceeded. Since there is still a query in the queue, it submits the third query.
11. The LLM send back the responses of query 2 and 3 to ``main()``.
12. :class:`~elm.ords.services.openai.OpenAIService` continues to monitor the queue, but ``main()`` has not submitted any more queries.
13. Having received all the responses, ``main()`` exists the context. :class:`~elm.ords.services.openai.OpenAIService` tears down the empty queue and stops running.

**Sequence Diagram:**

.. mermaid::

    sequenceDiagram

        participant B as LLMCaller
        participant A as main
        participant C as RunningAsyncServices
        participant D as OpenAIService
        participant E as OpenAIServiceQueue
        participant F as LLM (Chat GPT)

        A ->> D: Initialize
        activate D
        A ->> B: Initialize
        activate B
        A ->> C: Enter Context
        activate C
        C ->> D: Start Running
        D ->> E: Initialize
        activate E
        D ->> D: Check rate limit (OK)
        D ->> E: Check queue
        A ->> B: Submit queries
        B ->> E: Enqueue "Say this is a test: 1"
        D ->> D: Check rate limit (OK)
        D ->> E: Check queue
        E ->> D: Get LLM call request
        D ->> F: Submit call for "Say this is a test: 1"
        B ->> E: Enqueue "Say this is a test: 2"
        B ->> E: Enqueue "Say this is a test: 3"
        D ->> D: Check rate limit (OK)
        D ->> E: Check queue
        E ->> D: Get LLM call request
        D ->> F: Submit call for "Say this is a test: 2"
        F ->> A: Response: "This is a test: 1"
        D ->> D: Check rate limit (Failed)
        loop while rate limit exceeded
            D-->D: Check rate limit
        end
        D ->> D: Check rate limit (OK)
        D ->> E: Check queue
        E ->> D: Get LLM call request
        D ->> F: Submit call for "Say this is a test: 3"
        F ->> A: Response: "This is a test: 2"
        F ->> A: Response: "This is a test: 3"
        D ->> D: Check rate limit (OK)
        D ->> E: Check queue

        A ->> C: Exit context
        C ->> D: Teardown
        deactivate C
        D ->> E: Teardown
        deactivate D
        deactivate E

        A ->> B: Teardown
        deactivate B


Note that the interleaved call-and-response pairs are meant to exhibit the `async` nature of the process and do not reflect a deterministic execution order.

--------------------------------------------------------------------------------------------------------------------------------------------------

***************
**6. Appendix**
***************

**6.1 Tools and Libraries**
===========================

- **aiohttp/beautifulsoup4:** For fetching content from the web.
- **html2text:** For utilities to pull text from HTML.
- **langchain:** For utility classes like ``RecursiveCharacterTextSplitter``.
- **networkx:** For representing the DAG behind the decision tree(s).
- **pdftotext:** For robust PDF to text conversion using poppler.
- **playwright:** For navigating the web and performing Google searches.
- **PyPDF2:** For auxiliary PDF utilities.
- **pytesseract (optional):** For OCR utilities to read scanned PDF files.
- **tiktoken:** For counting the number of LLM tokens used by a query.

--------------------------------------------------------------------------------------------------------------------------------------------------

*******************
**7. Deliverables**
*******************

This document can serve as a foundational guide for developers and analysis.
It helps users of this code understand the software's design and allows for
smoother onboarding and more informed project planning.
