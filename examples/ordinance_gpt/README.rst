*************
Ordinance GPT
*************

This example folder contains supporting documents, results, and code for the Ordinance GPT experiment.

Prerequisites
=============
We recommend installing the pytesseract module to allow PDF retrieval for scanned documents.
See the `ordinance-specific installation instructions <https://github.com/NREL/elm/blob/main/elm/ords/README.md>`_
for more details.

Running from Python
===================
This instruction set presents a simplified example to extract ordinance data from a ordinance document on disk. This corresponds with the ordinance data extraction from PDF results in `Buster et al., 2024 <https://doi.org/10.48550/arXiv.2403.12924>`_. 

To run this, first download one or more ordinance documents from `the Box folder <https://app.box.com/s/a8oi8jotb9vnu55rzdul7e291jnn7hmq>`_.

After downloading the ordinance document(s), set the relevant path for the ``fp_pdf`` variable, and then run the script:
.. code-block:: bash

    $ python parse_pdf.py

Running from the Command Line Utility
=====================================
This instruction set is an experimental process to use LLMs to search the internet for relevant ordinance documents, download those documents, and then extract the relevant ordinance data. 

There are a few key things you need to set up in order to run ordinance retrieval and extraction.
First, you must specify which counties you want to process. You can do this by setting up a CSV file
with a ``County`` and a ``State`` column. Each row in the CSV file then represents a single county to process.
See the `example CSV <https://github.com/NREL/elm/blob/main/examples/ordinance_gpt/counties.csv>`_
file for reference.

Once you have set up the county CSV, you can fill out the
`template JSON config <https://github.com/NREL/elm/blob/main/examples/ordinance_gpt/config.json>`_.
See the documentation for the `"process_counties_with_openai" function <https://github.com/NREL/elm/blob/main/elm/ords/process.py#L78>`_
for an explanation of all the allowed inputs to the configuration file.
Some notable inputs here are the ``azure*`` keys, which should be configured to match your Azure OpenAI API
deployment (unless it's defined in your environment with the ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_VERSION``,
and ``AZURE_OPENAI_ENDPOINT`` keys, in which case you can remove these keys completely),
and the ``pytesseract_exe_fp`` key, which should point to the pytesseract executable path on your
local machine (or removed from the config file if you are opting out of OCR). You may also have to adjust
the ``llm_service_rate_limit`` to match your deployment's API tokens-per-minute limit. Be sure to provide full
paths to all files/directories unless you are executing the program from your working folder.

Execution
---------
Once you are happy with the configuration parameters, you can kick off the processing using

.. code-block:: bash

    $ elm ords -c config.json

You may also wish to add a ``-v`` option to print logs to the terminal (however, keep in mind that the code runs
asynchronously, so the the logs will not print in order).

.. WARNING:: Running all of the 85 counties given in the sample county CSV file can cost $700-$1000 in API calls. We recommend running a smaller subset for example purposes.

Debugging
---------
Not sure why things aren't working? No error messages? Make sure you run the CLI call with a ``-v`` flag for "verbose" logging (e.g., ``$ elm ords -c config.json -v``)

Errors on import statements? Trouble importing ``pdftotext`` with cryptic error messages like ``symbol not found in flat namespace``? Follow the `ordinance-specific install instructions <https://github.com/NREL/elm/blob/main/elm/ords/README.md>`_ *exactly*.

Source Ordinance Documents
==========================

The ordinance documents downloaded using (an older version of) this example code can be downloaded `here
<https://app.box.com/s/a8oi8jotb9vnu55rzdul7e291jnn7hmq>`_.

Extension to Other Technologies
===============================
Extending this functionality to other technologies is possible but requires deeper understanding of the underlying processes.
We recommend you start out by examining the decision tree queries in `graphs.py <https://github.com/NREL/elm/blob/main/elm/ords/extraction/graphs.py>`_
as well as how they are applied in `parse.py <https://github.com/NREL/elm/blob/main/elm/ords/extraction/parse.py>`_. Once you
have a firm understanding of these two modules, look through the
`document validation routines <https://github.com/NREL/elm/blob/main/elm/ords/validation>`_ to get a better sense of how to
adjust the web-scraping portion of the code to your technology. When you have set up the validation and parsing for your
technology, put it all together by adjusting the `"process_counties_with_openai" function <https://github.com/NREL/elm/blob/main/elm/ords/process.py#L78>`_
to call your new routines.
