# Welcome to Energy Language Model - OrdinanceGPT

The ordinance web scraping and data extraction portion of this codebase required a few extra dependencies that do not come out-of-the-box with the base ELM software.
To set up ELM for ordinances, first create a conda environment. Then, _before installing ELM_, run the poppler installation:

    $ conda install -c conda-forge poppler

Then, install `pdftotext`:

    $ pip install pdftotext

(OPTIONAL) If you want to have access to Optical Character Recognition (OCR) for PDF parsing, you should also install pytesseract during this step:

    $ pip install pytesseract pdf2image

At this point, you can install ELM per the [front-page README](https://github.com/NREL/elm/blob/main/README.rst) instructions.

After ELM installs successfully, you must instantiate the playwright module, which is used for web scraping.
To do so, simply run:

    $ playwright install

Now you are ready to run ordinance retrieval and extraction. See the [example](https://github.com/NREL/elm/blob/main/examples/ordinance_gpt/README.rst) to get started.
