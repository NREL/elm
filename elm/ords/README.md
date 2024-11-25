# Welcome to Energy Language Model - OrdinanceGPT

The ordinance web scraping and data extraction portion of this codebase
required a few extra dependencies that do not come out-of-the-box with the base
ELM software. To set up ELM for ordinances, first create a conda environment.
We have had some issues using python 3.9 and recommend using python 3.11. Then,
_before installing ELM_, run the poppler installation:

    $ conda install -c conda-forge poppler

Then, install `pdftotext`:

    $ pip install pdftotext

(OPTIONAL) If you want to have access to Optical Character Recognition (OCR) for PDF parsing, you should also install pytesseract during this step. Note that there may be additional OS-specific installation steps to get tesseract working properly (see the [pytesseract install instructions](https://pypi.org/project/pytesseract/))

    $ pip install pytesseract pdf2image

At this point, you can install ELM per the [front-page README](https://github.com/NREL/elm/blob/main/README.rst) instructions, e.g.:

    $ pip install -e .

After ELM installs successfully, you must instantiate the playwright module, which is used for web scraping.
To do so, simply run:

    $ playwright install

Now you are ready to run ordinance retrieval and extraction. See the [example](https://github.com/NREL/elm/blob/main/examples/ordinance_gpt/README.rst) to get started. If you get additional import errors, just install additional packages as necessary, e.g.:

    $ pip install beautifulsoup4 html5lib


## Architecture

For information on the architectural design of this code, see the [design document](https://nrel.github.io/elm/dev/ords_architecture.html).
