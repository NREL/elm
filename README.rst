***************************
Energy Language Model (ELM)
***************************

The Energy Language Model (ELM) software provides interfaces to apply Large Language Models (LLMs) like ChatGPT and GPT-4 to energy research. For example, you might be interested in:

- `Converting PDFs into a text database <https://nrel.github.io/elm/_autosummary/elm.pdf.PDFtoTXT.html#elm.pdf.PDFtoTXT>`_
- `Chunking text documents and embedding into a vector database <https://nrel.github.io/elm/_autosummary/elm.embed.ChunkAndEmbed.html#elm.embed.ChunkAndEmbed>`_
- `Performing recursive document summarization <https://nrel.github.io/elm/_autosummary/elm.summary.Summary.html#elm.summary.Summary>`_
- `Building an automated data extraction workflow using decision trees <https://nrel.github.io/elm/_autosummary/elm.tree.DecisionTree.html#elm.tree.DecisionTree>`_
- `Building a chatbot to interface with your own text corpus <https://nrel.github.io/elm/_autosummary/elm.wizard.EnergyWizard.html#elm.wizard.EnergyWizard>`_

Installing ELM
==============

.. inclusion-install

#. from home dir, ``git clone git@github.com:NREL/elm.git``
#. Create ``elm`` environment and install package
    1) Create a conda env: ``conda create -n elm``
    2) Run the command: ``conda activate elm``
    3) ``cd`` into the repo cloned in 1.
    4) Prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    5) Install ``elm`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)

.. inclusion-acknowledgements

Acknowledgments
===============

This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the DOE Wind Energy Technologies Office (WETO), the DOE Solar Energy Technologies Office (SETO), and internal research funds at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
