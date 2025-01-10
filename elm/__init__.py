# -*- coding: utf-8 -*-
"""
Energy Language Model
"""

import os
import logging
from functools import partial, partialmethod
from elm.base import ApiBase
from elm.chunk import Chunker
from elm.embed import ChunkAndEmbed
from elm.pdf import PDFtoTXT
from elm.summary import Summary
from elm.tree import DecisionTree
from elm.wizard import EnergyWizard
from elm.web.osti import OstiRecord, OstiList

__author__ = """Grant Buster"""
__email__ = "Grant.Buster@nrel.gov"

ELM_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_DIR = os.path.join(os.path.dirname(ELM_DIR), 'tests', 'data')

logging.TRACE = 5
logging.addLevelName(logging.TRACE, 'TRACE')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
logging.trace = partial(logging.log, logging.TRACE)
