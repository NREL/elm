# -*- coding: utf-8 -*-
"""
Test
"""
import os
import pandas as pd
import numpy as np
from elm import TEST_DATA_DIR
from elm.embed import ChunkAndEmbed
from elm.wizard import EnergyWizard
import elm.embed

FP_PDF = os.path.join(TEST_DATA_DIR, 'GPT-4.pdf')
FP_TXT = os.path.join(TEST_DATA_DIR, 'gpt4.txt')

with open(FP_TXT, 'r') as f:
    TEXT = f.read()


class MockClass:
    """Dummy class to mock various api calls"""

    @staticmethod
    def get_embedding(*args, **kwargs):
        """Mock for ChunkAndEmbed.call_api()"""
        embedding = np.random.uniform(0, 1, 10)
        return embedding

    @staticmethod
    def call(*args, **kwargs):
        """Mock for ChunkAndEmbed.call_api()"""
        embedding = np.random.uniform(0, 1, 10)
        response = {'data': [{'embedding': embedding}]}
        return response

    @staticmethod
    def create(*args, **kwargs):
        """Mock for openai.ChatCompletion.create()"""
        response = {'choices': [{'message': {'content': 'hello!'}}]}
        return response


def test_chunk_and_embed(mocker):
    """Simple text to embedding test

    Note that embedding api is mocked here and not actually tested.
    """

    mocker.patch.object(elm.embed.ChunkAndEmbed, "call_api", MockClass.call)
    mocker.patch.object(elm.wizard.EnergyWizard, "get_embedding",
                        MockClass.get_embedding)
    mocker.patch.object(elm.wizard.openai.ChatCompletion, "create",
                        MockClass.create)

    ce0 = ChunkAndEmbed(TEXT, tokens_per_chunk=400)
    embeddings = ce0.run()
    corpus = []
    for i, emb in enumerate(embeddings):
        corpus.append({'text': ce0.text_chunks[i], 'embedding': emb,
                       'ref': 'source0'})

    wizard = EnergyWizard(pd.DataFrame(corpus), token_budget=1000,
                          ref_col='ref')
    question = 'What time is it?'
    out = wizard.ask(question, debug=True, stream=False, print_references=True)
    msg, query, ref = out

    assert msg == 'hello!'
    assert query.startswith(EnergyWizard.MODEL_INSTRUCTION)
    assert query.endswith(question)
    assert 'source0' in ref
