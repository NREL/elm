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

os.environ["OPENAI_API_KEY"] = "dummy"

FP_PDF = os.path.join(TEST_DATA_DIR, 'GPT-4.pdf')
FP_TXT = os.path.join(TEST_DATA_DIR, 'gpt4.txt')

with open(FP_TXT, 'r', encoding='utf8') as f:
    TEXT = f.read()


class MockObject:
    """Dummy class for mocking api response objects"""


class MockClass:
    """Dummy class to mock various api calls"""

    @staticmethod
    def get_embedding(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for ChunkAndEmbed.call_api()"""
        embedding = np.random.uniform(0, 1, 10)
        return embedding

    @staticmethod
    def call(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for ChunkAndEmbed.call_api()"""
        embedding = np.random.uniform(0, 1, 10)
        response = {'data': [{'embedding': embedding}]}
        return response

    @staticmethod
    def create(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for openai.ChatCompletion.create()"""
        # pylint: disable=attribute-defined-outside-init
        response = MockObject()
        response.choices = [MockObject()]
        response.choices[0].message = MockObject()
        response.choices[0].message.content = 'hello!'
        return response


def make_corpus(mocker):
    """Make a text corpus with embeddings for the wizard."""
    mocker.patch.object(elm.embed.ChunkAndEmbed, "call_api", MockClass.call)
    mocker.patch.object(elm.wizard.EnergyWizard, "get_embedding",
                        MockClass.get_embedding)

    ce0 = ChunkAndEmbed(TEXT, tokens_per_chunk=400)
    embeddings = ce0.run()
    corpus = []
    for i, emb in enumerate(embeddings):
        corpus.append({'text': ce0.text_chunks[i], 'embedding': emb,
                       'ref': 'source0'})
    return corpus


def test_chunk_and_embed(mocker):
    """Simple text to embedding test

    Note that embedding api is mocked here and not actually tested.
    """
    corpus = make_corpus(mocker)
    wizard = EnergyWizard(pd.DataFrame(corpus), token_budget=1000,
                          ref_col='ref')

    mocker.patch.object(wizard._client.chat.completions, "create",
                        MockClass.create)

    question = 'What time is it?'
    out = wizard.chat(question, stream=False, print_references=True)
    response_message, query, references, performance = out
    assert response_message.startswith('hello!')
    assert query.startswith(EnergyWizard.MODEL_INSTRUCTION)
    assert query.endswith(question)
    assert 'source0' in references
    assert isinstance(performance, dict)


def test_convo_query(mocker):
    """Query with multiple messages

    Note that embedding api is mocked here and not actually tested.
    """

    corpus = make_corpus(mocker)
    wizard = EnergyWizard(pd.DataFrame(corpus), token_budget=1000,
                          ref_col='ref')

    mocker.patch.object(wizard._client.chat.completions, "create",
                        MockClass.create)

    question1 = 'What time is it?'
    question2 = 'How about now?'

    query = wizard.chat(question1, stream=False, convo=True,
                        print_references=True)[1]
    assert question1 in query
    assert question2 not in query
    assert len(wizard.messages) == 3

    query = wizard.chat(question2, stream=False, convo=True,
                        print_references=True)[1]
    assert question1 in query
    assert question2 in query
    assert len(wizard.messages) == 5

    wizard.clear()
    assert len(wizard.messages) == 1

    query = wizard.chat(question2, stream=False, convo=True,
                        print_references=True)[1]
    assert question1 not in query
    assert question2 in query
    assert len(wizard.messages) == 3
