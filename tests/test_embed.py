# -*- coding: utf-8 -*-
"""
Test
"""
import os
import numpy as np
from elm import TEST_DATA_DIR
from elm.embed import ChunkAndEmbed
import elm.embed

os.environ["OPENAI_API_KEY"] = "dummy"

FP_PDF = os.path.join(TEST_DATA_DIR, 'GPT-4.pdf')
FP_TXT = os.path.join(TEST_DATA_DIR, 'gpt4.txt')

with open(FP_TXT, 'r', encoding='utf8') as f:
    TEXT = f.read()


class MockClass:
    """Dummy class to mock ChunkAndEmbed.call_api()"""

    @staticmethod
    def call(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for ChunkAndEmbed.call_api()"""
        embedding = np.random.uniform(0, 1, 10)
        response = {'data': [{'embedding': embedding}]}
        return response


def test_chunk_and_embed(mocker):
    """Simple text to embedding test

    Note that embedding api is mocked here and not actually tested.
    """
    mocker.patch.object(elm.embed.ChunkAndEmbed, "call_api", MockClass.call)
    ce0 = ChunkAndEmbed(TEXT, tokens_per_chunk=100)
    embeddings = ce0.run()
    assert len(embeddings) == len(ce0.text_chunks)

    ce1 = ChunkAndEmbed(TEXT, tokens_per_chunk=400)
    embeddings = ce1.run()
    assert len(ce1.text_chunks) < len(ce0.text_chunks)
    assert len(embeddings) == len(ce1.text_chunks)

    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert len(emb) == 10
