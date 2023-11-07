# -*- coding: utf-8 -*-
"""
Test
"""
import os
import numpy as np
from elm import TEST_DATA_DIR
from elm.chunk import Chunker


os.environ["OPENAI_API_KEY"] = "dummy"

FP_TXT = os.path.join(TEST_DATA_DIR, 'gpt4.txt')
with open(FP_TXT, 'r', encoding='utf8') as f:
    TEXT = f.read()


def test_overlap():
    """Test overlap of chunk paragraphs"""
    chunks0 = Chunker(TEXT, tokens_per_chunk=1, overlap=0)
    assert len(chunks0.text.split('\n\n')) == len(chunks0)

    chunks1 = Chunker(TEXT, tokens_per_chunk=1, overlap=1)
    assert len(chunks1.text.split('\n\n')) == len(chunks1)

    for i, c0 in enumerate(chunks0):
        assert len(c0) < len(chunks1[i])

    for i0 in [5, 10, 25]:
        c1 = chunks1[i0].split('\n\n')
        assert c1[0] == chunks0[i0 - 1]
        assert c1[1] == chunks0[i0]
        assert c1[2] == chunks0[i0 + 1]


def test_by_tokens():
    """Test chunking of text by token count"""
    chunks0 = Chunker(TEXT, tokens_per_chunk=100, overlap=0)
    chunks1 = Chunker(TEXT, tokens_per_chunk=200, overlap=0)
    chunks2 = Chunker(TEXT, tokens_per_chunk=400, overlap=0)

    assert len(chunks1) < len(chunks0)
    assert len(chunks2) < len(chunks0)

    assert np.max(chunks0.chunk_tokens) < 200
    assert np.max(chunks1.chunk_tokens) < 200
    assert np.max(chunks2.chunk_tokens) < 400

    assert sum(chunks0.chunk_tokens) == sum(chunks1.chunk_tokens)
    assert sum(chunks0.chunk_tokens) == sum(chunks2.chunk_tokens)

    assert len('\n\n'.join(chunks0.chunks)) == len('\n\n'.join(chunks1.chunks))
    assert len('\n\n'.join(chunks0.chunks)) == len('\n\n'.join(chunks2.chunks))
