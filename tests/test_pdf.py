# -*- coding: utf-8 -*-
"""
Test
"""
import os
from elm import TEST_DATA_DIR
from elm.pdf import PDFtoTXT
import elm.pdf

os.environ["OPENAI_API_KEY"] = "dummy"

FP_PDF = os.path.join(TEST_DATA_DIR, 'GPT-4.pdf')
FP_TXT = os.path.join(TEST_DATA_DIR, 'gpt4.txt')

with open(FP_TXT, 'r', encoding='utf8') as f:
    TEXT = f.read()


class MockClass:
    """Dummy class to mock requests.post"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def json(self):
        """Get json response"""
        content = self.kwargs['json']['messages'][1]['content']
        content = content[content.index('"""') + 3:]
        content = content[:content.index('"""')]
        response = {'choices': [{'message': {'content': content}}]}
        return response

    @classmethod
    def call(cls, **kwargs):
        """Mock for requests.post"""
        return cls(**kwargs)


def test_pdf_txt_clean(mocker):
    """Simple text on PDF text cleaning functionality.

    Note that LLM-based text cleaning is mocked here and not actually tested.
    """
    mocker.patch.object(elm.pdf.requests, "post", MockClass.call)
    pdf = PDFtoTXT(FP_PDF)
    pdf.clean_txt()

    missing = len(set(TEXT.split(' ')) - set(pdf.full.split(' ')))
    ntotal = len(TEXT.split(' '))

    assert (missing / ntotal) < 0.1
