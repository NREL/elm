from urllib.request import urlopen
from elm.web.rhub import ResearchOutputs
import elm.web.rhub

fp = '../data/rhub_html.txt'

with open(fp, 'r', encoding='utf8') as f:
    TEXT = f.read()

class MockClass:
    """Dummy class to mock ResearchOutputs.html_response()"""

    @staticmethod
    def call(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for ResearchOutputs.html_response()"""
        return TEXT 

def test_rhub(mocker):
    """Test to ensure correct response from research hub."""
    mocker.patch.object(elm.web.rhub.ResearchOutputs, 'html_response', MockClass.call)

    out = ResearchOutputs("dummy")

    meta = out.build_meta()

    assert len(meta) > 10
    assert 'title' in meta.columns
    assert 'url' in meta.columns
    assert meta['title'].isna().sum() == 0
    assert meta['url'].isna().sum() == 0

    
