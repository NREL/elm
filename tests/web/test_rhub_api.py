"""Test research hub api response"""
import os
import json
from elm import TEST_DATA_DIR
from elm.web.rhub import ProfilesList
from elm.web.rhub import PublicationsList
import elm.web.rhub

os.environ["RHUB_API_KEY"] = "dummy"

PROFILES_JSON = os.path.join(TEST_DATA_DIR, 'rhub_api_profiles.json')
PUBLICATIONS_JSON = os.path.join(TEST_DATA_DIR, 'rhub_api_publications.json')

with open(PROFILES_JSON, 'r') as json_file:
    PROFILES_RECORDS = json.load(json_file)

with open(PUBLICATIONS_JSON, 'r') as json_file:
    PUBLICATIONS_RECORDS = json.load(json_file)


class MockClass:
    """Dummy class to mock rhub API responses"""

    @staticmethod
    def profiles_call(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for ProfilesList._get_first()"""
        return PROFILES_RECORDS

    @staticmethod
    def publications_call(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for PublicationsList._get_first()"""
        return PUBLICATIONS_RECORDS


def test_rhub_profiles(mocker):
    """Test to ensure correct response from research hub."""
    mocker.patch.object(elm.web.rhub.ProfilesList,
                        '_get_first', MockClass.profiles_call)

    out = ProfilesList("dummy")

    meta = out.meta()

    assert len(meta) == 10
    assert 'title' in meta.columns
    assert 'url' in meta.columns
    assert meta['title'].isna().sum() == 0
    assert meta['url'].isna().sum() == 0


def test_rhub_publications(mocker):
    """Test to ensure correct response from research hub."""
    mocker.patch.object(elm.web.rhub.PublicationsList,
                        '_get_first', MockClass.publications_call)

    out = PublicationsList("dummy")

    meta = out.meta()

    assert len(meta) == 10
    assert 'title' in meta.columns
    assert 'url' in meta.columns
    assert meta['title'].isna().sum() == 0
    assert meta['url'].isna().sum() == 0
