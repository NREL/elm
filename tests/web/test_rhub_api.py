"""
Test Research Hub API
"""
import os
import tempfile
from elm.web.rhub import ProfilesList
from elm.web.rhub import PublicationsList

rhub_api_key = os.getenv("RHUB_API_KEY")
PROFILES_URL = (f'https://research-hub.nrel.gov/ws/api'
                f'/524/persons?order=lastName'
                f'&pageSize=20&apiKey={rhub_api_key}')
PUBLICATIONS_URL = (f'https://research-hub.nrel.gov/ws/api'
               f'/524/research-outputs?'
               f'order=publicationYearAndAuthor&'
               f'orderBy=descending&pageSize=20&'
               f'apiKey={rhub_api_key}')

def test_rhub_from_url():
    """Test osti list, make sure we can retrieve technical reports"""
    url = (PUBLICATIONS_URL)
    rhub = PublicationsList(url, n_pages=1)
    docs = [pub.title for pub in rhub if pub.category == 'Technical Report']
    assert len(docs) > 0
