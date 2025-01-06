import requests
url = "https://www.osti.gov/api/v1/records"

_session = requests.Session()
response = _session.get(url)