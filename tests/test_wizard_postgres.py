"""
Test
"""
import os
import ast
import json
from io import BytesIO
import numpy as np
from elm import TEST_DATA_DIR
from elm.wizard import EnergyWizardPostgres


FP_REF_TXT = os.path.join(TEST_DATA_DIR, 'postgres_ref_output.txt')
FP_QUERY_TXT = os.path.join(TEST_DATA_DIR, 'postgres_query_db.txt')

with open(FP_REF_TXT, 'r', encoding='utf8') as f:
    REF_TEXT = f.read()

with open(FP_QUERY_TXT, 'r', encoding='utf8') as f:
    QUERY_TEXT = f.read()

QUERY_TUPLE = ast.literal_eval(QUERY_TEXT)
REF_TUPLE = ast.literal_eval(REF_TEXT)

os.environ["EWIZ_DB_USER"] = "user"
os.environ["EWIZ_DB_PASSWORD"] = "password"


class Cursor:
    """Dummy class for mocking database cursor objects"""
    def __init__(self):
        self.query = None

    def execute(self, query, vars):  # pylint: disable=unused-argument
        """Mock for cursor.execute()
        Parameters:
        -----------
        query : str
            SQL query to send to database
        vars : str
            Variables to include in query, ex: embedding, ids
        """
        self.query = query

    def fetchall(self):
        """Mock for cursor.fetchall()"""
        if "vector as score" in self.query:
            return QUERY_TUPLE
        if "id IN (" in self.query:
            return REF_TUPLE
        else:
            return None


class BotoClient:
    """Dummy class for mocking boto3 objects"""

    def invoke_model(self, **kwargs):  # pylint: disable=unused-argument
        """Mock for boto3.invoke_model()"""
        dummy_json_content = json.dumps({"key": "value"}).encode('utf-8')

        dummy_response = {
            'statusCode': 200,
            'body': BytesIO(dummy_json_content)
        }

        return dummy_response


def test_postgres(mocker):
    """Test to ensure correct response vector db."""

    mock_conn_cm = mocker.MagicMock()
    mock_conn = mock_conn_cm.__enter__.return_value
    mock_conn.cursor.return_value = Cursor()

    mock_connect = mocker.patch('psycopg2.connect')
    mock_connect.return_value = mock_conn_cm
    wizard = EnergyWizardPostgres(db_host='Dummy', db_port='Dummy',
                                  db_name='Dummy', db_schema='Dummy',
                                  db_table='Dummy',
                                  boto_client=BotoClient())

    question = 'Is this a dummy question?'

    strings, scores, ids = wizard.query_vector_db(question)

    assert ids is not None
    assert len(strings) == len(scores)
    assert len(ids) == len(strings)

    ref_list = wizard.make_ref_list(ids)

    assert len(ref_list) > 0
    assert 'title' in str(ref_list)
    assert 'url' in str(ref_list)
    assert 'research-hub.nrel.gov' in str(ref_list)


def test_ref_replace(mocker):
    """Test to ensure removal of double quotes from references."""
    mock_conn_cm = mocker.MagicMock()
    mock_conn = mock_conn_cm.__enter__.return_value
    mock_conn.cursor.return_value = Cursor()

    mock_connect = mocker.patch('psycopg2.connect')
    mock_connect.return_value = mock_conn_cm

    wizard = EnergyWizardPostgres(db_host='Dummy', db_port='Dummy',
                                  db_name='Dummy', db_schema='Dummy',
                                  db_table='Dummy',
                                  boto_client=BotoClient(),
                                  meta_columns=['title', 'url', 'id'])

    refs = [(chr(34), 'test.com', '5a'),
            ('remove "double" quotes', 'test_2.com', '7b')]

    ids = np.array(['7b', '5a'])

    out = wizard._format_refs(refs, ids)

    assert len(out) > 1
    for i in out:
        assert json.loads(i)


def test_ids(mocker):
    """Test to ensure only records with valid ids are returned."""
    mock_conn_cm = mocker.MagicMock()
    mock_conn = mock_conn_cm.__enter__.return_value
    mock_conn.cursor.return_value = Cursor()

    mock_connect = mocker.patch('psycopg2.connect')
    mock_connect.return_value = mock_conn_cm

    wizard = EnergyWizardPostgres(db_host='Dummy', db_port='Dummy',
                                  db_name='Dummy', db_schema='Dummy',
                                  db_table='Dummy',
                                  boto_client=BotoClient(),
                                  meta_columns=['title', 'url', 'id'])

    refs = [('title', 'test.com', '5a'),
            ('title2', 'test_2.com', '7b')]

    ids = np.array(['7c', '5a'])

    out = wizard._format_refs(refs, ids)

    assert len(out) == 1
    assert '7b' not in out


def test_sorted_refs(mocker):
    """Test to ensure references are sorted in same order as ids."""
    mock_conn_cm = mocker.MagicMock()
    mock_conn = mock_conn_cm.__enter__.return_value
    mock_conn.cursor.return_value = Cursor()

    mock_connect = mocker.patch('psycopg2.connect')
    mock_connect.return_value = mock_conn_cm

    wizard = EnergyWizardPostgres(db_host='Dummy', db_port='Dummy',
                                  db_name='Dummy', db_schema='Dummy',
                                  db_table='Dummy',
                                  boto_client=BotoClient(),
                                  meta_columns=['title', 'url', 'id'])

    refs = [('title', 'test.com', '5a'),
            ('title2', 'test_2.com', '7b')]

    ids = np.array(['7b', '5a'])

    expected = ['{"title": "title2", "url": "test_2.com", "id": "7b"}',
                '{"title": "title", "url": "test.com", "id": "5a"}']

    out = wizard._format_refs(refs, ids)

    assert expected == out
