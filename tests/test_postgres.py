"""
Test
"""
import os
import ast
from elm import TEST_DATA_DIR
from elm.wizard import EnergyWizardPostgres


FP_REF_TXT = os.path.join(TEST_DATA_DIR, 'postgres_ref_list.txt')
FP_QUERY_TXT = os.path.join(TEST_DATA_DIR, 'postgres_query_test.txt')

with open(FP_REF_TXT, 'r', encoding='utf8') as f:
    REF_TEXT = f.read()

with open(FP_QUERY_TXT, 'r', encoding='utf8') as f:
    QUERY_TEXT = f.read()

QUERY_TUPLE = ast.literal_eval(QUERY_TEXT)



class Cursor:
    def __init__(self):

    def fetchall(query):
        if "vecotor as score" in query:
            return SQL_EMBEDDING_LOOKUP
        if "id IN (" in query:
           return SQL_REF_RESPONSE
        else:
            return None


def test_ref_list(mocker):
    """Test to ensure correct response vector db."""

    os.environ["EWIZ_DB_USER"] = "user"

    wizard = EnergyWizardPostgres(Cursor())

    strings, score, ids = wizard.query_vector_db(question)

    assert

    ref_list = wizard.make_ref_list(ids)

    assert
