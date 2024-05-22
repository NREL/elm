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

DB_HOST = ("aurora-postgres-low-stage.cluster-"
           "ccklrxkcenui.us-west-2.rds.amazonaws.com")
DB_PORT = "5432"
DB_NAME = "ewiz_analysis"
DB_SCHEMA = "ewiz_schema"
DB_TABLE = "ewiz_kb"

os.environ['AWS_ACCESS_KEY_ID'] = "dummy"
os.environ['AWS_SECRET_ACCESS_KEY'] = "dummy"
os.environ['AWS_SESSION_TOKEN'] = "dummy"


class MockClass:
    """Dummy class to mock EnergyWizardPostgres.make_ref_list()"""

    @staticmethod
    def ref_call(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for EnergyWizardPostgres.make_ref_list()"""
        return REF_TEXT

    @staticmethod
    def query_call(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for EnergyWizardPostgres.make_ref_list()"""
        return QUERY_TUPLE


def test_ref_list(mocker):
    """Test to ensure correct response from research hub."""
    wizard = EnergyWizardPostgres(db_host=DB_HOST, db_port=DB_PORT,
                                  db_name=DB_NAME, db_schema=DB_SCHEMA,
                                  db_table=DB_TABLE)

    mocker.patch.object(wizard,
                        'make_ref_list', MockClass.ref_call)
    mocker.patch.object(wizard,
                        'query_vector_db', MockClass.query_call)

    question = "What is a dummy question?"

    message = wizard.engineer_query(question)[0]
    refs = wizard.engineer_query(question)[-1]

    assert len(refs) > 0
    assert 'parentTitle' in str(refs)
    assert 'parentUrl' in str(refs)
    assert 'research-hub.nrel.gov' in str(refs)
    assert question in message
