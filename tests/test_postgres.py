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


class MockClass:
    """Dummy class to mock EnergyWizardPostgres.make_ref_list()"""

    @staticmethod
    def ref_call(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for EnergyWizardPostgres.make_ref_list()"""
        return REF_TEXT

    @staticmethod
    def query_call(*args, **kwargs):  # pylint: disable=unused-argument
        """Mock for EnergyWizardPostgres.query_vector_db()"""
        return QUERY_TUPLE


def test_ref_list(mocker):
    """Test to ensure correct response vector db."""
    wizard_mock = mocker.patch('elm.wizard.EnergyWizardPostgres', autospec=True)

    wizard = wizard_mock.return_value
    wizard.messages = []
    wizard.MODEL_INSTRUCTION = "Model instruction dummy."
    wizard.token_budget = 500
    wizard.model = "dummy-model-name"
    wizard.count_tokens = mocker.Mock(return_value=50)
    wizard.make_ref_list.side_effect= MockClass.ref_call
    wizard.query_vector_db.side_effect = MockClass.query_call

    question = "What is a dummy question?"
    wizard.messages.append({"role": "user", "content": question})

    message, refs = EnergyWizardPostgres.engineer_query(wizard, question)

    assert len(refs) > 0
    assert 'parentTitle' in str(refs)
    assert 'parentUrl' in str(refs)
    assert 'research-hub.nrel.gov' in str(refs)
    assert question in message
