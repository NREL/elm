"""ELM validation utilities."""
import os


def validate_azure_api_params(azure_api_key=None, azure_version=None,
                              azure_endpoint=None):
    """Validate Azure OpenAI API parameters.

    Parameters
    ----------
    azure_api_key, azure_version, azure_endpoint : str, optional
        Azure OpenAI API key, version, and endpoint. By default,
        ``None``, which attempts to read these variables from the
        following environment variables:

            - AZURE_OPENAI_API_KEY
            - AZURE_OPENAI_VERSION
            - AZURE_OPENAI_ENDPOINT

        If any of these are still `None` after reading from the
        environment, an error is raised.

    Returns
    -------
    azure_api_key, azure_version, azure_endpoint : str
        API key, version, and endpoint that can be used to initialize
        an Azure OpenAI service.
    """
    azure_api_key = azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    azure_version = azure_version or os.environ.get("AZURE_OPENAI_VERSION")
    azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    assert azure_api_key is not None, "Must set AZURE_OPENAI_API_KEY!"
    assert azure_version is not None, "Must set AZURE_OPENAI_VERSION!"
    assert azure_endpoint is not None, "Must set AZURE_OPENAI_ENDPOINT!"
    return azure_api_key, azure_version, azure_endpoint
