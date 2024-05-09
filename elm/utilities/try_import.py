"""ELM import utility."""
import importlib
import logging
from warnings import warn

logger = logging.getLogger(__name__)


def try_import(package_name):
    """Function to import packages needed for
    postgres and aws.

    Parameters
    ----------
    package_name : str
        Name of package to import.

    Returns
    -------
    p : package
        imported package.
    """
    try:
        p = importlib.import_module(package_name)
        return p
    except ImportError:
        msg = (f'Unable to import {package_name}. '
               'Please ensure you have the package '
               'installed and spelled correctly '
               'before proceeding.')
        logger.warning(msg)
        warn(msg)
