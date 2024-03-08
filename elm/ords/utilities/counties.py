# -*- coding: utf-8 -*-
"""ELM Ordinance county info"""
import os

import pandas as pd

from elm import ELM_DIR


_COUNTY_DATA_FP = os.path.join(ELM_DIR, "ords", "data", "conus_counties.csv")


def load_county_info():
    """Load DataFrame containing county info like Name, FIPS, and Website.

    Returns
    -------
    pd.DataFrame
        DataFrame containing county info like Name, FIPS, and Website.
    """
    county_info = pd.read_csv(_COUNTY_DATA_FP)
    county_info = _convert_to_title(county_info, "County")
    county_info = _convert_to_title(county_info, "State")
    return county_info


def county_websites(county_info=None):
    """Load mapping of county name and state to website.

    Parameters
    ----------
    county_info : pd.DataFrame, optional
        DataFrame containing county names and websites. If ``None``,
        this info is loaded using :func:`load_county_info`.
        By default, ``None``.

    Returns
    -------
    dict
        Dictionary where keys are tuples of (county, state) and keys are
        the relevant website URL. Note that county and state names are
        lowercase.
    """
    if county_info is None:
        county_info = load_county_info()

    return {
        (row["County"].casefold(), row["State"].casefold()): row["Website"]
        for __, row in county_info.iterrows()
    }


def _convert_to_title(df, column):
    """Convert the values of a DataFrame column to titles."""
    df[column] = df[column].str.strip().str.casefold().str.title()
    return df
