# -*- coding: utf-8 -*-
"""ELM Ordinance county info"""
import os
import logging
from warnings import warn

import pandas as pd

from elm import ELM_DIR
from elm.ords.utilities.exceptions import ELMOrdsValueError


logger = logging.getLogger(__name__)
_COUNTY_DATA_FP = os.path.join(ELM_DIR, "ords", "data", "conus_counties.csv")


def load_all_county_info():
    """Load DataFrame containing info like names and websites for all counties.

    Returns
    -------
    pd.DataFrame
        DataFrame containing county info like names, FIPS, websites,
        etc. for all counties.
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
        county_info = load_all_county_info()

    return {
        (row["County"].casefold(), row["State"].casefold()): row["Website"]
        for __, row in county_info.iterrows()
    }


def load_counties_from_fp(county_fp):
    """Load county info base don counties in the input fp.

    Parameters
    ----------
    county_fp : path-like
        Path to csv file containing "County" and "State" columns that
        define the counties for which info should be loaded.

    Returns
    -------
    pd.DataFrame
        DataFrame containing county info like names, FIPS, websites,
        etc. for all requested counties (that were found).
    """
    counties = pd.read_csv(county_fp)
    _validate_county_input(counties)

    counties = _convert_to_title(counties, "County")
    counties = _convert_to_title(counties, "State")

    all_county_info = load_all_county_info()
    counties = counties.merge(
        all_county_info, on=["County", "State"], how="left"
    )

    counties = _filter_not_found_counties(counties)
    return _format_county_df_for_output(counties)


def _validate_county_input(df):
    """Throw error if user is missing required columns"""
    expected_cols = ["County", "State"]
    missing = [col for col in expected_cols if col not in df]
    if missing:
        msg = (
            "The following required columns were not found in the county "
            f"input: {missing}"
        )
        raise ELMOrdsValueError(msg)


def _filter_not_found_counties(df):
    """Filter out counties with null FIPS codes."""
    _warn_about_missing_counties(df)
    return df[~df.FIPS.isna()].copy()


def _warn_about_missing_counties(df):
    """Throw warning about counties that were not found in the main list."""
    not_found_counties = df[df.FIPS.isna()]
    if len(not_found_counties):
        not_found_counties_str = not_found_counties[
            ["County", "State"]
        ].to_markdown(index=False, tablefmt="psql")
        msg = (
            "The following counties were not found! Please make sure to "
            "use proper spelling and capitalization.\n"
            f"{not_found_counties_str}"
        )
        logger.warning(msg)
        warn(msg)


def _format_county_df_for_output(df):
    """Format county DataFrame for output."""
    out_cols = ["County", "State", "County Type", "FIPS", "Website"]
    df.FIPS = df.FIPS.astype(int)
    return df[out_cols].reset_index(drop=True)


def _convert_to_title(df, column):
    """Convert the values of a DataFrame column to titles."""
    df[column] = df[column].str.strip().str.casefold().str.title()
    return df
