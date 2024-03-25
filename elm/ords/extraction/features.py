# -*- coding: utf-8 -*-
"""ELM Ordinance mutually-exclusive features class."""


class SetbackFeatures:
    """Utility class to get mutually-exclusive feature descriptions."""

    DEFAULT_FEATURE_DESCRIPTIONS = {
        "struct": [
            "occupied dwellings",
            "buildings",
            "structures",
            "residences",
        ],
        "pline": ["property lines", "parcels", "subdivisions"],
        "roads": ["roads"],  # , "rights-of-way"],
        "rail": ["railroads"],
        "trans": [
            "overhead electrical transmission lines",
            "overhead utility lines",
            "utility easements",
            "utility lines",
            "power lines",
            "electrical lines",
            "transmission lines",
        ],
        "water": ["lakes", "reservoirs", "streams", "rivers", "wetlands"],
    }
    FEATURES_AS_IGNORE = {
        "struct": "structures",
        "pline": "property lines",
        "roads": "roads",
        "rail": "railroads",
        "trans": "transmission lines",
        "water": "wetlands",
    }
    FEATURE_CLARIFICATIONS = {
        "struct": "",
        "pline": "",
        "roads": "Roads may also be labeled as rights-of-way. ",
        "rail": "",
        "trans": "",
        "water": "",
    }

    def __init__(self):
        self._validate_descriptions()

    def __iter__(self):
        for feature_id in self.DEFAULT_FEATURE_DESCRIPTIONS:
            feature, ignore = self._keep_and_ignore(feature_id)
            clarification = self.FEATURE_CLARIFICATIONS.get(feature_id, "")
            yield {
                "feature_id": feature_id,
                "feature": feature,
                "ignore_features": ignore,
                "feature_clarifications": clarification,
            }

    def _validate_descriptions(self):
        """Ensure all features have at least one description."""
        features_missing_descriptors = set()
        for feature, descriptions in self.DEFAULT_FEATURE_DESCRIPTIONS.items():
            if len(descriptions) < 1:
                features_missing_descriptors.add(feature)

        if any(features_missing_descriptors):
            raise ValueError(
                f"The following features are missing descriptors: "
                f"{features_missing_descriptors}"
            )

    def _keep_and_ignore(self, feature_id):
        """Get the keep and ignore phrases for a feature."""
        keep_keywords = self.DEFAULT_FEATURE_DESCRIPTIONS[feature_id]
        ignore = [
            keyword
            for feat_id, keyword in self.FEATURES_AS_IGNORE.items()
            if feat_id != feature_id
        ]

        keep_phrase = _join_keywords(keep_keywords, final_sep=", and/or ")
        ignore_phrase = _join_keywords(ignore, final_sep=", and ")

        return keep_phrase, ignore_phrase


def _join_keywords(keywords, final_sep):
    """Join a list of keywords/descriptions."""
    if len(keywords) < 1:
        return ""

    if len(keywords) == 1:
        return keywords[0]

    comma_separated = ", ".join(keywords[:-1])
    return final_sep.join([comma_separated, keywords[-1]])
