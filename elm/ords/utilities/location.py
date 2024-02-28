# -*- coding: utf-8 -*-
"""ELM Ordinance location specification utilities"""
from abc import ABC, abstractmethod


class Location(ABC):
    """Abstract location representation."""

    def __init__(self, name):
        """

        Parameters
        ----------
        name : str
            Name of location.
        """
        self.name = name

    @property
    @abstractmethod
    def full_name(self):
        """str: Full name of location"""


class County(Location):
    """Class representing a county"""

    def __init__(self, name, state, is_parish=False):
        """

        Parameters
        ----------
        name : str
            Name of the county.
        state : str
            State containing the county.
        is_parish : bool, optional
            Flag indicating wether or not this county is classified as
            a parish. By default, ``False``.
        """
        super().__init__(name)
        self.state = state
        self.is_parish = is_parish

    @property
    def full_name(self):
        """str: Full county name in format '{name} County, {state}'"""
        loc_id = "Parish" if self.is_parish else "County"
        return f"{self.name} {loc_id}, {self.state}"
