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

    def __init__(self, name, state, fips=None, is_parish=False):
        """

        Parameters
        ----------
        name : str
            Name of the county.
        state : str
            State containing the county.
        fips : int | str, optional
            Optional county FIPS code. By default, ``None``.
        is_parish : bool, optional
            Flag indicating wether or not this county is classified as
            a parish. By default, ``False``.
        """
        super().__init__(name)
        self.state = state
        self.fips = fips
        self.is_parish = is_parish

    @property
    def full_name(self):
        """str: Full county name in format '{name} County, {state}'"""
        loc_id = "Parish" if self.is_parish else "County"
        return f"{self.name} {loc_id}, {self.state}"

    def __repr__(self):
        return f"County({self.name}, {self.state}, is_parish={self.is_parish})"

    def __str__(self):
        return self.full_name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.name.casefold() == other.name.casefold()
                and self.state.casefold() == other.state.casefold()
                and self.is_parish == other.is_parish
            )
        if isinstance(other, str):
            return (
                self.full_name.casefold() == other.casefold()
                or f"{self.name}, {self.state}".casefold() == other.casefold()
            )
        return False
