# -*- coding: utf-8 -*-
"""ELM Ordinance location specification utilities"""
from abc import ABC, abstractmethod
from elm.ords.utilities.location import County


class WaterDistrict(County):
    """Class representing a conservation district"""

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
        super().__init__(name, state=state, fips=fips, is_parish=is_parish)
    
    @property
    def acronym(self):
        """str: Acronym for the GCD"""
        loc = "".join(part[0].upper() for part in self.name.split())
        loc = loc.replace('&', '')
        
        return loc
