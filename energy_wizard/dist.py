# -*- coding: utf-8 -*-
"""
Energy Wizard distance metrics
"""
from scipy import spatial


class DistanceMetrics:
    """Class to hold functions to evaluate the similarity (distance) between
    text embeddings"""

    @staticmethod
    def cosine_dist(x, y):
        out = 1 - spatial.distance.cosine(x, y)
        return out
