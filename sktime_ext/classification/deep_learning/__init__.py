"""
Deep Learning Time Series Classifiers for sktime

This module provides TSPulse, IBM's foundation model for time series classification.
"""

from .tspulse import TSPulseClassifier


__all__ = [
    'TSPulseClassifier',
]

__author__ = ['IBM Integration Team']
__version__ = '1.0.0'
