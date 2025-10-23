"""
Truffle Occurrence Data Downloader

A specialized Python package for downloading and processing truffle occurrence data
from GBIF (Global Biodiversity Information Facility).
"""

__version__ = "1.0.0"
__author__ = "Truffle Research Community"
__email__ = "support@truffle-downloader.org"

from .downloader import GBIFTruffleDownloader
from .validators import DataValidator
from .exporters import DataExporter

__all__ = [
    "GBIFTruffleDownloader",
    "DataValidator", 
    "DataExporter"
]