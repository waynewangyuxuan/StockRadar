"""
Data fetching module providing unified interface for various data sources
"""

from data_layer.fetcher.base import DataFetcherBase
from data_layer.fetcher.yfinance_provider import YFinanceProvider

__all__ = [
    'DataFetcherBase',
    'YFinanceProvider',
]
