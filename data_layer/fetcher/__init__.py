"""
数据获取模块，提供各种数据源的统一接口
"""

from data_layer.fetcher.base import DataFetcherBase
from data_layer.fetcher.yfinance_provider import YFinanceProvider

__all__ = [
    'DataFetcherBase',
    'YFinanceProvider',
]
