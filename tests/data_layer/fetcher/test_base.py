import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union

from data_layer.fetcher.base import DataFetcherBase

class MockDataFetcher(DataFetcherBase):
    """用于测试的模拟数据获取器"""
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        self._setup_lineage(
            source_id="mock_source",
            source_name="Mock Data Source",
            metadata={"provider": "mock"}
        )
    
    def get_historical_data(self, symbols, start_date, end_date, fields=None):
        """模拟历史数据获取"""
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # 验证日期
        start_date, end_date = self.validate_dates(start_date, end_date)
        
        # 过滤数据
        mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
        filtered_data = self.data[mask].copy()
        
        # 过滤股票代码
        filtered_data = filtered_data[filtered_data['symbol'].isin(symbols)]
        
        # 过滤字段
        if fields:
            filtered_data = filtered_data[['symbol', 'date'] + [f for f in fields if f not in ['symbol', 'date']]]
        
        return {
            'data': filtered_data,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def get_latest_data(self, symbols, fields=None):
        """模拟最新数据获取"""
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # 获取每个股票的最新数据
        latest_data = (
            self.data[self.data['symbol'].isin(symbols)]
            .sort_values('date')
            .groupby('symbol')
            .last()
            .reset_index()
        )
        
        # 过滤字段
        if fields:
            latest_data = latest_data[['symbol', 'date'] + [f for f in fields if f not in ['symbol', 'date']]]
        
        return {
            'data': latest_data,
            'symbols': symbols,
            'timestamp': datetime.now()
        }
    
    def validate_symbols(self, symbols):
        """模拟股票代码验证"""
        if isinstance(symbols, str):
            symbols = [symbols]
        valid_symbols = [s for s in symbols if s in self.data['symbol'].unique()]
        if not valid_symbols:
            raise ValueError("No valid symbols provided")
        return valid_symbols

def test_normalize_data(sample_market_data):
    """测试数据标准化功能"""
    fetcher = MockDataFetcher(sample_market_data)
    
    # 测试正常数据
    normalized = fetcher.normalize_data(sample_market_data)
    assert pd.api.types.is_datetime64_any_dtype(normalized['date'])
    assert normalized['open'].dtype == np.float64
    assert normalized['volume'].dtype == np.float64
    
    # 测试缺失字段
    with pytest.raises(ValueError):
        bad_data = sample_market_data.drop(columns=['close'])
        fetcher.normalize_data(bad_data)

def test_validate_dates():
    """测试日期验证功能"""
    fetcher = MockDataFetcher(pd.DataFrame())
    
    # 测试字符串日期
    start_date, end_date = fetcher.validate_dates("2023-01-01", "2023-01-10")
    assert isinstance(start_date, datetime)
    assert isinstance(end_date, datetime)
    
    # 测试datetime对象
    start_date, end_date = fetcher.validate_dates(
        datetime(2023, 1, 1),
        datetime(2023, 1, 10)
    )
    assert isinstance(start_date, datetime)
    assert isinstance(end_date, datetime)
    
    # 测试无效日期范围
    with pytest.raises(ValueError):
        fetcher.validate_dates("2023-01-10", "2023-01-01")

def test_historical_data_fetching(sample_market_data, date_range):
    """测试历史数据获取"""
    fetcher = MockDataFetcher(sample_market_data)
    start_date, end_date = date_range
    
    # 测试单个股票
    result = fetcher.get_historical_data(
        symbols="AAPL",
        start_date=start_date,
        end_date=end_date
    )
    assert 'AAPL' in result['symbols']
    assert not result['data'].empty
    assert all(result['data']['symbol'] == 'AAPL')
    
    # 测试多个股票
    result = fetcher.get_historical_data(
        symbols=["AAPL", "GOOGL"],
        start_date=start_date,
        end_date=end_date
    )
    assert set(result['symbols']) == {'AAPL', 'GOOGL'}
    assert not result['data'].empty
    
    # 测试字段过滤
    result = fetcher.get_historical_data(
        symbols="AAPL",
        start_date=start_date,
        end_date=end_date,
        fields=['open', 'close']
    )
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'close'}

def test_latest_data_fetching(sample_market_data):
    """测试最新数据获取"""
    fetcher = MockDataFetcher(sample_market_data)
    
    # 测试单个股票
    result = fetcher.get_latest_data(symbols="AAPL")
    assert 'AAPL' in result['symbols']
    assert not result['data'].empty
    assert len(result['data']) == 1
    assert result['data'].iloc[0]['symbol'] == 'AAPL'
    
    # 测试多个股票
    result = fetcher.get_latest_data(symbols=["AAPL", "MSFT"])
    assert set(result['symbols']) == {'AAPL', 'MSFT'}
    assert not result['data'].empty
    assert len(result['data']) == 2
    
    # 测试字段过滤
    result = fetcher.get_latest_data(
        symbols="AAPL",
        fields=['open', 'close']
    )
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'close'}

@pytest.fixture
def sample_market_data():
    """创建样本市场数据"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31')
    data = []
    
    # 生成AAPL数据
    for date in dates:
        data.append({
            'symbol': 'AAPL',
            'date': date,
            'open': np.random.uniform(90, 110),
            'high': np.random.uniform(100, 120),
            'low': np.random.uniform(80, 100),
            'close': np.random.uniform(90, 110),
            'volume': np.random.randint(1000000, 5000000)
        })
    
    # 生成MSFT数据
    for date in dates:
        data.append({
            'symbol': 'MSFT',
            'date': date,
            'open': np.random.uniform(35000, 37000),
            'high': np.random.uniform(36000, 38000),
            'low': np.random.uniform(35000, 37000),
            'close': np.random.uniform(35000, 37000),
            'volume': np.random.randint(1000000, 5000000)
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def date_range():
    """创建测试日期范围"""
    return (
        datetime(2023, 1, 1),
        datetime(2023, 12, 31)
    ) 