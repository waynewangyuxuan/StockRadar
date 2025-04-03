import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union

from data_layer.fetcher.base import DataFetcherBase

class MockDataFetcher(DataFetcherBase):
    """Mock data fetcher for testing"""
    
    def __init__(self):
        super().__init__()
        self.data = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL'] * 5,
            'date': pd.date_range(start='2023-01-01', periods=10),
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100,
            'volume': np.random.randint(1000, 10000, 10)
        })
        self._setup_lineage(
            source_id="mock_source",
            source_name="Mock Data Source",
            metadata={"provider": "mock"}
        )
    
    def get_historical_data(self, symbols: List[str], start_date: datetime, 
                          end_date: datetime, fields: List[str] = None) -> Dict[str, Any]:
        """Mock historical data retrieval"""
        result = self.data.copy()
        
        # Validate dates
        result = result[result['date'].between(start_date, end_date)]
        
        # Filter data
        if symbols:
            result = result[result['symbol'].isin(symbols)]
        
        # Filter symbols
        if fields:
            result = result[['symbol', 'date'] + fields]
        
        return {
            'data': result,
            'metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'symbols': symbols
            }
        }
    
    def get_latest_data(self, symbols: List[str], fields: List[str] = None) -> Dict[str, Any]:
        """Mock latest data retrieval"""
        result = {}
        
        # Get latest data for each symbol
        for symbol in symbols:
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            if not symbol_data.empty:
                latest_data = symbol_data.iloc[-1]
                if fields:
                    latest_data = latest_data[fields]
                result[symbol] = latest_data
        
        return {
            'data': result,
            'metadata': {
                'timestamp': datetime.now(),
                'symbols': symbols
            }
        }
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Mock symbol validation"""
        valid_symbols = ['AAPL', 'GOOGL', 'MSFT']
        return [symbol for symbol in symbols if symbol in valid_symbols]

class TestDataFetcher(unittest.TestCase):
    """Test data standardization functionality"""
    
    def setUp(self):
        # Test normal data
        self.fetcher = MockDataFetcher()
        self.test_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': ['2023-01-01'],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000]
        })
    
    def test_missing_fields(self):
        # Test missing fields
        test_data = self.test_data.drop(['close'], axis=1)
        with self.assertRaises(ValueError):
            self.fetcher.normalize_data(test_data)
    
    def test_date_validation(self):
        """Test date validation functionality"""
        
        # Test string dates
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        validated_start, validated_end = self.fetcher.validate_dates(start_date, end_date)
        
        self.assertIsInstance(validated_start, datetime)
        self.assertIsInstance(validated_end, datetime)
        
        # Test invalid date order
        with self.assertRaises(ValueError):
            self.fetcher.validate_dates('2023-01-31', '2023-01-01')
        
        # Test invalid date format
        with self.assertRaises(ValueError):
            self.fetcher.validate_dates('invalid_date', '2023-01-01')

def test_historical_data_fetching(sample_market_data, date_range):
    """Test historical data fetching"""
    fetcher = MockDataFetcher()
    start_date, end_date = date_range
    
    # Test single stock
    result = fetcher.get_historical_data(
        symbols=["AAPL"],
        start_date=start_date,
        end_date=end_date
    )
    assert 'AAPL' in result['metadata']['symbols']
    assert not result['data'].empty
    assert all(result['data']['symbol'] == 'AAPL')
    
    # Test multiple stocks
    result = fetcher.get_historical_data(
        symbols=["AAPL", "GOOGL"],
        start_date=start_date,
        end_date=end_date
    )
    assert set(result['metadata']['symbols']) == {'AAPL', 'GOOGL'}
    assert not result['data'].empty
    
    # Test field filtering
    result = fetcher.get_historical_data(
        symbols=["AAPL"],
        start_date=start_date,
        end_date=end_date,
        fields=['open', 'close']
    )
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'close'}

def test_latest_data_fetching(sample_market_data):
    """Test latest data fetching"""
    fetcher = MockDataFetcher()
    
    # Test single stock
    result = fetcher.get_latest_data(symbols=["AAPL"])
    assert 'AAPL' in result['metadata']['symbols']
    assert not result['data'].empty
    assert len(result['data']) == 1
    
    # Test multiple stocks
    result = fetcher.get_latest_data(symbols=["AAPL", "MSFT"])
    assert set(result['metadata']['symbols']) == {'AAPL', 'MSFT'}
    assert not result['data'].empty
    assert len(result['data']) == 2
    
    # Test field filtering
    result = fetcher.get_latest_data(
        symbols=["AAPL"],
        fields=['open', 'close']
    )
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'close'}

def test_normalize_data(sample_market_data):
    """Test data standardization functionality"""
    fetcher = MockDataFetcher()
    
    # Test normal data
    normalized = fetcher.normalize_data(sample_market_data)
    assert pd.api.types.is_datetime64_any_dtype(normalized['date'])
    assert normalized['open'].dtype == np.float64
    assert normalized['volume'].dtype == np.float64
    
    # Test missing fields
    with self.assertRaises(ValueError):
        bad_data = sample_market_data.drop(columns=['close'])
        fetcher.normalize_data(bad_data)

@pytest.fixture
def sample_market_data():
    """Create sample market data"""
    
    # Generate AAPL data
    aapl_data = pd.DataFrame({
        'symbol': ['AAPL'] * 5,
        'date': pd.date_range('2023-01-01', periods=5),
        'open': np.random.rand(5) * 100,
        'high': np.random.rand(5) * 100,
        'low': np.random.rand(5) * 100,
        'close': np.random.rand(5) * 100,
        'volume': np.random.randint(1000, 10000, 5)
    })
    
    # Generate MSFT data
    msft_data = pd.DataFrame({
        'symbol': ['MSFT'] * 5,
        'date': pd.date_range('2023-01-01', periods=5),
        'open': np.random.rand(5) * 200,
        'high': np.random.rand(5) * 200,
        'low': np.random.rand(5) * 200,
        'close': np.random.rand(5) * 200,
        'volume': np.random.randint(1000, 10000, 5)
    })
    
    return pd.concat([aapl_data, msft_data], ignore_index=True)

@pytest.fixture
def date_range():
    """Create test date range"""
    return (
        datetime(2023, 1, 1),
        datetime(2023, 12, 31)
    )

if __name__ == '__main__':
    unittest.main() 