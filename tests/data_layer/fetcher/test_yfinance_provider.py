"""Test YFinanceProvider"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import Mock, patch

from data_layer.fetcher.yfinance_provider import YFinanceProvider
from monitoring.metrics import DefaultMetricsCollector
from monitoring.alerts import AlertManager
from monitoring.lineage import LineageTracker

@pytest.fixture
def mock_data():
    """Create mock data that matches yfinance structure"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    data = []
    for date in dates:
        data.append({
            'Date': date,
            'Open': 100.0,
            'High': 101.0,
            'Low': 99.0,
            'Close': 100.5,
            'Volume': 1000000,
            'symbol': 'AAPL'
        })
    return pd.DataFrame(data)

@pytest.fixture
def mock_yfinance(mock_data):
    """Mock yfinance module"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        yield mock_ticker

@pytest.fixture
def provider():
    """Create YFinanceProvider instance with mocked dependencies"""
    metrics_collector = Mock(spec=DefaultMetricsCollector)
    alert_manager = Mock(spec=AlertManager)
    lineage_tracker = Mock(spec=LineageTracker)
    
    return YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )

def test_get_historical_data(mock_yfinance):
    """Test getting historical data"""
    provider = YFinanceProvider()
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    symbols = ['AAPL']
    
    result = provider.get_historical_data(symbols, start_date, end_date)
    
    assert isinstance(result, dict)
    assert 'data' in result
    assert 'metadata' in result
    
    df = result['data']
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 31  # One row per day in January
    assert 'symbol' in df.columns
    assert 'Date' in df.columns
    assert 'Close' in df.columns
    
    metadata = result['metadata']
    assert metadata['symbols'] == symbols
    assert metadata['start_date'] == start_date
    assert metadata['end_date'] == end_date
    assert metadata['fields'] is None
    assert 'fetching_time_ms' in metadata

def test_get_historical_data_with_fields(provider, mock_data):
    """Test historical data fetching with field filtering"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data.set_index('Date')
        mock_ticker.return_value = mock_ticker_instance
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 30)
        symbols = ['AAPL', 'MSFT']
        fields = ['Open', 'Close']
        
        result = provider.get_historical_data(symbols, start_date, end_date, fields)
        
        # Verify filtered columns
        df = result['data']
        assert set(df.columns) == {'symbol', 'Date', 'Open', 'Close'}

def test_get_latest_data(provider, mock_data):
    """Test latest data fetching"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data.set_index('Date')
        mock_ticker.return_value = mock_ticker_instance
        
        symbols = ['AAPL', 'MSFT']
        
        result = provider.get_latest_data(symbols)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'metadata' in result
        
        # Verify DataFrame
        df = result['data']
        assert isinstance(df, pd.DataFrame)
        assert 'symbol' in df.columns
        assert 'Date' in df.columns
        assert len(df) == len(symbols)  # One row per symbol
        
        # Verify metadata
        metadata = result['metadata']
        assert 'fetching_time_ms' in metadata
        assert metadata['symbols'] == symbols
        assert 'timestamp' in metadata

def test_get_latest_data_with_fields(provider, mock_data):
    """Test latest data fetching with field filtering"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data.set_index('Date')
        mock_ticker.return_value = mock_ticker_instance
        
        symbols = ['AAPL', 'MSFT']
        fields = ['Open', 'Close']
        
        result = provider.get_latest_data(symbols, fields)
        
        # Verify filtered columns
        df = result['data']
        assert set(df.columns) == {'symbol', 'Date', 'Open', 'Close'}

def test_empty_data_handling(provider):
    """Test handling of empty data"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker.return_value = mock_ticker_instance
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 30)
        symbols = ['AAPL']
        
        # Should raise ValueError when no data is returned
        with pytest.raises(ValueError, match="Failed to fetch data for any symbols"):
            provider.get_historical_data(symbols, start_date, end_date)
        
        # Verify warning was logged
        provider.alert_manager.warning.assert_called_once()

def test_error_handling(provider):
    """Test error handling during data fetching"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = Exception("API Error")
        mock_ticker.return_value = mock_ticker_instance
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 30)
        symbols = ['AAPL']
        
        # Should raise the original exception
        with pytest.raises(Exception, match="API Error"):
            provider.get_historical_data(symbols, start_date, end_date)
        
        # Verify error was logged
        provider.alert_manager.error.assert_called_once()

def test_validate_symbols(provider):
    """Test symbol validation"""
    with patch('yfinance.Ticker') as mock_ticker:
        # Configure mock for valid and invalid symbols
        mock_valid = Mock()
        mock_valid.info = {'symbol': 'AAPL'}
        mock_invalid = Mock()
        mock_invalid.info = None
        
        mock_ticker.side_effect = lambda symbol: mock_valid if symbol == 'AAPL' else mock_invalid
        
        # Test with valid and invalid symbols
        symbols = ['AAPL', 'INVALID']
        valid_symbols = provider.validate_symbols(symbols)
        
        assert valid_symbols == ['AAPL']
        provider.alert_manager.warning.assert_called_once() 