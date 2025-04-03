import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock

from data_layer.fetcher.yfinance_provider import YFinanceProvider
from monitoring.metrics.collector import DefaultMetricsCollector
from monitoring.alerts.alert_manager import AlertManager
from monitoring.lineage.tracker import LineageTracker

@pytest.fixture
def mock_yfinance_data():
    """Mock YFinance data"""
    return pd.DataFrame({
        'symbol': ['AAPL'] * 5,
        'date': pd.date_range('2023-01-01', periods=5),
        'open': np.random.rand(5) * 100,
        'high': np.random.rand(5) * 100,
        'low': np.random.rand(5) * 100,
        'close': np.random.rand(5) * 100,
        'volume': np.random.randint(1000, 10000, 5)
    })

class MockYFinanceTicker:
    """Mock YFinance Ticker object"""
    def __init__(self, data):
        self.data = data
    
    def history(self, start=None, end=None, period=None):
        return self.data

def test_provider_init():
    """Test YFinance provider initialization"""
    metrics_collector = DefaultMetricsCollector()
    alert_manager = AlertManager()
    lineage_tracker = LineageTracker()
    
    provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    
    assert provider.source_node is not None
    assert provider.source_node.metadata["description"] == "YFinance API data source"

def test_historical_data_fetching(mock_yfinance_data, monkeypatch):
    """Test historical data fetching"""
    def mock_ticker(*args, **kwargs):
        return MockYFinanceTicker(mock_yfinance_data)
    
    monkeypatch.setattr("yfinance.Ticker", mock_ticker)
    
    provider = YFinanceProvider()
    result = provider.get_historical_data(
        symbols=["AAPL"],
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    assert not result['data'].empty
    assert 'AAPL' in result['metadata']['symbols']
    
    # Test field filtering
    result = provider.get_historical_data(
        symbols=["AAPL"],
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        fields=['open', 'close']
    )
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'close'}

def test_latest_data_fetching(mock_yfinance_data, monkeypatch):
    """Test latest data fetching"""
    def mock_ticker(*args, **kwargs):
        return MockYFinanceTicker(mock_yfinance_data)
    
    monkeypatch.setattr("yfinance.Ticker", mock_ticker)
    
    provider = YFinanceProvider()
    result = provider.get_latest_data(symbols=["AAPL"])
    
    assert not result['data'].empty
    assert 'AAPL' in result['metadata']['symbols']
    
    # Test field filtering
    result = provider.get_latest_data(
        symbols=["AAPL"],
        fields=['open', 'close']
    )
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'close'}

def test_error_handling(monkeypatch):
    """Test error handling"""
    def mock_ticker_error(*args, **kwargs):
        raise Exception("API Error")
    
    monkeypatch.setattr("yfinance.Ticker", mock_ticker_error)
    
    provider = YFinanceProvider()
    with pytest.raises(ValueError):
        provider.get_historical_data(
            symbols=["AAPL"],
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )

def test_symbol_validation():
    """Test symbol validation"""
    provider = YFinanceProvider()
    
    # Test valid symbols
    valid_symbols = provider.validate_symbols(["AAPL", "GOOGL"])
    assert set(valid_symbols) == {"AAPL", "GOOGL"}
    
    # Test invalid symbols
    with pytest.raises(ValueError):
        provider.validate_symbols([])

@patch('yfinance.Ticker')
def test_yfinance_provider_init(mock_yf_ticker, mock_yfinance_data, mock_ticker,
                           metrics_collector, alert_manager, lineage_tracker):
    """Test YFinance provider initialization"""
    mock_yf_ticker.return_value = mock_ticker
    
    provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    
    assert provider.source_node is not None
    assert provider.source_node.id == "yfinance"
    assert provider.source_node.metadata["provider"] == "yfinance"
    assert provider.source_node.metadata["description"] == "YFinance API data source"

@patch('yfinance.Ticker')
def test_historical_data_fetching(mock_yf_ticker, mock_yfinance_data, mock_ticker):
    """Test historical data fetching"""
    mock_yf_ticker.return_value = mock_ticker
    
    provider = YFinanceProvider()
    result = provider.get_historical_data(
        symbols=["AAPL"],
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    assert not result['data'].empty
    assert 'AAPL' in result['metadata']['symbols']
    
    # Test field filtering
    result = provider.get_historical_data(
        symbols=["AAPL"],
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        fields=['open', 'close']
    )
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'close'}

@patch('yfinance.Ticker')
def test_latest_data_fetching(mock_yf_ticker, mock_yfinance_data, mock_ticker):
    """Test latest data fetching"""
    mock_yf_ticker.return_value = mock_ticker
    
    provider = YFinanceProvider()
    result = provider.get_latest_data(symbols=["AAPL"])
    
    assert not result['data'].empty
    assert 'AAPL' in result['metadata']['symbols']
    
    # Test field filtering
    result = provider.get_latest_data(
        symbols=["AAPL"],
        fields=['open', 'close']
    )
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'close'}

@patch('yfinance.Ticker')
def test_error_handling(mock_yf_ticker):
    """Test error handling"""
    mock_yf_ticker.side_effect = Exception("API Error")
    
    provider = YFinanceProvider()
    with pytest.raises(ValueError):
        provider.get_historical_data(
            symbols=["AAPL"],
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )

def test_symbol_validation():
    """Test symbol validation"""
    provider = YFinanceProvider()
    
    # Test valid symbols
    valid_symbols = provider.validate_symbols(["AAPL", "GOOGL"])
    assert set(valid_symbols) == {"AAPL", "GOOGL"}
    
    # Test invalid symbols
    with pytest.raises(ValueError):
        provider.validate_symbols([]) 