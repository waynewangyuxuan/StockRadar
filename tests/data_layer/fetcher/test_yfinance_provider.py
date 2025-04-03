import pytest
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import patch, MagicMock, Mock

from data_layer.fetcher.yfinance_provider import YFinanceProvider

@pytest.fixture
def mock_yfinance_data():
    """模拟YFinance数据"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    data = pd.DataFrame({
        'Date': dates,
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.5,
        'Volume': 1000000
    })
    return data

@pytest.fixture
def mock_ticker(mock_yfinance_data):
    """模拟YFinance Ticker对象"""
    mock = MagicMock()
    mock.history.return_value = mock_yfinance_data
    mock.info = {'regularMarketPrice': 100.0}
    return mock

def test_yfinance_provider_init(metrics_collector, alert_manager, lineage_tracker):
    """测试YFinance提供者初始化"""
    provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    
    assert provider.source_node is not None
    assert provider.source_node.id == "yfinance"
    assert provider.source_node.metadata["provider"] == "yfinance"
    assert provider.source_node.metadata["description"] == "YFinance API数据源"

@patch('yfinance.Ticker')
def test_get_historical_data(mock_yf_ticker, mock_yfinance_data, mock_ticker,
                           metrics_collector, alert_manager, lineage_tracker):
    """测试历史数据获取"""
    mock_yf_ticker.return_value = mock_ticker
    
    provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)
    
    result = provider.get_historical_data(
        symbols=['AAPL'],
        start_date=start_date,
        end_date=end_date
    )
    
    assert not result['data'].empty
    assert 'AAPL' in result['symbols']
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'high', 'low', 'close', 'volume'}
    assert result['data']['symbol'].iloc[0] == 'AAPL'
    
    # 测试字段过滤
    result = provider.get_historical_data(
        symbols=['AAPL'],
        start_date=start_date,
        end_date=end_date,
        fields=['open', 'close']
    )
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'close'}

@patch('yfinance.Ticker')
def test_get_latest_data(mock_yf_ticker, mock_yfinance_data, mock_ticker,
                        metrics_collector, alert_manager, lineage_tracker):
    """测试最新数据获取"""
    mock_yf_ticker.return_value = mock_ticker
    
    provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    
    result = provider.get_latest_data(symbols=['AAPL'])
    
    assert not result['data'].empty
    assert 'AAPL' in result['symbols']
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'high', 'low', 'close', 'volume'}
    assert result['data']['symbol'].iloc[0] == 'AAPL'
    
    # 测试字段过滤
    result = provider.get_latest_data(
        symbols=['AAPL'],
        fields=['open', 'close']
    )
    assert set(result['data'].columns) == {'symbol', 'date', 'open', 'close'}

@patch('yfinance.Ticker')
def test_error_handling(mock_yf_ticker, metrics_collector, alert_manager, lineage_tracker):
    """测试错误处理"""
    mock_ticker = MagicMock()
    mock_ticker.info = None
    mock_yf_ticker.return_value = mock_ticker
    
    provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    
    with pytest.raises(ValueError):
        provider.validate_symbols(['INVALID'])

@patch('yfinance.Ticker')
def test_validate_symbols(mock_yf_ticker, metrics_collector, alert_manager, lineage_tracker):
    """测试股票代码验证"""
    mock_ticker = MagicMock()
    mock_ticker.info = None
    mock_yf_ticker.return_value = mock_ticker
    
    provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    
    with pytest.raises(ValueError):
        provider.validate_symbols(['INVALID']) 