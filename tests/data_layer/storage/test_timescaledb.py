import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from psycopg2 import OperationalError as Psycopg2OperationalError

from data_layer.storage.timescaledb import TimescaleDBStorage
from data_layer.monitoring.alerts import AlertManager
from data_layer.monitoring.metrics import MetricsCollector
from data_layer.cache.redis_cache import RedisCache

@pytest.fixture
def mock_data():
    """Create mock market data"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    data = []
    for date in dates:
        data.append({
            'Date': date,
            'symbol': 'AAPL',
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000000
        })
    return pd.DataFrame(data)

@pytest.fixture
def mock_engine():
    """Create mock SQLAlchemy engine"""
    engine = MagicMock()
    connection = MagicMock()
    connection.__enter__.return_value = connection
    engine.connect.return_value = connection
    return engine

@pytest.fixture
def mock_cache():
    """Create mock Redis cache"""
    return Mock(spec=RedisCache)

@pytest.fixture
def storage(mock_engine, mock_cache):
    """Create TimescaleDB storage instance with mocked dependencies"""
    storage = TimescaleDBStorage(
        alert_manager=Mock(spec=AlertManager),
        metrics_collector=Mock(spec=MetricsCollector),
        cache=mock_cache
    )
    storage.engine = mock_engine
    return storage


def test_connect_error(storage):
    """Test database connection error"""
    with patch('sqlalchemy.create_engine') as mock_create_engine:
        # Mock connection error
        mock_create_engine.side_effect = OperationalError(
            "Connection failed",
            None,
            Psycopg2OperationalError("Connection refused")
        )
        
        with pytest.raises(OperationalError):
            storage.connect()
        
        # Verify error alert was sent
        storage.alert_manager.error.assert_called_once_with(
            title="Database Connection Error",
            message="Failed to connect to TimescaleDB: (psycopg2.OperationalError) connection to server at \"localhost\" (::1), port 5432 failed: Connection refused\n\tIs the server running on that host and accepting TCP/IP connections?\nconnection to server at \"localhost\" (127.0.0.1), port 5432 failed: Connection refused\n\tIs the server running on that host and accepting TCP/IP connections?\n\n(Background on this error at: https://sqlalche.me/e/20/e3q8)",
            source="TimescaleDBStorage",
            metadata={'error_type': 'OperationalError'}
        )

def test_disconnect(storage):
    """Test database disconnection"""
    storage.disconnect()
    
    # Verify engine was disposed
    storage.engine.dispose.assert_called_once()
    
    # Verify alert was sent
    storage.alert_manager.info.assert_called_once_with(
        title="Database Connection",
        message="Disconnected from TimescaleDB",
        source="TimescaleDBStorage"
    )

def test_save_market_data(storage, mock_data):
    """Test saving market data"""
    # Mock successful save
    storage.engine.connect.return_value.__enter__.return_value.execute.return_value = None
    
    # Mock pandas to_sql
    with patch('pandas.DataFrame.to_sql') as mock_to_sql:
        mock_to_sql.return_value = None
        
        data = {'data': mock_data}
        metadata = {'source': 'test'}
        
        result = storage.save_market_data(data, metadata)
        
        assert result is True
        storage.alert_manager.info.assert_called_once()
        
        # Verify cache was updated
        storage.cache.set.assert_called_once()
        
        # Verify performance metrics were recorded
        storage.metrics_collector.record_latency.assert_called_once_with(
            "timescaledb_data_saving",
            pytest.approx(0.729, abs=0.5)  # Allow for small variations in timing
        )

def test_save_market_data_error(storage):
    """Test saving market data with error"""
    # Mock save error
    storage.engine.connect.side_effect = Exception("Save failed")
    
    data = {'data': pd.DataFrame()}  # Empty DataFrame
    
    result = storage.save_market_data(data)
    
    assert result is False
    storage.alert_manager.error.assert_called_once()

def test_get_market_data_cache_hit(storage, mock_data):
    """Test getting market data from cache"""
    # Mock cache hit
    storage.cache.get.return_value = {
        'data': mock_data,
        'metadata': {
            'query_time_ms': 0,
            'symbols': ['AAPL'],
            'start_date': datetime(2024, 1, 1),
            'end_date': datetime(2024, 1, 31),
            'fields': None
        }
    }
    
    result = storage.get_market_data(
        symbols=['AAPL'],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31)
    )
    
    assert isinstance(result, dict)
    assert 'data' in result
    assert 'metadata' in result
    storage.cache.get.assert_called_once()
    storage.alert_manager.info.assert_called_once_with(
        title="Cache Hit",
        message="Retrieved market data for AAPL from cache",
        source="TimescaleDBStorage",
        metadata={
            'symbols': ['AAPL'],
            'start_date': datetime(2024, 1, 1),
            'end_date': datetime(2024, 1, 31),
            'fields': None
        }
    )

def test_get_market_data_cache_miss(storage, mock_data):
    """Test getting market data from database when cache miss"""
    # Mock cache miss
    storage.cache.get.return_value = None
    
    # Mock database query
    with patch('pandas.read_sql') as mock_read_sql:
        mock_read_sql.return_value = mock_data
        
        result = storage.get_market_data(
            symbols=['AAPL'],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'metadata' in result
        storage.cache.get.assert_called_once()
        storage.cache.set.assert_called_once()
        storage.alert_manager.info.assert_called_once()

def test_get_market_data(storage, mock_data):
    """Test getting market data"""
    # Mock successful query
    storage.engine.connect.return_value.__enter__.return_value.execute.return_value = None
    
    # Mock cache miss
    storage.cache.get.return_value = None
    
    # Mock DataFrame returned by read_sql
    with patch('pandas.read_sql') as mock_read_sql:
        mock_read_sql.return_value = mock_data
        
        result = storage.get_market_data(
            symbols=['AAPL'],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'metadata' in result
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) == len(mock_data)
        
        # Verify performance metrics were recorded
        storage.metrics_collector.record_latency.assert_called_once_with(
            "timescaledb_data_querying",
            pytest.approx(0.0, abs=1.0)
        )

def test_get_latest_data(storage, mock_data):
    """Test getting latest market data"""
    # Mock successful query
    storage.engine.connect.return_value.__enter__.return_value.execute.return_value = None
    
    # Mock DataFrame returned by read_sql
    with patch('pandas.read_sql') as mock_read_sql:
        mock_read_sql.return_value = mock_data.tail(1)  # Return last row
        
        result = storage.get_latest_data(symbols=['AAPL'])
        
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'metadata' in result
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) == 1
        
        # Verify performance metrics were recorded
        storage.metrics_collector.record_latency.assert_called_once_with(
            "timescaledb_latest_data_querying",
            pytest.approx(0.0, abs=1.0)
        )

def test_delete_market_data(storage):
    """Test deleting market data"""
    # Mock successful delete
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.return_value = MagicMock(rowcount=1)  # Mock successful deletion
    storage.engine.connect.return_value = connection
    
    result = storage.delete_market_data(
        symbols=['AAPL'],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31)
    )
    
    assert result is True
    storage.alert_manager.info.assert_called_once()
    storage.cache.delete.assert_called_once()
    
    # Verify performance metrics were recorded
    storage.metrics_collector.record_latency.assert_called_once_with(
        "timescaledb_data_deletion",
        pytest.approx(0.0, abs=1.0)
    ) 