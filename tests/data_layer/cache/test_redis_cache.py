import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from redis.exceptions import RedisError

from data_layer.cache.redis_cache import RedisCache
from monitoring.alerts.alert_manager import AlertManager
from monitoring.metrics.collector import MetricsCollector

@pytest.fixture
def mock_data():
    """Create mock market data"""
    return {
        'data': {
            'symbol': 'AAPL',
            'date': '2024-01-01',
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000000
        },
        'metadata': {
            'source': 'test',
            'timestamp': datetime.now().isoformat()
        }
    }

@pytest.fixture
def cache():
    """Create Redis cache instance with mocked dependencies"""
    cache = RedisCache(
        host="localhost",
        port=6379,
        db=0,
        alert_manager=Mock(spec=AlertManager),
        metrics_collector=Mock(spec=MetricsCollector)
    )
    return cache

def test_connect(cache):
    """Test Redis connection"""
    with patch('redis.Redis') as mock_redis:
        # Mock successful connection
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        cache.connect()
        
        # Verify connection was attempted
        mock_redis.assert_called_once_with(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            decode_responses=True
        )
        
        # Verify alert was sent
        cache.alert_manager.info.assert_called_once_with(
            title="Cache Connection",
            message="Successfully connected to Redis",
            source="RedisCache"
        )

def test_connect_error(cache):
    """Test Redis connection error"""
    with patch('redis.Redis') as mock_redis:
        # Mock connection error
        mock_redis.side_effect = RedisError("Connection failed")
        
        with pytest.raises(RedisError):
            cache.connect()
        
        # Verify error alert was sent
        cache.alert_manager.error.assert_called_once_with(
            title="Cache Connection Error",
            message="Failed to connect to Redis: Connection failed",
            source="RedisCache",
            metadata={'error_type': 'RedisError'}
        )

def test_disconnect(cache):
    """Test Redis disconnection"""
    # Mock client
    cache.client = MagicMock()
    
    cache.disconnect()
    
    # Verify client was closed
    cache.client.close.assert_called_once()
    
    # Verify alert was sent
    cache.alert_manager.info.assert_called_once_with(
        title="Cache Connection",
        message="Disconnected from Redis",
        source="RedisCache"
    )

def test_is_available(cache):
    """Test Redis availability check"""
    with patch('redis.Redis') as mock_redis:
        # Mock successful check
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        assert cache.is_available() is True
        
        # Mock failed check
        mock_client.ping.return_value = False
        assert cache.is_available() is False

def test_get(cache, mock_data):
    """Test getting data from cache"""
    with patch('redis.Redis') as mock_redis:
        # Mock successful get
        mock_client = MagicMock()
        mock_client.get.return_value = '{"data": {"symbol": "AAPL"}}'
        mock_redis.return_value = mock_client
        
        cache.connect()
        result = cache.get("test_key")
        
        assert result == {"data": {"symbol": "AAPL"}}
        mock_client.get.assert_called_once_with("test_key")
        
        # Verify performance metrics were recorded
        cache.metrics_collector.record_latency.assert_called_once_with(
            "redis_cache_get",
            pytest.approx(0.0, abs=1.0)
        )

def test_get_error(cache):
    """Test getting data from cache with error"""
    with patch('redis.Redis') as mock_redis:
        # Mock get error
        mock_client = MagicMock()
        mock_client.get.side_effect = RedisError("Get failed")
        mock_redis.return_value = mock_client
        
        cache.connect()
        result = cache.get("test_key")
        
        assert result is None
        cache.alert_manager.error.assert_called_once()

def test_set(cache, mock_data):
    """Test setting data in cache"""
    with patch('redis.Redis') as mock_redis:
        # Mock successful set
        mock_client = MagicMock()
        mock_client.set.return_value = True
        mock_redis.return_value = mock_client
        
        cache.connect()
        result = cache.set("test_key", mock_data)
        
        assert result is True
        mock_client.set.assert_called_once()
        
        # Verify performance metrics were recorded
        cache.metrics_collector.record_latency.assert_called_once_with(
            "redis_cache_set",
            pytest.approx(0.0, abs=1.0)
        )

def test_set_error(cache, mock_data):
    """Test setting data in cache with error"""
    with patch('redis.Redis') as mock_redis:
        # Mock set error
        mock_client = MagicMock()
        mock_client.set.side_effect = RedisError("Set failed")
        mock_redis.return_value = mock_client
        
        cache.connect()
        result = cache.set("test_key", mock_data)
        
        assert result is False
        cache.alert_manager.error.assert_called_once()

def test_delete(cache):
    """Test deleting data from cache"""
    with patch('redis.Redis') as mock_redis:
        # Mock successful delete
        mock_client = MagicMock()
        mock_client.delete.return_value = True
        mock_redis.return_value = mock_client
        
        cache.connect()
        result = cache.delete("test_key")
        
        assert result is True
        mock_client.delete.assert_called_once_with("test_key")
        
        # Verify performance metrics were recorded
        cache.metrics_collector.record_latency.assert_called_once_with(
            "redis_cache_delete",
            pytest.approx(0.0, abs=1.0)
        )

def test_delete_error(cache):
    """Test deleting data from cache with error"""
    with patch('redis.Redis') as mock_redis:
        # Mock delete error
        mock_client = MagicMock()
        mock_client.delete.side_effect = RedisError("Delete failed")
        mock_redis.return_value = mock_client
        
        cache.connect()
        result = cache.delete("test_key")
        
        assert result is False
        cache.alert_manager.error.assert_called_once()

def test_clear(cache):
    """Test clearing cache"""
    with patch('redis.Redis') as mock_redis:
        # Mock successful clear
        mock_client = MagicMock()
        mock_client.flushdb.return_value = True
        mock_redis.return_value = mock_client
        
        cache.connect()
        result = cache.clear()
        
        assert result is True
        mock_client.flushdb.assert_called_once()
        
        # Verify performance metrics were recorded
        cache.metrics_collector.record_latency.assert_called_once_with(
            "redis_cache_clear",
            pytest.approx(0.0, abs=1.0)
        )

def test_clear_error(cache):
    """Test clearing cache with error"""
    with patch('redis.Redis') as mock_redis:
        # Mock clear error
        mock_client = MagicMock()
        mock_client.flushdb.side_effect = RedisError("Clear failed")
        mock_redis.return_value = mock_client
        
        cache.connect()
        result = cache.clear()
        
        assert result is False
        cache.alert_manager.error.assert_called_once() 