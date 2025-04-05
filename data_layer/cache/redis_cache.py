import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError

from data_layer.monitoring.alerts import AlertManager
from data_layer.monitoring.metrics import MetricsCollector

class RedisCache:
    """Redis cache implementation for market data"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        alert_manager: Optional[AlertManager] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize Redis cache
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            alert_manager: Alert manager instance
            metrics_collector: Metrics collector instance
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.alert_manager = alert_manager or AlertManager()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.client = None
    
    def connect(self) -> None:
        """Connect to Redis"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            self.client.ping()  # Test connection
            self.alert_manager.info(
                title="Cache Connection",
                message="Successfully connected to Redis",
                source=self.__class__.__name__
            )
        except RedisError as e:
            self.alert_manager.error(
                title="Cache Connection Error",
                message=f"Failed to connect to Redis: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': type(e).__name__}
            )
            raise
    
    def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.client:
            self.client.close()
            self.alert_manager.info(
                title="Cache Connection",
                message="Disconnected from Redis",
                source=self.__class__.__name__
            )
    
    def is_available(self) -> bool:
        """Check if Redis is available
        
        Returns:
            bool: True if Redis is available, False otherwise
        """
        try:
            if not self.client:
                self.connect()
            return self.client.ping()
        except RedisError:
            return False
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Dict[str, Any]]: Cached data if found, None otherwise
        """
        try:
            start_time = datetime.now()
            data = self.client.get(key)
            if data:
                return json.loads(data)
            return None
        except RedisError as e:
            self.alert_manager.error(
                title="Cache Error",
                message=f"Failed to get data from Redis: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': type(e).__name__, 'key': key}
            )
            return None
        finally:
            if self.metrics_collector:
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self.metrics_collector.record_latency("redis_cache_get", latency)
    
    def set(self, key: str, value: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """Set data in cache
        
        Args:
            key: Cache key
            value: Data to cache
            expire: Expiration time in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = datetime.now()
            data = json.dumps(value)
            self.client.set(key, data, ex=expire)
            return True
        except RedisError as e:
            self.alert_manager.error(
                title="Cache Error",
                message=f"Failed to set data in Redis: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': type(e).__name__, 'key': key}
            )
            return False
        finally:
            if self.metrics_collector:
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self.metrics_collector.record_latency("redis_cache_set", latency)
    
    def delete(self, key: str) -> bool:
        """Delete data from cache
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = datetime.now()
            self.client.delete(key)
            return True
        except RedisError as e:
            self.alert_manager.error(
                title="Cache Error",
                message=f"Failed to delete data from Redis: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': type(e).__name__, 'key': key}
            )
            return False
        finally:
            if self.metrics_collector:
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self.metrics_collector.record_latency("redis_cache_delete", latency)
    
    def clear(self) -> bool:
        """Clear all data from cache
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = datetime.now()
            self.client.flushdb()
            return True
        except RedisError as e:
            self.alert_manager.error(
                title="Cache Error",
                message=f"Failed to clear Redis cache: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': type(e).__name__}
            )
            return False
        finally:
            if self.metrics_collector:
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self.metrics_collector.record_latency("redis_cache_clear", latency) 