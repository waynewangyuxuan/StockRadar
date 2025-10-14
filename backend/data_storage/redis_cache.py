"""Redis-based cache implementation."""

import redis
import pandas as pd
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from .base import DataStorageBase
import logging
import io

class RedisCache(DataStorageBase):
    """Redis-based cache for market data.
    
    This implementation uses Redis for fast in-memory access to frequently used data.
    Data is stored in Redis using the following key patterns:
    - dataset:{name} -> Main data
    - dataset:{name}:version:{version} -> Versioned data
    - datasets -> Set of dataset names
    - versions:{dataset_name} -> Set of version names
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Redis cache.
        
        Args:
            config: Dictionary with Redis configuration:
                - host: Redis host (default: localhost)
                - port: Redis port (default: 6379)
                - db: Redis database number (default: 0)
                - password: Optional Redis password
                - prefix: Key prefix (default: stockradar:)
        """
        super().__init__(config)
        
        # Set up Redis connection
        self.redis = redis.Redis(
            host=self.config.get('host', 'localhost'),
            port=self.config.get('port', 6379),
            db=self.config.get('db', 0),
            password=self.config.get('password'),
            decode_responses=False  # Don't decode responses since we're storing binary data
        )
        
        self.prefix = self.config.get('prefix', 'stockradar:')
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _get_dataset_key(self, dataset_name: str) -> str:
        """Get Redis key for a dataset."""
        return f"{self.prefix}dataset:{dataset_name}"
        
    def _get_version_key(self, dataset_name: str, version: str) -> str:
        """Get Redis key for a versioned dataset."""
        return f"{self.prefix}dataset:{dataset_name}:version:{version}"
        
    def _get_datasets_key(self) -> str:
        """Get Redis key for the set of dataset names."""
        return f"{self.prefix}datasets"
        
    def _get_versions_key(self, dataset_name: str) -> str:
        """Get Redis key for the set of version names."""
        return f"{self.prefix}versions:{dataset_name}"
        
    def _serialize_data(self, data: pd.DataFrame) -> bytes:
        """Serialize DataFrame to bytes for Redis storage."""
        buffer = io.BytesIO()
        data.to_parquet(buffer)
        return buffer.getvalue()
        
    def _deserialize_data(self, data_bytes: bytes) -> pd.DataFrame:
        """Deserialize bytes from Redis to DataFrame."""
        buffer = io.BytesIO(data_bytes)
        return pd.read_parquet(buffer)
    
    def save_data(self, 
                 data: pd.DataFrame,
                 dataset_name: str,
                 version: Optional[str] = None) -> bool:
        """Save market data to Redis cache.
        
        Args:
            data: DataFrame containing market data
            dataset_name: Name/identifier for the dataset
            version: Optional version identifier
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Validate data first
            self.validate_data(data)
            
            # Serialize data
            data_bytes = self._serialize_data(data)
            
            # Start Redis transaction
            pipe = self.redis.pipeline()
            
            # Save main data
            key = self._get_dataset_key(dataset_name)
            pipe.set(key, data_bytes)
            
            # Add to datasets set
            pipe.sadd(self._get_datasets_key(), dataset_name)
            
            # Save versioned data if version provided
            if version:
                version_key = self._get_version_key(dataset_name, version)
                pipe.set(version_key, data_bytes)
                pipe.sadd(self._get_versions_key(dataset_name), version)
                
            # Execute transaction
            pipe.execute()
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data to Redis: {str(e)}")
            return False
    
    def load_data(self,
                 dataset_name: str,
                 tickers: Optional[List[str]] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 version: Optional[str] = None) -> pd.DataFrame:
        """Load market data from Redis cache.
        
        Args:
            dataset_name: Name/identifier for the dataset
            tickers: Optional list of ticker symbols to load
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            version: Optional version to load
            
        Returns:
            DataFrame containing the requested market data
        """
        try:
            # Determine which key to use
            if version:
                key = self._get_version_key(dataset_name, version)
            else:
                key = self._get_dataset_key(dataset_name)
                
            # Load data from Redis
            data_bytes = self.redis.get(key)
            if data_bytes is None:
                raise KeyError(f"Dataset not found: {dataset_name}")
                
            # Deserialize data
            data = self._deserialize_data(data_bytes)
            
            # Apply filters
            if tickers:
                data = data[data['ticker'].isin(tickers)]
            if start_date:
                data = data[data['date'] >= start_date]
            if end_date:
                data = data[data['date'] <= end_date]
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data from Redis: {str(e)}")
            return pd.DataFrame()
    
    def delete_data(self,
                   dataset_name: str,
                   version: Optional[str] = None) -> bool:
        """Delete market data from Redis cache.
        
        Args:
            dataset_name: Name/identifier for the dataset
            version: Optional version to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            pipe = self.redis.pipeline()
            
            if version:
                # Delete specific version
                version_key = self._get_version_key(dataset_name, version)
                pipe.delete(version_key)
                pipe.srem(self._get_versions_key(dataset_name), version)
            else:
                # Delete all versions
                versions = self.get_versions(dataset_name)
                for ver in versions:
                    pipe.delete(self._get_version_key(dataset_name, ver))
                pipe.delete(self._get_versions_key(dataset_name))
                
                # Delete main data
                pipe.delete(self._get_dataset_key(dataset_name))
                pipe.srem(self._get_datasets_key(), dataset_name)
                
            pipe.execute()
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting data from Redis: {str(e)}")
            return False
    
    def list_datasets(self) -> List[str]:
        """List all available datasets in Redis cache.
        
        Returns:
            List of dataset names/identifiers
        """
        try:
            datasets = self.redis.smembers(self._get_datasets_key())
            return sorted(list(datasets))
            
        except Exception as e:
            self.logger.error(f"Error listing datasets from Redis: {str(e)}")
            return []
    
    def get_versions(self, dataset_name: str) -> List[str]:
        """Get available versions for a dataset.
        
        Args:
            dataset_name: Name/identifier for the dataset
            
        Returns:
            List of version identifiers
        """
        try:
            versions = self.redis.smembers(self._get_versions_key(dataset_name))
            return [v.decode('utf-8') for v in versions] if versions else []
        except Exception as e:
            self.logger.error(f"Error getting versions from Redis: {str(e)}")
            return []
