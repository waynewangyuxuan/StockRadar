"""Tests for the data storage module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import redis
import psycopg2
from data_storage.base import DataStorageBase
from data_storage.local_storage import LocalStorage
from data_storage.redis_cache import RedisCache
from data_storage.timescaledb_storage import TimescaleDBStorage
from data_storage.version_control import VersionControl

# Test data
def create_test_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D', tz=None)
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                'ticker': ticker,
                'date': date,
                'open': np.random.uniform(100, 200),
                'high': np.random.uniform(150, 250),
                'low': np.random.uniform(50, 150),
                'close': np.random.uniform(100, 200),
                'volume': np.random.randint(1000000, 10000000)
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def test_data():
    """Fixture providing test market data."""
    return create_test_data()

@pytest.fixture
def local_storage():
    """Fixture providing a temporary local storage instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorage({'storage_path': temp_dir})
        yield storage
        # Cleanup is handled by tempfile

@pytest.fixture
def redis_storage():
    """Fixture providing a Redis storage instance."""
    storage = RedisCache({
        'host': 'localhost',
        'port': 6379,
        'db': 15,  # Use a separate test database
        'prefix': 'test:'
    })
    yield storage
    # Cleanup
    storage.redis.flushdb()

@pytest.fixture
def timescaledb_storage():
    """Fixture providing a TimescaleDB storage instance."""
    storage = TimescaleDBStorage({
        'host': 'localhost',
        'port': 5432,
        'dbname': 'stockradar_test',
        'user': 'postgres',
        'password': 'postgres',
        'schema': 'test'
    })
    yield storage
    # Cleanup
    with storage._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA test CASCADE;")
        conn.commit()

class TestDataStorageBase:
    """Test the base storage class."""
    
    def test_validate_data(self, test_data):
        """Test data validation."""
        storage = DataStorageBase()
        
        # Valid data should pass
        assert storage.validate_data(test_data)
        
        # Missing required column should fail
        invalid_data = test_data.drop(columns=['volume'])
        with pytest.raises(ValueError):
            storage.validate_data(invalid_data)
        
        # Empty data should fail
        with pytest.raises(ValueError):
            storage.validate_data(pd.DataFrame())

class TestLocalStorage:
    """Test local file-based storage."""
    
    def test_save_and_load(self, local_storage, test_data):
        """Test saving and loading data."""
        # Save data
        assert local_storage.save_data(test_data, 'test_dataset')
        
        # Load data
        loaded_data = local_storage.load_data('test_dataset')
        pd.testing.assert_frame_equal(test_data, loaded_data)
        
        # Test with filters
        filtered_data = local_storage.load_data(
            'test_dataset',
            tickers=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-05'
        )
        assert len(filtered_data) < len(test_data)
        assert all(filtered_data['ticker'] == 'AAPL')
    
    def test_versioning(self, local_storage, test_data):
        """Test data versioning."""
        # Save initial version
        assert local_storage.save_data(test_data, 'test_dataset')
        
        # Save versioned data
        modified_data = test_data.copy()
        modified_data['close'] *= 1.1
        assert local_storage.save_data(modified_data, 'test_dataset', 'v1')
        
        # Check versions
        versions = local_storage.get_versions('test_dataset')
        assert 'v1' in versions
        
        # Load specific version
        v1_data = local_storage.load_data('test_dataset', version='v1')
        pd.testing.assert_frame_equal(modified_data, v1_data)
    
    def test_delete(self, local_storage, test_data):
        """Test data deletion."""
        # Save data
        assert local_storage.save_data(test_data, 'test_dataset')
        assert local_storage.save_data(test_data, 'test_dataset', 'v1')
        
        # Delete version
        assert local_storage.delete_data('test_dataset', 'v1')
        assert 'v1' not in local_storage.get_versions('test_dataset')
        
        # Delete dataset
        assert local_storage.delete_data('test_dataset')
        assert 'test_dataset' not in local_storage.list_datasets()

class TestRedisCache:
    """Test Redis-based cache storage."""
    
    def test_save_and_load(self, redis_storage, test_data):
        """Test saving and loading data."""
        # Save data
        assert redis_storage.save_data(test_data, 'test_dataset')
        
        # Load data
        loaded_data = redis_storage.load_data('test_dataset')
        pd.testing.assert_frame_equal(test_data, loaded_data)
        
        # Test with filters
        filtered_data = redis_storage.load_data(
            'test_dataset',
            tickers=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-01-05'
        )
        assert len(filtered_data) < len(test_data)
        assert all(filtered_data['ticker'] == 'AAPL')
    
    def test_versioning(self, redis_storage, test_data):
        """Test data versioning."""
        # Save initial version
        assert redis_storage.save_data(test_data, 'test_dataset')
        
        # Save versioned data
        modified_data = test_data.copy()
        modified_data['close'] *= 1.1
        assert redis_storage.save_data(modified_data, 'test_dataset', 'v1')
        
        # Check versions
        versions = redis_storage.get_versions('test_dataset')
        assert 'v1' in versions
        
        # Load specific version
        v1_data = redis_storage.load_data('test_dataset', version='v1')
        pd.testing.assert_frame_equal(modified_data, v1_data)
    
    def test_delete(self, redis_storage, test_data):
        """Test data deletion."""
        # Save data
        assert redis_storage.save_data(test_data, 'test_dataset')
        assert redis_storage.save_data(test_data, 'test_dataset', 'v1')
        
        # Delete version
        assert redis_storage.delete_data('test_dataset', 'v1')
        assert 'v1' not in redis_storage.get_versions('test_dataset')
        
        # Delete dataset
        assert redis_storage.delete_data('test_dataset')
        assert 'test_dataset' not in redis_storage.list_datasets()

class TestTimescaleDBStorage:
    """Test TimescaleDB storage."""
    
    def test_save_and_load(self, timescaledb_storage, test_data):
        """Test saving and loading data."""
        # Sort test data
        test_data = test_data.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # Save data
        assert timescaledb_storage.save_data(test_data, 'test_dataset')
        
        # Load data
        loaded_data = timescaledb_storage.load_data('test_dataset')
        pd.testing.assert_frame_equal(test_data, loaded_data)
    
    def test_versioning(self, timescaledb_storage, test_data):
        """Test data versioning."""
        # Sort test data
        test_data = test_data.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # Save initial version
        assert timescaledb_storage.save_data(test_data, 'test_dataset')
        
        # Save versioned data
        modified_data = test_data.copy()
        modified_data['close'] *= 1.1
        assert timescaledb_storage.save_data(modified_data, 'test_dataset', 'v1')
        
        # Check versions
        versions = timescaledb_storage.get_versions('test_dataset')
        assert 'v1' in versions
        
        # Load specific version
        v1_data = timescaledb_storage.load_data('test_dataset', version='v1')
        pd.testing.assert_frame_equal(modified_data, v1_data)
    
    def test_delete(self, timescaledb_storage, test_data):
        """Test data deletion."""
        # Save data
        assert timescaledb_storage.save_data(test_data, 'test_dataset')
        assert timescaledb_storage.save_data(test_data, 'test_dataset', 'v1')
        
        # Delete version
        assert timescaledb_storage.delete_data('test_dataset', 'v1')
        assert 'v1' not in timescaledb_storage.get_versions('test_dataset')
        
        # Delete dataset
        assert timescaledb_storage.delete_data('test_dataset')
        assert 'test_dataset' not in timescaledb_storage.list_datasets()

class TestVersionControl:
    """Test version control system."""
    
    @pytest.fixture
    def version_control(self, local_storage):
        """Fixture providing a version control instance."""
        return VersionControl(local_storage)
    
    def test_create_version(self, version_control, test_data):
        """Test version creation."""
        # Create version
        version_id, success = version_control.create_version(
            test_data,
            'test_dataset',
            {'description': 'Test version'}
        )
        
        assert success
        assert version_id is not None
        assert len(version_id) == 8
        
        # Check version exists
        versions = version_control.list_versions('test_dataset')
        assert len(versions) == 1
        assert versions[0]['version_id'] == version_id
    
    def test_get_version(self, version_control, test_data):
        """Test version retrieval."""
        # Create version
        version_id, _ = version_control.create_version(
            test_data,
            'test_dataset',
            {'description': 'Test version'}
        )
        
        # Get version
        loaded_data, metadata = version_control.get_version(
            'test_dataset',
            version_id
        )
        
        pd.testing.assert_frame_equal(test_data, loaded_data)
        assert metadata['description'] == 'Test version'
    
    def test_compare_versions(self, version_control, test_data):
        """Test version comparison."""
        # Create first version
        v1_id, _ = version_control.create_version(
            test_data,
            'test_dataset',
            {'description': 'First version'}
        )
        
        # Create second version with modified data
        modified_data = test_data.copy()
        modified_data['close'] *= 1.1
        v2_id, _ = version_control.create_version(
            modified_data,
            'test_dataset',
            {'description': 'Second version'}
        )
        
        # Compare versions
        comparison = version_control.compare_versions(
            'test_dataset',
            v1_id,
            v2_id
        )
        
        assert comparison['version1']['id'] == v1_id
        assert comparison['version2']['id'] == v2_id
        assert comparison['differences']['records_diff'] == 0  # Same number of records
        assert len(comparison['differences']['tickers_added']) == 0
        assert len(comparison['differences']['tickers_removed']) == 0
    
    def test_delete_version(self, version_control, test_data):
        """Test version deletion."""
        # Create version
        version_id, _ = version_control.create_version(
            test_data,
            'test_dataset',
            {'description': 'Test version'}
        )
        
        # Delete version
        assert version_control.delete_version('test_dataset', version_id)
        
        # Check version is gone
        versions = version_control.list_versions('test_dataset')
        assert len(versions) == 0 