# Data Storage Module

This module provides a unified interface for managing data storage operations across different storage backends. It simplifies the process of saving, loading, and managing various types of data used in the StockRadar trading system.

## Overview

The data storage module consists of:

- **Base class** (`DataStorageBase`): Abstract interface for storage implementations
- **Storage backends**:
  - `LocalStorage`: File-based storage using Parquet format
  - `RedisCache`: In-memory cache using Redis
  - `TimescaleDBStorage`: Time-series optimized PostgreSQL storage
- **Version control** (`VersionControl`): Data versioning and snapshot management 
- **Unified manager** (`StorageManager`): Consolidated interface for all storage operations

## Using the StorageManager

The `StorageManager` class provides a unified interface for all storage operations and should be the primary entry point for interacting with the storage system:

```python
from backend.data_storage import StorageManager

# Initialize with default config
storage = StorageManager()

# Store market data
data_id = storage.store_market_data(market_df, metadata={"source": "yahoo"})

# Store backtest results
backtest_id = storage.store_backtest_results(
    results=backtest_results,
    strategy_name="momentum_strategy", 
    metadata={"author": "trading_team"}
)

# Retrieve data
market_data = storage.get_market_data(version_id=data_id)

# Get backtest results
results = storage.get_backtest_results(
    strategy_name="momentum_strategy",
    version_id=backtest_id
)
```

## Configuration

The storage system is configured using YAML files. The default configuration file location is `config/storage_config.yaml`. 

A typical configuration might look like:

```yaml
storage:
  backend: "local"  # Options: local, redis, timescaledb
  local:
    base_path: "data/"
  redis:
    enabled: false
    host: "localhost"
    port: 6379
    db: 0
  timescaledb:
    enabled: false
    connection_string: "postgresql://postgres:postgres@localhost:5432/stockradar"

data_types:
  market_data:
    path: "market_data/"
    format: "parquet"
    retention:
      days: 90
      count: 100
  backtest_results:
    path: "backtest_results/"
    format: "parquet"
    retention:
      days: 365
      count: 50
```

## Extending the Storage System

To add a new storage backend:

1. Create a new class that inherits from `DataStorageBase`
2. Implement all the required methods (see `base.py`)
3. Update the `StorageManager._initialize_backends()` method to include the new backend

## Note on Consolidation

This module consolidates functionality previously found in:
- `backend/core/storage_integration.py` 
- `backend/storage_integration/storage_manager.py`

All code should now use this unified implementation instead. 