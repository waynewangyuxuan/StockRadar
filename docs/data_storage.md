# Data Storage Module Documentation

## Overview

The data storage module provides a flexible and extensible system for storing and managing market data. It supports multiple storage backends and includes version control capabilities.

## Storage Backends

### 1. Local Storage (`LocalStorage`)

Local storage uses Parquet files for efficient storage and fast reading. It's organized in the following directory structure:

```
storage_root/
    dataset_name/
        data.parquet (main data file)
        versions/
            version_id.parquet (versioned data files)
```

#### Usage
```python
from data_storage import LocalStorage

# Initialize storage
storage = LocalStorage({
    'storage_path': '/path/to/storage'
})

# Save data
storage.save_data(
    data=df,
    dataset_name='market_data',
    version='v1'  # Optional
)

# Load data
df = storage.load_data(
    dataset_name='market_data',
    tickers=['AAPL', 'MSFT'],  # Optional
    start_date='2024-01-01',   # Optional
    end_date='2024-02-01',     # Optional
    version='v1'               # Optional
)
```

### 2. Redis Cache (`RedisCache`)

Redis cache provides fast in-memory access to frequently used market data. It uses the following key patterns:

- `dataset:{name}` - Main data
- `dataset:{name}:version:{version}` - Versioned data
- `datasets` - Set of dataset names
- `versions:{dataset_name}` - Set of version names

#### Usage
```python
from data_storage import RedisCache

# Initialize cache
cache = RedisCache({
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'password': None,
    'key_prefix': 'stockradar:'
})

# Save data
cache.save_data(
    data=df,
    dataset_name='market_data',
    version='v1'  # Optional
)

# Load data
df = cache.load_data(
    dataset_name='market_data',
    tickers=['AAPL', 'MSFT'],  # Optional
    start_date='2024-01-01',   # Optional
    end_date='2024-02-01',     # Optional
    version='v1'               # Optional
)
```

### 3. TimescaleDB Storage (`TimescaleDBStorage`)

TimescaleDB storage is optimized for time-series market data. It uses the following schema:

```sql
-- Datasets table
CREATE TABLE datasets (
    name VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Dataset versions table
CREATE TABLE dataset_versions (
    dataset_name VARCHAR(255) REFERENCES datasets(name),
    version VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (dataset_name, version)
);

-- Market data table (hypertable)
CREATE TABLE market_data (
    dataset_name VARCHAR(255),
    version VARCHAR(255),
    ticker VARCHAR(10),
    date TIMESTAMPTZ,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    PRIMARY KEY (dataset_name, version, ticker, date)
);
```

#### Usage
```python
from data_storage import TimescaleDBStorage

# Initialize storage
storage = TimescaleDBStorage({
    'host': 'localhost',
    'port': 5432,
    'database': 'stockradar',
    'user': 'postgres',
    'password': 'postgres',
    'schema': 'public'
})

# Save data
storage.save_data(
    data=df,
    dataset_name='market_data',
    version='v1'  # Optional
)

# Load data
df = storage.load_data(
    dataset_name='market_data',
    tickers=['AAPL', 'MSFT'],  # Optional
    start_date='2024-01-01',   # Optional
    end_date='2024-02-01',     # Optional
    version='v1'               # Optional
)
```

### 4. Version Control (`VersionControl`)

The version control system manages data snapshots and tracks changes. It provides:

- Version creation with metadata
- Version retrieval
- Version comparison
- Version deletion

#### Usage
```python
from data_storage import VersionControl, LocalStorage

# Initialize version control
storage = LocalStorage()
version_control = VersionControl(storage)

# Create a new version
version_id, success = version_control.create_version(
    data=df,
    dataset_name='market_data',
    metadata={
        'description': 'Initial version',
        'source': 'yahoo_finance',
        'update_frequency': 'daily'
    }
)

# Get a specific version
data, metadata = version_control.get_version(
    dataset_name='market_data',
    version_id=version_id,
    tickers=['AAPL', 'MSFT'],  # Optional
    start_date='2024-01-01',   # Optional
    end_date='2024-02-01'      # Optional
)

# Compare versions
diff = version_control.compare_versions(
    dataset_name='market_data',
    version1='v1',
    version2='v2'
)
```

## Data Validation

All storage implementations validate data before saving. The validation checks for:

1. Required columns:
   - ticker
   - date
   - open
   - high
   - low
   - close
   - volume

2. Data types:
   - ticker: string
   - date: datetime
   - open, high, low, close: numeric
   - volume: integer

3. Data constraints:
   - high >= low
   - high >= open
   - high >= close
   - low <= open
   - low <= close
   - volume >= 0

## Error Handling

All storage implementations include comprehensive error handling:

1. Connection errors
2. Data validation errors
3. File system errors
4. Database errors
5. Version control errors

Errors are logged and appropriate exceptions are raised with descriptive messages.

## Best Practices

1. **Data Versioning**
   - Create versions for significant data changes
   - Include descriptive metadata
   - Use semantic versioning when possible

2. **Storage Selection**
   - Use Redis for frequently accessed data
   - Use TimescaleDB for historical analysis
   - Use local storage for development and testing

3. **Data Management**
   - Regularly clean up old versions
   - Monitor storage usage
   - Implement data retention policies

4. **Performance**
   - Use appropriate indexes
   - Implement caching strategies
   - Monitor query performance

## Docker Setup

The storage services can be run using Docker Compose:

```bash
# Start services
docker compose up -d

# Verify services
docker compose ps

# Check Redis connectivity
docker compose exec redis redis-cli ping

# Check TimescaleDB connectivity
docker compose exec timescaledb psql -U postgres -d stockradar -c "SELECT version();"
```

## Environment Variables

Configure storage services using environment variables:

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# TimescaleDB Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=stockradar
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

## Testing

Run the storage module tests:

```bash
# Run all storage tests
pytest tests/test_data_storage.py

# Run specific storage backend tests
pytest tests/test_local_storage.py
pytest tests/test_redis_cache.py
pytest tests/test_timescaledb_storage.py
pytest tests/test_version_control.py
``` 