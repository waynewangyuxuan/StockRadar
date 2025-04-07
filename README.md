# StockRadar

A stock market data analysis and storage system with support for multiple storage backends.

## Features

- Multiple storage backends:
  - Local file storage (Parquet format)
  - Redis cache for fast access
  - TimescaleDB for time-series data
- Data versioning and snapshot management
- Comprehensive test suite
- Docker-based deployment

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Git

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stockradar.git
cd stockradar
```

2. Start the services using Docker Compose:
```bash
docker compose up -d
```

This will start:
- Redis on port 6379
- TimescaleDB on port 5432
- The StockRadar application

3. Verify the services are running:
```bash
docker compose ps
```

## Development Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Run the tests:
```bash
pytest
```

## Project Structure

```
stockradar/
├── data_storage/          # Storage implementations
│   ├── base.py           # Base storage class
│   ├── local_storage.py  # Local file storage
│   ├── redis_cache.py    # Redis cache storage
│   ├── timescaledb_storage.py  # TimescaleDB storage
│   └── version_control.py # Version control system
├── tests/                # Test suite
├── docker-compose.yml    # Docker services configuration
├── Dockerfile           # Application container
├── requirements.txt     # Python dependencies
└── setup.py            # Package configuration
```

## Storage Backends

### Local Storage
- Uses Parquet format for efficient storage
- Supports data versioning
- Suitable for development and testing

### Redis Cache
- Fast in-memory storage
- Supports data versioning
- Ideal for frequently accessed data

### TimescaleDB
- Time-series optimized storage
- Supports data versioning
- Best for historical data analysis

## Data Versioning

The system supports data versioning through the `VersionControl` class:
```python
from data_storage import VersionControl, LocalStorage

# Initialize version control
storage = LocalStorage()
version_control = VersionControl(storage)

# Create a new version
version_id, success = version_control.create_version(
    data,
    'dataset_name',
    {'description': 'Initial version'}
)

# Get a specific version
data, metadata = version_control.get_version('dataset_name', version_id)
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data_storage.py

# Run with coverage report
pytest --cov=stockradar
```

## Docker Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f

# Rebuild containers
docker compose up -d --build

# Access Redis CLI
docker compose exec redis redis-cli

# Access TimescaleDB
docker compose exec timescaledb psql -U postgres -d stockradar
```

## Environment Variables

The following environment variables can be configured:

- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `POSTGRES_HOST`: TimescaleDB host (default: localhost)
- `POSTGRES_PORT`: TimescaleDB port (default: 5432)
- `POSTGRES_DB`: Database name (default: stockradar)
- `POSTGRES_USER`: Database user (default: postgres)
- `POSTGRES_PASSWORD`: Database password (default: postgres)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 