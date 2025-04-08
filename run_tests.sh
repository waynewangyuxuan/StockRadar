#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting test setup...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Start Docker containers if not running
echo "Starting Docker containers..."
docker compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Create test database and schema
echo "Setting up test database..."
docker compose exec timescaledb psql -U postgres -c "DROP DATABASE IF EXISTS stockradar_test;"
docker compose exec timescaledb psql -U postgres -c "CREATE DATABASE stockradar_test;"

# Enable TimescaleDB extension
echo "Enabling TimescaleDB extension..."
docker compose exec timescaledb psql -U postgres -d stockradar_test -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Create schema and set search path
echo "Setting up test schema..."
docker compose exec timescaledb psql -U postgres -d stockradar_test -c "CREATE SCHEMA IF NOT EXISTS test;"
docker compose exec timescaledb psql -U postgres -d stockradar_test -c "SET search_path TO test, public;"

# Create tables in the test schema
echo "Creating tables..."
docker compose exec timescaledb psql -U postgres -d stockradar_test -c "
    CREATE TABLE IF NOT EXISTS test.datasets (
        name TEXT PRIMARY KEY,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS test.dataset_versions (
        dataset_name TEXT REFERENCES test.datasets(name),
        version TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (dataset_name, version)
    );

    CREATE TABLE IF NOT EXISTS test.market_data (
        dataset_name TEXT,
        version TEXT,
        ticker TEXT,
        date TIMESTAMPTZ,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume BIGINT,
        PRIMARY KEY (dataset_name, version, ticker, date)
    );

    SELECT create_hypertable('test.market_data', 'date',
        if_not_exists => TRUE,
        migrate_data => TRUE
    );
"

# Run tests
echo -e "${GREEN}Running tests...${NC}"
pytest tests/test_data_storage.py::TestRedisCache tests/test_data_storage.py::TestTimescaleDBStorage -v

# Cleanup
echo "Cleaning up..."
docker compose exec timescaledb psql -U postgres -d stockradar_test -c "DROP SCHEMA IF EXISTS test CASCADE;"
docker compose exec timescaledb psql -U postgres -c "DROP DATABASE IF EXISTS stockradar_test;"

echo -e "${GREEN}Test setup and execution completed.${NC}" 