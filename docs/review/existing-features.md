# Existing Feature Inventory

## Data Ingestion
- `data_fetcher/base.py` defines a validated provider contract with symbol/date/interval checks and standardized DataFrame output.
- `data_fetcher/yfinance_provider.py` brings a Yahoo Finance connector with retry logic and metadata enrichment.
- `data_fetcher/simulator_provider.py` supplies deterministic synthetic data for repeatable tests and demos.

## Data Processing
- `data_processor/processor.py` performs validation, gap filling, and enriches OHLCV streams with metrics such as returns, ATR, VWAP, momentum, and relative volume.
- `data_processor/base.py` exposes guard rails (required column list, factor validation, date sanity checks) for alternative processors.

## Factor & Signal Plugins
- `core/factor_base.py` and `plugins/factors/` provide a plugin system for factor computation (moving averages, volume spikes) with support for batch execution.
- `core/strategy_base.py` and `plugins/strategies/` expose reusable strategy contracts plus example implementations (Golden Cross, Mean Reversion, Momentum Breakout).

## Storage Backends
- `data_storage/local_storage.py` persists datasets and versions as Parquet files.
- `data_storage/redis_cache.py` offers in-memory caching with Redis key conventions.
- `data_storage/timescaledb_storage.py` adapts TimescaleDB hypertables for time-series persistence.
- `data_storage/version_control.py` layers version metadata and comparison utilities over any `DataStorageBase` implementation.

## Configuration & Tooling
- `core/config.py` models configuration sections with dataclasses and validators, defining supported providers, factors, strategies, outputs, caching, and monitoring.
- `tests/` covers the major subsystems (fetchers, processor, factors, strategies, storage) to illustrate intended behaviour.
