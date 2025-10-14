# Refactor Roadmap

## Layered Architecture
1. **Infrastructure Layer** – encapsulate providers, storage adapters, logging, and configuration loading behind interfaces (`data_fetcher`, `data_storage`, logging factory, `core/config.py`).
2. **Domain Services** – introduce dedicated services for market data retrieval, factor computation, and strategy execution that accept typed configs instead of raw dicts.
3. **Application Layer** – replace `core/runner.py` with an orchestrator service that composes domain services, handles lifecycle management, and writes outputs.

## Configuration Unification
- Promote `core/config.ConfigLoader` (currently unused) to the canonical path and have the application layer request `StockRadarConfig` instances.
- Build a registry that maps config class paths to callable factories; validate all dependencies (factor outputs, strategy requirements) before execution.

## Factor & Strategy Improvements
- Implement a factor registry capable of declaring outputs, dependencies, and pre/post-processing hooks.
- Add factor pipelines that operate on MultiIndex data to match strategy expectations.
- Ensure strategies operate on aligned indices and avoid `.values`; prefer explicit reindexing and DataFrame joins.

## Storage & Versioning
- Redesign `VersionControl` to persist metadata separately (JSON sidecar file, Redis hash, or TimescaleDB table) while keeping version identifiers filesystem/DB safe.
- Normalise version naming across backends and expose a lightweight repository interface for version CRUD.

## Logging & Observability
- Centralise logger configuration so multiple runs do not re-attach handlers; allow config-driven sinks and levels.
- Expose structured execution metrics (duration, dataset sizes, signal counts) through a monitoring service that can back Prometheus or simple CSV summaries.

## Testing & Tooling
- Provide fakes for Redis and TimescaleDB or docker-compose based fixtures that can be skipped locally.
- Seed random number generators in fixtures to make regression runs deterministic.
- Add end-to-end simulator-based tests that exercise the full pipeline once the orchestrator is rewritten.
