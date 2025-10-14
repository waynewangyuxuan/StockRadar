# Immediate Next Steps

1. **Fix Configuration Drift**
   - Align `configs/config.yaml` with actual class names (`MovingAverageFactor`) and ensure factors emit the columns each strategy requires.
   - Integrate `core/config.ConfigLoader` into the execution path to validate configs before runtime.

2. **Stabilise Versioning & Storage**
   - Refactor `data_storage/version_control.py` to decouple metadata from filenames and update Redis/Timescale adapters to accept canonical version IDs.
   - Add unit tests covering round-trips across all storage backends with the new scheme.

3. **Orchestrator Rewrite**
   - Replace `StrategyRunner` with an application service that uses typed configs, explicit services for market data, factors, and strategies, and produces consistent outputs.
   - Add an end-to-end simulator-driven test to verify the pipeline yields signals.

4. **Logging & Observability Hardening**
   - Build a logger factory to avoid duplicate handlers and honour config-driven log settings.
   - Emit execution metrics (duration, records processed, signal count) for future monitoring hooks.

5. **Testing Hygiene**
   - Seed RNGs in tests, remove reliance on unavailable infrastructure, and introduce fakes/mocks where appropriate.
   - Expand coverage for strategy alignment logic to catch index-related regressions.
