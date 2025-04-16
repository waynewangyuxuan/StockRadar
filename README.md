# StockRadar Trading System

StockRadar is a flexible, modular trading system that supports both backtesting and live trading with customizable strategies and factors.

## Todo
1. Streamline the process to connect with the the storage system.
2. Test more about the pipeline.
2. Frontened Implementation.

## Quick Start

```bash
# Run live trading
python run.py --mode live --config config/trading_config.yaml

# Run backtesting
python run.py --mode backtest --config config/backtest_config.yaml
```

## System Architecture

The system is orchestrated through `run.py`, which serves as the main entry point. Here's how it works:

### Core Components

1. **Trading Runner**: The central orchestrator that manages:
   - Market data fetching and processing
   - Strategy execution
   - Portfolio management
   - Risk management
   - Trade execution

2. **Data Pipeline**:
   - Data Providers (e.g., Yahoo Finance)
   - Data Processors for technical indicators
   - Factor calculation engine

3. **Strategy Framework**:
   - Modular strategy implementation
   - Signal generation
   - Factor-based analysis

### Implementation Logic

The `run.py` script follows this sequence:

1. **Configuration Loading**:
   ```python
   # Load and validate configuration
   config = load_config(config_path)
   ```

2. **Component Initialization**:
   ```python
   # Initialize core components
   data_provider = YahooFinanceProvider(config)
   data_processor = DataProcessor(config)
   strategy_registry = StrategyRegistry()
   factor_registry = FactorRegistry()
   ```

3. **Trading Runner Setup**:
   ```python
   # Create and configure trading runner
   runner = TradingRunner(
       config=config,
       strategy_registry=strategy_registry,
       factor_registry=factor_registry,
       data_provider=data_provider,
       data_processor=data_processor
   )
   ```

4. **Execution**:
   ```python
   # Run in specified mode
   if mode == "live":
       runner.run_live()
   else:
       results = runner.run_backtest()
   ```

### Live Trading Processtr

1. **Data Fetching**: Continuously fetches market data at configured intervals
2. **Processing**: Applies technical indicators and calculates factors
3. **Signal Generation**: Strategies analyze data and generate trading signals
4. **Trade Execution**: Executes trades based on signals and risk parameters
5. **Portfolio Updates**: Tracks positions, P&L, and risk metrics

### Configuration

The system is configured through YAML files:

```yaml
data:
  interval: "1m"  # Data update interval
  universe_type: "custom"
  custom_universe: ["AAPL", "GOOGL", "MSFT"]

portfolio:
  initial_capital: 100000
  commission: 0.001

risk:
  max_position_size: 10000
  stop_loss: 0.02
  take_profit: 0.05
```

## Adding Custom Components

### Custom Strategy
```python
class MyStrategy(StrategyBase):
    def generate_signals(self, market_data, factor_data):
        # Implement signal generation logic
        pass
```

### Custom Factor
```python
class MyFactor(FactorBase):
    def calculate(self, market_data):
        # Implement factor calculation
        pass
```

## Monitoring and Logging

The system provides detailed logging of:
- Trade execution
- Portfolio updates
- Factor calculations
- System status

Logs are written to both console and file for monitoring and debugging.

## Dependencies

- pandas
- numpy
- yfinance
- PyYAML
- ta (Technical Analysis library)

## License

MIT License 