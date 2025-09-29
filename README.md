# StockRadar Trading System

StockRadar is a flexible, modular trading system that supports both backtesting and live trading with customizable strategies and factors.

## Todo
1. Streamline the process to connect with the the storage system.
2. Test more about the pipeline.
3. Frontend Implementation.

## Quick Start

### Using the Startup Script

```bash
# Start the API server with default settings (port 5000)
./start_api.sh

# Start with a specific port
./start_api.sh --port=8080

# Start in CLI mode for backtesting
./start_api.sh --mode=cli --config=backend/config/api_config.yaml --trading-mode=backtest

# Start in CLI mode for live trading
./start_api.sh --mode=cli --config=backend/config/api_config.yaml --trading-mode=live
```

### Manual Startup

```bash
# Run in API mode (default)
python main.py

# Run in CLI mode with live trading
python main.py --mode cli --config config/trading_config.yaml --trading-mode live

# Run in CLI mode with backtesting
python main.py --mode cli --config config/backtest_config.yaml --trading-mode backtest
```

## API Endpoints

StockRadar provides a RESTful API for interacting with the trading system. The API runs on port 5000 by default.

### Strategy Management
- `GET /api/strategies` - List all available strategies
- `GET /api/strategies/{id}` - Get details about a specific strategy
- `POST /api/strategies/{id}/enable` - Enable a strategy for trading
- `POST /api/strategies/{id}/disable` - Disable a strategy

### Trading Control
- `POST /api/trading/start` - Start trading with specified mode and configuration
- `POST /api/trading/stop` - Stop the current trading session
- `GET /api/trading/status` - Get the current status of the trading system

### Configuration Management
- `GET /api/config` - Get the current configuration
- `PUT /api/config` - Update the configuration

### Data Management
- `GET /api/data/market?symbols=AAPL,MSFT&start_date=2023-01-01&end_date=2023-12-31` - Get market data for specified symbols

### Portfolio Management
- `GET /api/portfolio` - Get the current portfolio status

## Example API Usage

Here are some examples of how to interact with the API using curl:

```bash
# List all available strategies
curl http://localhost:5000/api/strategies

# Get market data for specific symbols
curl http://localhost:5000/api/data/market?symbols=AAPL,MSFT

# Start backtesting with the provided configuration
curl -X POST http://localhost:5000/api/trading/start \
  -H "Content-Type: application/json" \
  -d '{"mode": "backtest", "config_path": "backend/config/api_config.yaml"}'

# Get the current trading status
curl http://localhost:5000/api/trading/status
```

## System Architecture

The system is organized into two main components:
1. **Backend** - Core trading and data processing logic
2. **API** - RESTful interface for interacting with the system

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

### Live Trading Process

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
- Flask (for API mode)

## License

MIT License 