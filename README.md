# StockRadar Trading System

StockRadar is a flexible, modular trading system that supports both backtesting and live trading with customizable strategies and factors.

## Features

- **Multiple Trading Strategies**: Mean reversion, moving average crossover, momentum breakout, and golden cross
- **Flexible Data Sources**: Yahoo Finance integration with extensible provider framework
- **Storage Options**: Local filesystem, Redis cache, and TimescaleDB support
- **REST API**: Comprehensive API for strategy management, trading control, and data access
- **Backtesting Engine**: Complete performance metrics and visualization
- **Paper Trading Ready**: Framework prepared for live paper trading integration

## Installation

### Prerequisites
- Python 3.8+
- pip
- Docker (optional, for advanced storage backends)

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd StockRadar

# Install dependencies
pip install -r backend/requirements.txt

# Create necessary directories
mkdir -p data config

# Copy sample configuration
cp backend/config/trading_config.yaml config/
```

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

# Run in CLI mode with backtesting
python main.py --mode cli --config backend/config/trading_config.yaml --trading-mode backtest

# Run in CLI mode with live trading (paper trading)
python main.py --mode cli --config backend/config/trading_config.yaml --trading-mode live
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

# Get details about a specific strategy
curl http://localhost:5000/api/strategies/mean_reversion

# Enable a strategy for trading
curl -X POST http://localhost:5000/api/strategies/mean_reversion/enable \
  -H "Content-Type: application/json" \
  -d '{"parameters": {"lookback_period": 20, "entry_threshold": 2.0}}'

# Get market data for specific symbols
curl "http://localhost:5000/api/data/market?symbols=AAPL,MSFT&start_date=2023-01-01&end_date=2023-12-31"

# Start backtesting with configuration
curl -X POST http://localhost:5000/api/trading/start \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "backtest",
    "config": {
      "general": {"mode": "backtest"},
      "data": {"symbols": ["AAPL", "MSFT"], "start_date": "2023-01-01", "end_date": "2023-12-31"},
      "portfolio": {"initial_capital": 100000},
      "strategies": [{"name": "mean_reversion", "enabled": true}]
    }
  }'

# Get the current trading status
curl http://localhost:5000/api/trading/status

# Stop trading session
curl -X POST http://localhost:5000/api/trading/stop
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

## Available Strategies

- **Mean Reversion**: Trades based on price deviations from moving averages
- **Moving Average Crossover**: Signals based on short/long MA crossovers
- **Momentum Breakout**: Detects breakouts above resistance levels
- **Golden Cross**: Long-term trend following using 50/200 day MA cross

## Dependencies

See `backend/requirements.txt` for complete list. Key dependencies:
- **Data Processing**: pandas, numpy, yfinance, ta
- **Web API**: Flask, flask-cors
- **Storage**: redis, psycopg2-binary, pyarrow
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest, pytest-cov

## Project Structure

```
StockRadar/
├── backend/                    # Core backend code
│   ├── api.py                 # REST API endpoints
│   ├── app.py                 # Application launcher
│   ├── run.py                 # CLI trading runner
│   ├── core/                  # Core framework
│   ├── strategy_engine/       # Strategy management
│   ├── data_fetcher/          # Data providers
│   ├── data_processor/        # Data processing
│   ├── data_storage/          # Storage backends
│   ├── backtester/           # Backtesting engine
│   ├── plugins/              # Strategy & factor plugins
│   └── config/               # Configuration files
├── main.py                   # Main entry point
├── start_api.sh             # Startup script
└── config/                  # User configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `cd backend && ./run_tests.sh`
5. Submit a pull request

## License

MIT License 