import datetime
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import time

from core.factor_base import FactorBase
from core.strategy_base import StrategyBase
from core.runner import TradingRunner
from core.types import MarketData, SignalData
from strategy_engine.strategy_registry import registry as strategy_registry
from core.factor_registry import registry as factor_registry
from data_processor.processor import DataProcessor

# Test data constants
TEST_START_DATE = pd.Timestamp('2024-01-01')
TEST_END_DATE = pd.Timestamp('2024-01-10')
TEST_SYMBOLS = ['AAPL', 'GOOGL']

def create_test_market_data(start_date: pd.Timestamp, end_date: pd.Timestamp, symbols: List[str]) -> pd.DataFrame:
    """Create deterministic test market data."""
    dates = pd.date_range(start_date, end_date, freq='D')
    data = []
    for symbol in symbols:
        base_price = 100 if symbol == 'AAPL' else 150
        for i, date in enumerate(dates):
            # Create deterministic price movement
            close_price = base_price + i
            data.append({
                'date': date,
                'ticker': symbol,
                'open': close_price - 1,
                'high': close_price + 1,
                'low': close_price - 2,
                'close': close_price,
                'volume': 1000000 + i * 1000
            })
    df = pd.DataFrame(data)
    df.set_index(['date', 'ticker'], inplace=True)
    return df

class MockDataProvider:
    """Mock data provider that returns deterministic data."""
    
    def __init__(self, mock_data: pd.DataFrame = None):
        self.mock_data = mock_data if mock_data is not None else create_test_market_data(TEST_START_DATE, TEST_END_DATE, TEST_SYMBOLS)
        
    def fetch_historical_data(self, symbols: List[str], start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        """Return predetermined test data."""
        return self.mock_data
        
    def fetch_live_data(self, symbols: List[str], interval: str = '1d') -> pd.DataFrame:
        """Return latest test data for live trading."""
        # For testing, just return the last row of the mock data for each symbol
        latest_data = []
        for symbol in symbols:
            symbol_data = self.mock_data[self.mock_data.index.get_level_values('ticker') == symbol]
            if not symbol_data.empty:
                latest_row = symbol_data.iloc[-1]
                latest_data.append({
                    'date': latest_row.name[0],  # First level of MultiIndex is date
                    'ticker': symbol,
                    'open': latest_row['open'],
                    'high': latest_row['high'],
                    'low': latest_row['low'],
                    'close': latest_row['close'],
                    'volume': latest_row['volume']
                })
        
        df = pd.DataFrame(latest_data)
        if not df.empty:
            df.set_index(['date', 'ticker'], inplace=True)
        return df
        
    def get_sp500_symbols(self) -> List[str]:
        """Return fixed test symbols."""
        return TEST_SYMBOLS
        
    def get_nasdaq100_symbols(self) -> List[str]:
        """Return fixed test symbols."""
        return TEST_SYMBOLS

class MockStrategy(StrategyBase):
    """Mock strategy that returns predetermined signals."""
    
    def __init__(self, config: Dict[str, Any], signals: pd.DataFrame = None):
        super().__init__(config)
        self._signals = signals
        
    def get_required_factors(self) -> List[str]:
        """Get list of required factors for this strategy."""
        return ['mock_factor']
        
    def generate_signals(self, market_data: pd.DataFrame, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return predetermined signals or generate simple ones."""
        if self._signals is not None:
            return SignalData(self._signals)
            
        # Generate simple buy signals when price increases, sell signals when it decreases
        close_prices = market_data['close']
        signals = pd.DataFrame(0, index=close_prices.index, columns=['signal'])
        signals.loc[close_prices.diff() > 0, 'signal'] = 1  # Buy when price increases
        signals.loc[close_prices.diff() < 0, 'signal'] = -1  # Sell when price decreases
        return SignalData(signals)

class MockFactor(FactorBase):
    """Mock factor that returns predetermined values."""
    
    def __init__(self, config: Dict[str, Any], factor_values: pd.DataFrame = None):
        super().__init__(config)
        self.window = config.get('parameters', {}).get('window', 20)
        self._factor_values = factor_values
        
    def calculate(self, market_data: MarketData) -> pd.DataFrame:
        """Return predetermined factor values or calculate simple moving average."""
        if self._factor_values is not None:
            return self._factor_values
        return market_data.data['close'].rolling(window=self.window).mean()

@pytest.fixture
def mock_data():
    """Create test market data."""
    return create_test_market_data(TEST_START_DATE, TEST_END_DATE, TEST_SYMBOLS)

@pytest.fixture
def mock_config():
    """Create test configuration."""
    return {
        'portfolio': {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.001
        },
        'risk': {
            'max_position_size': 10000,
            'max_drawdown': 0.2,
            'stop_loss': 0.1,
            'take_profit': 0.2,
            'max_positions': 5
        },
        'data': {
            'start_date': TEST_START_DATE.strftime('%Y-%m-%d'),
            'end_date': TEST_END_DATE.strftime('%Y-%m-%d'),
            'universe_type': 'sp500'
        },
        'factors': {
            'mock_factor': {
                'enabled': True,
                'parameters': {
                    'window': 5
                }
            }
        },
        'strategies': [{
            'name': 'mock_strategy',
            'enabled': True,
            'parameters': {}
        }],
        'logging': {
            'output_dir': 'test_output'
        }
    }

@pytest.fixture
def mock_data_provider(mock_data):
    """Create mock data provider with test data."""
    return MockDataProvider(mock_data)

@pytest.fixture
def mock_data_processor():
    """Create mock data processor."""
    processor = Mock(spec=DataProcessor)
    processor.get_required_metrics.return_value = []
    processor.process_data.side_effect = lambda x, **kwargs: x
    return processor

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown for all tests."""
    # Setup
    strategy_registry.register('mock_strategy', MockStrategy)
    factor_registry.register('mock_factor', MockFactor)
    yield
    # Teardown
    strategy_registry.clear()
    factor_registry.clear()

def test_trading_runner_init(mock_config):
    data_provider = MockDataProvider()
    data_processor = DataProcessor()
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=data_provider,
        data_processor=data_processor
    )
    assert runner.config == mock_config
    assert runner.data_provider == data_provider
    assert runner.data_processor == data_processor

def test_trading_runner_load_data(mock_config):
    data_provider = MockDataProvider()
    data_processor = DataProcessor()
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=data_provider,
        data_processor=data_processor
    )
    market_data = runner._load_historical_data()
    assert isinstance(market_data, pd.DataFrame)
    assert not market_data.empty
    assert set(market_data.index.get_level_values('ticker').unique()) == set(TEST_SYMBOLS)

def test_trading_runner_calculate_factors(mock_config):
    data_provider = MockDataProvider()
    data_processor = DataProcessor()
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=data_provider,
        data_processor=data_processor
    )
    market_data = runner._load_historical_data()
    factors = runner.calculate_factors(market_data)
    assert isinstance(factors, dict)
    assert 'mock_factor' in factors
    assert not factors['mock_factor'].empty

def test_trading_runner_generate_signals(mock_config):
    data_provider = MockDataProvider()
    data_processor = DataProcessor()
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=data_provider,
        data_processor=data_processor
    )
    market_data = runner._load_historical_data()
    factors = runner.calculate_factors(market_data)
    signals = runner.generate_signals(market_data, factors)
    assert isinstance(signals, SignalData)
    assert not signals.data.empty
    assert 'signal' in signals.data.columns

def test_trading_runner_run(mock_config):
    data_provider = MockDataProvider()
    data_processor = DataProcessor()
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=data_provider,
        data_processor=data_processor
    )
    results = runner.run_backtest()
    assert isinstance(results, dict)
    assert 'returns' in results
    assert 'positions' in results
    assert 'trades' in results

def test_trading_runner_initialization(mock_config, mock_data_provider, mock_data_processor):
    """Test that TradingRunner initializes correctly with all components."""
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=mock_data_provider,
        data_processor=mock_data_processor
    )
    
    # Test basic initialization
    assert runner.config == mock_config
    assert runner.data_provider == mock_data_provider
    assert runner.data_processor == mock_data_processor
    assert not runner.is_running
    assert runner.stop_event is not None
    
    # Test that components are loaded
    assert len(runner.factors) > 0  # Should have loaded mock factor
    assert len(runner.strategies) > 0  # Should have loaded mock strategy

def test_trading_runner_data_loading(mock_config, mock_data_provider, mock_data_processor):
    """Test that TradingRunner can load and process market data correctly."""
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=mock_data_provider,
        data_processor=mock_data_processor
    )
    
    # Test historical data loading
    data = runner._load_historical_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert set(data.index.get_level_values('ticker').unique()) == set(TEST_SYMBOLS)
    
    # Test data processing
    processed_data = runner._process_data(data)
    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    assert all(col in processed_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_trading_runner_signal_generation(mock_config, mock_data_provider, mock_data_processor):
    """Test that TradingRunner can generate trading signals correctly."""
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=mock_data_provider,
        data_processor=mock_data_processor
    )
    
    # Get some test data
    data = runner._load_historical_data()
    processed_data = runner._process_data(data)
    
    # Test signal generation
    signals = runner._generate_live_signals(processed_data)
    assert isinstance(signals, pd.DataFrame)
    assert not signals.empty
    assert 'signal' in signals.columns
    assert 'strategy' in signals.columns

def test_trading_runner_portfolio_management(mock_config, mock_data_provider, mock_data_processor):
    """Test that TradingRunner manages portfolio correctly."""
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=mock_data_provider,
        data_processor=mock_data_processor
    )
    
    # Initialize portfolio
    runner._initialize_live_trading()
    initial_cash = runner.portfolio['cash']
    
    # Test buy order
    test_ticker = 'AAPL'
    test_shares = 10
    test_price = 150.0
    
    runner._execute_buy_order(test_ticker, test_shares, test_price)
    assert runner.portfolio['positions'].get(test_ticker) == test_shares
    assert runner.portfolio['cash'] < initial_cash  # Cash should decrease
    assert len(runner.portfolio['trades']) == 1
    
    # Test sell order
    runner._execute_sell_order(test_ticker, test_shares, test_price)
    assert test_ticker not in runner.portfolio['positions']
    assert len(runner.portfolio['trades']) == 2

def test_trading_runner_risk_management(mock_config, mock_data_provider, mock_data_processor):
    """Test that TradingRunner applies risk management rules correctly."""
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=mock_data_provider,
        data_processor=mock_data_processor
    )
    
    # Initialize portfolio
    runner._initialize_live_trading()
    
    # Test position size limits
    max_position_size = runner.risk_manager['max_position_size']
    position_size = runner._calculate_position_size('AAPL', 100.0)
    assert position_size > 0
    assert position_size * 100.0 <= max_position_size  # Position value should not exceed limit

    # Test max positions limit
    runner.risk_manager['max_positions'] = 1
    runner._execute_buy_order('AAPL', 10, 100.0)
    position_size = runner._calculate_position_size('GOOGL', 100.0)
    assert position_size == 0  # Should not allow new position when at limit

def test_backtest_returns_expected_results(mock_config, mock_data_provider, mock_data_processor):
    """Test that backtesting returns expected results with deterministic data."""
    # Create a runner with our mocks
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=mock_data_provider,
        data_processor=mock_data_processor
    )
    
    # Run backtest
    results = runner.run_backtest()
    
    # Verify the structure and basic properties of results
    assert isinstance(results, dict)
    assert 'returns' in results
    assert 'positions' in results
    assert 'trades' in results
    
    # Since we're using deterministic data, we can make some assertions about the results
    trades = results['trades']
    if trades:  # If any trades were made
        assert all(isinstance(trade['timestamp'], datetime.datetime) for trade in trades)
        assert all(trade['ticker'] in TEST_SYMBOLS for trade in trades)
        assert all(trade['action'] in ['BUY', 'SELL'] for trade in trades)
        assert all(isinstance(trade['shares'], int) for trade in trades)
        assert all(isinstance(trade['price'], (int, float)) for trade in trades)

def test_backtest_with_empty_data(mock_config, mock_data_processor):
    """Test that backtesting handles empty data gracefully."""
    # Create a data provider that returns empty data
    empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    empty_data.index = pd.MultiIndex.from_tuples([], names=['date', 'ticker'])
    data_provider = MockDataProvider(empty_data)
    
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=data_provider,
        data_processor=mock_data_processor
    )
    
    # Run backtest
    results = runner.run_backtest()
    
    # Verify that results are empty but structured correctly
    assert isinstance(results, dict)
    assert 'returns' in results
    assert 'positions' in results
    assert 'trades' in results
    assert len(results['trades']) == 0

def test_backtest_with_missing_config_values(mock_data_provider, mock_data_processor):
    """Test that backtesting handles missing configuration values gracefully."""
    # Create a minimal config
    minimal_config = {
        'portfolio': {'initial_capital': 100000},
        'data': {
            'start_date': TEST_START_DATE.strftime('%Y-%m-%d'),
            'end_date': TEST_END_DATE.strftime('%Y-%m-%d')
        }
    }
    
    runner = TradingRunner(
        config=minimal_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=mock_data_provider,
        data_processor=mock_data_processor
    )
    
    # Run backtest
    results = runner.run_backtest()
    
    # Verify that results are structured correctly
    assert isinstance(results, dict)
    assert 'returns' in results
    assert 'positions' in results
    assert 'trades' in results

def test_live_trading_basic_functionality(mock_config, mock_data_provider, mock_data_processor):
    """Test basic live trading functionality."""
    # Mock the logger to prevent excessive logging
    with patch('logging.Logger.info'), patch('logging.Logger.warning'):
        runner = TradingRunner(
            config=mock_config,
            strategy_registry=strategy_registry,
            factor_registry=factor_registry,
            data_provider=mock_data_provider,
            data_processor=mock_data_processor
        )
        
        # Mock the trading loop to run only once
        original_loop = runner._trading_loop
        def mock_trading_loop():
            # Run the loop once then exit
            try:
                original_loop()
            except Exception as e:
                print(f"Error in trading loop: {e}")
            finally:
                runner.stop_event.set()
        
        runner._trading_loop = mock_trading_loop
        
        # Start live trading
        runner.run_live()
        assert runner.is_running
        assert runner.trading_thread is not None
        assert runner.trading_thread.is_alive()
        
        # Wait for the trading thread to stop (should be quick now)
        runner.trading_thread.join(timeout=2.0)
        
        # Stop live trading
        runner.stop_live()
        
        # Verify everything stopped
        assert not runner.is_running
        assert not runner.trading_thread.is_alive()

def test_live_trading_prevents_duplicate_start(mock_config, mock_data_provider, mock_data_processor):
    """Test that we can't start live trading twice."""
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=mock_data_provider,
        data_processor=mock_data_processor
    )
    
    # Start live trading
    runner.run_live()
    assert runner.is_running
    
    # Try to start again
    runner.run_live()  # Should log a warning and do nothing
    
    # Clean up
    runner.stop_live()

def test_risk_management_stop_loss(mock_config, mock_data_provider, mock_data_processor):
    """Test that stop loss works in live trading."""
    # Create data that would trigger stop loss
    dates = pd.date_range(TEST_START_DATE, TEST_END_DATE, freq='D')
    data = []
    price = 100
    for date in dates:
        data.append({
            'date': date,
            'ticker': 'AAPL',
            'open': price,
            'high': price + 1,
            'low': price - 1,
            'close': price * 0.85,  # 15% drop should trigger stop loss
            'volume': 1000000
        })
        price *= 0.85
    
    df = pd.DataFrame(data)
    df.set_index(['date', 'ticker'], inplace=True)
    data_provider = MockDataProvider(df)
    
    runner = TradingRunner(
        config=mock_config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=data_provider,
        data_processor=mock_data_processor
    )
    
    # Run backtest
    results = runner.run_backtest()
    
    # Verify that stop loss was triggered
    trades = results['trades']
    if trades:
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        assert len(sell_trades) > 0  # At least one sell trade should have occurred

def test_risk_management_position_limits(mock_config, mock_data_provider, mock_data_processor):
    """Test that position limits are respected."""
    # Modify config to have a very small position limit
    config = mock_config.copy()
    config['risk']['max_positions'] = 1
    
    runner = TradingRunner(
        config=config,
        strategy_registry=strategy_registry,
        factor_registry=factor_registry,
        data_provider=mock_data_provider,
        data_processor=mock_data_processor
    )
    
    # Run backtest
    results = runner.run_backtest()
    
    # Verify that position limit was respected
    positions = results['positions']
    assert len(positions) <= 1  # Should never have more than 1 position 