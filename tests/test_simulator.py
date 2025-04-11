"""
Tests for the backtester simulator module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtester.simulator import BacktestSimulator, Order, OrderType, Position

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    # Create multi-index DataFrame
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    # Generate sample price data
    np.random.seed(42)  # For reproducibility
    n_records = len(dates) * len(tickers)
    
    data = {
        'open': np.random.normal(100, 10, n_records),
        'high': np.random.normal(105, 10, n_records),
        'low': np.random.normal(95, 10, n_records),
        'close': np.random.normal(100, 10, n_records),
        'volume': np.random.randint(1000, 10000, n_records)
    }
    
    # Ensure high > open/close > low
    df = pd.DataFrame(data, index=index)
    df['high'] = df[['open', 'high', 'close']].max(axis=1) + 1
    df['low'] = df[['open', 'low', 'close']].min(axis=1) - 1
    
    return df

@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing."""
    class MockStrategy:
        def generate_signals(self, market_data):
            signals = pd.DataFrame(
                index=market_data.index,
                columns=['signal', 'signal_strength']
            )
            # Generate some sample signals
            signals['signal'] = np.random.choice([-1, 0, 1], size=len(market_data))
            signals['signal_strength'] = np.random.uniform(0.1, 1.0, size=len(market_data))
            return signals
    
    return MockStrategy()

class TestBacktestSimulator:
    """Test suite for BacktestSimulator class."""
    
    def test_initialization(self):
        """Test simulator initialization with default and custom parameters."""
        # Test default parameters
        sim = BacktestSimulator()
        assert sim.initial_capital == 100000.0
        assert sim.transaction_cost == 0.001
        assert sim.slippage == 0.0001
        
        # Test custom parameters
        sim = BacktestSimulator(
            initial_capital=200000.0,
            transaction_cost=0.002,
            slippage=0.0002
        )
        assert sim.initial_capital == 200000.0
        assert sim.transaction_cost == 0.002
        assert sim.slippage == 0.0002
        
        # Test initial state
        assert sim.positions == {}
        assert sim.orders == []
        assert sim.trades == []
        assert sim.portfolio_value == []
        assert sim.timestamps == []
    
    def test_reset(self):
        """Test simulator reset functionality."""
        sim = BacktestSimulator(initial_capital=100000.0)
        
        # Modify state
        sim.current_capital = 90000.0
        sim.positions['AAPL'] = Position(
            ticker='AAPL',
            quantity=100,
            average_price=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,
            realized_pnl=0.0
        )
        sim.orders.append(Order(
            ticker='AAPL',
            order_type=OrderType.MARKET,
            side=1,
            quantity=100
        ))
        sim.portfolio_value.append(90000.0)
        sim.timestamps.append(datetime.now())
        
        # Reset
        sim._reset()
        
        # Verify reset state
        assert sim.current_capital == sim.initial_capital
        assert sim.positions == {}
        assert sim.orders == []
        assert sim.trades == []
        assert sim.portfolio_value == []
        assert sim.timestamps == []
    
    def test_order_execution(self):
        """Test order execution with slippage and transaction costs."""
        sim = BacktestSimulator(
            initial_capital=100000.0,
            transaction_cost=0.001,
            slippage=0.001
        )
        
        # Create and execute a buy order
        order = Order(
            ticker='AAPL',
            order_type=OrderType.MARKET,
            side=1,
            quantity=100,
            timestamp=datetime.now()
        )
        current_price = 150.0
        
        sim._execute_order(order, current_price)
        
        # Calculate expected values
        execution_price = current_price * (1 + sim.slippage)  # Buy slippage
        trade_value = execution_price * order.quantity
        transaction_cost = trade_value * sim.transaction_cost
        expected_capital = sim.initial_capital - (trade_value + transaction_cost)
        
        # Verify execution results
        assert sim.current_capital == pytest.approx(expected_capital)
        assert 'AAPL' in sim.positions
        assert sim.positions['AAPL'].quantity == 100
        assert sim.positions['AAPL'].average_price == pytest.approx(execution_price)
    
    def test_position_update(self):
        """Test position updates with market data."""
        sim = BacktestSimulator()
        
        # Create initial position
        sim.positions['AAPL'] = Position(
            ticker='AAPL',
            quantity=100,
            average_price=150.0,
            current_price=150.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
        
        # Create market data
        market_data = pd.DataFrame({
            'close': {'AAPL': 160.0}
        })
        
        # Update positions
        sim._update_positions(market_data)
        
        # Verify position update
        position = sim.positions['AAPL']
        assert position.current_price == 160.0
        assert position.unrealized_pnl == pytest.approx(1000.0)  # (160 - 150) * 100
    
    def test_portfolio_state_recording(self):
        """Test recording of portfolio state."""
        sim = BacktestSimulator(initial_capital=100000.0)
        
        # Create a position
        sim.positions['AAPL'] = Position(
            ticker='AAPL',
            quantity=100,
            average_price=150.0,
            current_price=160.0,
            unrealized_pnl=1000.0,
            realized_pnl=0.0
        )
        
        # Record state
        timestamp = datetime(2023, 1, 1)
        sim._record_portfolio_state(timestamp)
        
        # Verify recorded state
        assert len(sim.portfolio_value) == 1
        assert len(sim.timestamps) == 1
        assert sim.timestamps[0] == timestamp
        assert sim.portfolio_value[0] == pytest.approx(101000.0)  # Initial capital + unrealized PnL
    
    def test_full_backtest_run(self, sample_market_data, mock_strategy):
        """Test full backtest simulation run."""
        sim = BacktestSimulator(
            initial_capital=100000.0,
            transaction_cost=0.001,
            slippage=0.001
        )
        
        # Run backtest
        results = sim.run(
            strategy=mock_strategy,
            market_data=sample_market_data,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        # Verify results structure
        assert 'portfolio_value' in results
        assert 'trades' in results
        assert 'positions' in results
        assert 'metrics' in results
        
        # Verify portfolio value history
        assert len(results['portfolio_value']) == len(sample_market_data.index.get_level_values('date').unique())
        assert results['portfolio_value'][0] == pytest.approx(sim.initial_capital)
    
    def test_multiple_orders(self):
        """Test handling of multiple orders for the same ticker."""
        sim = BacktestSimulator()
        
        # Execute multiple buy orders
        orders = [
            Order(ticker='AAPL', order_type=OrderType.MARKET, side=1, quantity=50),
            Order(ticker='AAPL', order_type=OrderType.MARKET, side=1, quantity=50),
            Order(ticker='AAPL', order_type=OrderType.MARKET, side=-1, quantity=30)
        ]
        
        current_price = 150.0
        for order in orders:
            sim._execute_order(order, current_price)
        
        # Verify final position
        assert 'AAPL' in sim.positions
        assert sim.positions['AAPL'].quantity == 70  # 50 + 50 - 30
    
    def test_position_closing(self):
        """Test complete position closure."""
        sim = BacktestSimulator()
        
        # Open position
        buy_order = Order(ticker='AAPL', order_type=OrderType.MARKET, side=1, quantity=100)
        sim._execute_order(buy_order, 150.0)
        
        # Close position
        sell_order = Order(ticker='AAPL', order_type=OrderType.MARKET, side=-1, quantity=100)
        sim._execute_order(sell_order, 160.0)
        
        # Verify position is closed
        assert 'AAPL' not in sim.positions
    
    def test_invalid_order_handling(self):
        """Test handling of invalid orders."""
        sim = BacktestSimulator()
        
        # Test order with zero quantity
        order = Order(ticker='AAPL', order_type=OrderType.MARKET, side=1, quantity=0)
        with pytest.raises(ValueError):
            sim._execute_order(order, 150.0)
        
        # Test order with insufficient capital
        order = Order(ticker='AAPL', order_type=OrderType.MARKET, side=1, quantity=1000000)
        with pytest.raises(ValueError):
            sim._execute_order(order, 150.0) 