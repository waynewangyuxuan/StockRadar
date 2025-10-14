"""
Tests for the base strategy engine.

This module contains tests for the core functionality of the strategy engine,
including strategy evaluation, metrics calculation, and result generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy_engine.base import StrategyEngine, StrategyResult
from strategy_engine.schema import StrategyInterface

class MockStrategy(StrategyInterface):
    """Mock strategy for testing."""
    
    def __init__(self, signals=None):
        self.signals = signals
        self.parameters = {}
        
    def generate_signals(self, data):
        if self.signals is not None:
            return self.signals
        return pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
        
    def get_parameters(self):
        return self.parameters
        
    def set_parameters(self, parameters):
        self.parameters = parameters

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

@pytest.fixture
def sample_signals(sample_data):
    """Create sample signals for testing."""
    signals = pd.DataFrame(index=sample_data.index)
    signals['signal'] = 0
    
    # Create some trading signals
    signals.loc['2020-03-01':'2020-03-15', 'signal'] = 1  # Long position
    signals.loc['2020-06-01':'2020-06-15', 'signal'] = -1  # Short position
    
    return signals

@pytest.fixture
def engine():
    """Create a strategy engine instance for testing."""
    return StrategyEngine(initial_capital=100000.0)

@pytest.fixture
def strategy(sample_signals):
    """Create a mock strategy instance for testing."""
    return MockStrategy(signals=sample_signals)

def test_engine_initialization():
    """Test strategy engine initialization."""
    engine = StrategyEngine()
    assert engine.initial_capital == 100000.0
    
    engine = StrategyEngine(initial_capital=50000.0)
    assert engine.initial_capital == 50000.0

def test_evaluate_strategy_empty_data(engine, strategy):
    """Test strategy evaluation with empty data."""
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError, match="Empty data provided"):
        engine.evaluate_strategy(strategy, empty_data)

def test_evaluate_strategy_missing_columns(engine, strategy):
    """Test strategy evaluation with missing columns."""
    incomplete_data = pd.DataFrame({'open': [1, 2, 3]})
    with pytest.raises(KeyError, match="Missing required columns"):
        engine.evaluate_strategy(strategy, incomplete_data)

def test_evaluate_strategy(engine, strategy, sample_data):
    """Test strategy evaluation with valid data."""
    result = engine.evaluate_strategy(strategy, sample_data)
    
    # Check result type
    assert isinstance(result, StrategyResult)
    
    # Check result components
    assert isinstance(result.equity_curve, pd.DataFrame)
    assert isinstance(result.drawdown_curve, pd.DataFrame)
    assert isinstance(result.trade_summary, pd.DataFrame)
    assert isinstance(result.position_summary, pd.DataFrame)
    assert isinstance(result.metrics, dict)
    assert isinstance(result.signals, pd.DataFrame)
    assert isinstance(result.positions, pd.DataFrame)
    assert isinstance(result.portfolio_values, pd.DataFrame)
    
    # Check equity curve
    assert 'value' in result.equity_curve.columns
    assert result.equity_curve.index.equals(sample_data.index)
    
    # Check drawdown curve
    assert 'value' in result.drawdown_curve.columns
    assert result.drawdown_curve.index.equals(sample_data.index)
    
    # Check metrics
    expected_metrics = [
        'total_return', 'annualized_return', 'sharpe_ratio',
        'max_drawdown', 'win_rate', 'profit_factor'
    ]
    for metric in expected_metrics:
        assert metric in result.metrics
        assert isinstance(result.metrics[metric], (int, float))

def test_calculate_positions(engine, sample_signals, sample_data):
    """Test position calculation."""
    positions = engine._calculate_positions(sample_signals, sample_data)
    
    assert isinstance(positions, pd.DataFrame)
    assert 'position' in positions.columns
    assert positions.index.equals(sample_signals.index)
    
    # Check position values
    assert positions.loc['2020-03-01':'2020-03-15', 'position'].equals(pd.Series(1, index=positions.loc['2020-03-01':'2020-03-15'].index))
    assert positions.loc['2020-06-01':'2020-06-15', 'position'].equals(pd.Series(-1, index=positions.loc['2020-06-01':'2020-06-15'].index))
    assert positions.loc['2020-01-01':'2020-02-29', 'position'].equals(pd.Series(0, index=positions.loc['2020-01-01':'2020-02-29'].index))

def test_calculate_portfolio_values(engine, sample_signals, sample_data):
    """Test portfolio value calculation."""
    positions = engine._calculate_positions(sample_signals, sample_data)
    portfolio_values = engine._calculate_portfolio_values(positions, sample_data)
    
    assert isinstance(portfolio_values, pd.DataFrame)
    assert 'value' in portfolio_values.columns
    assert portfolio_values.index.equals(sample_data.index)
    
    # Check initial value
    assert portfolio_values.iloc[0]['value'] == engine.initial_capital
    
    # Check that values change with positions
    assert portfolio_values.loc['2020-03-01':'2020-03-15', 'value'].std() > 0
    assert portfolio_values.loc['2020-06-01':'2020-06-15', 'value'].std() > 0

def test_calculate_equity_curve(engine, sample_signals, sample_data):
    """Test equity curve calculation."""
    positions = engine._calculate_positions(sample_signals, sample_data)
    portfolio_values = engine._calculate_portfolio_values(positions, sample_data)
    equity_curve = engine._calculate_equity_curve(portfolio_values)
    
    assert isinstance(equity_curve, pd.DataFrame)
    assert 'value' in equity_curve.columns
    assert equity_curve.index.equals(portfolio_values.index)
    assert equity_curve['value'].equals(portfolio_values['value'])

def test_calculate_drawdowns(engine, sample_signals, sample_data):
    """Test drawdown calculation."""
    positions = engine._calculate_positions(sample_signals, sample_data)
    portfolio_values = engine._calculate_portfolio_values(positions, sample_data)
    equity_curve = engine._calculate_equity_curve(portfolio_values)
    drawdowns = engine._calculate_drawdowns(equity_curve)
    
    assert isinstance(drawdowns, pd.DataFrame)
    assert 'value' in drawdowns.columns
    assert drawdowns.index.equals(equity_curve.index)
    
    # Check drawdown values
    assert drawdowns['value'].min() <= 0  # Should have at least one drawdown
    assert drawdowns['value'].max() == 0  # Maximum drawdown should be 0

def test_extract_trades(engine, sample_signals, sample_data):
    """Test trade extraction."""
    positions = engine._calculate_positions(sample_signals, sample_data)
    trades = engine._extract_trades(positions, sample_data)
    
    assert isinstance(trades, list)
    
    # Should have at least 4 trades (2 entries and 2 exits)
    assert len(trades) >= 4
    
    # Check trade structure
    for trade in trades:
        assert 'entry_date' in trade
        assert 'entry_price' in trade
        assert 'position' in trade
        assert 'size' in trade
        assert trade['position'] in [-1, 1]
        assert trade['size'] == 1

def test_calculate_metrics(engine, sample_signals, sample_data):
    """Test metrics calculation."""
    positions = engine._calculate_positions(sample_signals, sample_data)
    portfolio_values = engine._calculate_portfolio_values(positions, sample_data)
    equity_curve = engine._calculate_equity_curve(portfolio_values)
    drawdown_curve = engine._calculate_drawdowns(equity_curve)
    trades = engine._extract_trades(positions, sample_data)
    
    metrics = engine._calculate_metrics(equity_curve, drawdown_curve, trades)
    
    assert isinstance(metrics, dict)
    expected_metrics = [
        'total_return', 'annualized_return', 'sharpe_ratio',
        'max_drawdown', 'win_rate', 'profit_factor'
    ]
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))

def test_generate_trade_summary(engine, sample_signals, sample_data):
    """Test trade summary generation."""
    positions = engine._calculate_positions(sample_signals, sample_data)
    trades = engine._extract_trades(positions, sample_data)
    
    # Add exit information to trades
    for i, trade in enumerate(trades):
        if i % 2 == 0 and i + 1 < len(trades):
            trade['exit_date'] = trades[i+1]['entry_date']
            trade['exit_price'] = sample_data.loc[trade['exit_date'], 'close']
            trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['position']
    
    summary = engine._generate_trade_summary(trades)
    
    assert isinstance(summary, pd.DataFrame)
    if not summary.empty:
        assert 'duration' in summary.columns
        assert 'entry_date' in summary.columns
        assert 'exit_date' in summary.columns
        assert 'pnl' in summary.columns

def test_generate_position_summary(engine, sample_signals, sample_data):
    """Test position summary generation."""
    positions = engine._calculate_positions(sample_signals, sample_data)
    summary = engine._generate_position_summary(positions)
    
    assert isinstance(summary, pd.DataFrame)
    assert 'position' in summary.columns
    assert 'position_value' in summary.columns
    assert summary.index.equals(positions.index) 