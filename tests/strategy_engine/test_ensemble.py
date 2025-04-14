"""
Tests for the strategy ensemble module.

This module contains tests for the ensemble functionality of the strategy engine,
including combining multiple strategies and generating ensemble signals.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy_engine.base import StrategyEngine
from strategy_engine.ensemble import StrategyEnsemble
from strategy_engine.schema import StrategyInterface

class MockStrategy(StrategyInterface):
    """Mock strategy for testing."""
    
    def __init__(self, name, signals=None):
        self.name = name
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
def strategy1(sample_data):
    """Create a mock strategy with long signals."""
    signals = pd.DataFrame(index=sample_data.index)
    signals['signal'] = 0
    signals.loc['2020-03-01':'2020-03-15', 'signal'] = 1
    signals.loc['2020-06-01':'2020-06-15', 'signal'] = 1
    return MockStrategy('Strategy1', signals)

@pytest.fixture
def strategy2(sample_data):
    """Create a mock strategy with short signals."""
    signals = pd.DataFrame(index=sample_data.index)
    signals['signal'] = 0
    signals.loc['2020-03-01':'2020-03-15', 'signal'] = -1
    signals.loc['2020-06-01':'2020-06-15', 'signal'] = -1
    return MockStrategy('Strategy2', signals)

@pytest.fixture
def strategy3(sample_data):
    """Create a mock strategy with mixed signals."""
    signals = pd.DataFrame(index=sample_data.index)
    signals['signal'] = 0
    signals.loc['2020-03-01':'2020-03-15', 'signal'] = 1
    signals.loc['2020-06-01':'2020-06-15', 'signal'] = -1
    return MockStrategy('Strategy3', signals)

@pytest.fixture
def ensemble(strategy1, strategy2, strategy3):
    """Create a strategy ensemble for testing."""
    return StrategyEnsemble([strategy1, strategy2, strategy3])

def test_ensemble_initialization(strategy1, strategy2, strategy3):
    """Test ensemble initialization."""
    # Test with default weights
    ensemble = StrategyEnsemble([strategy1, strategy2, strategy3])
    assert len(ensemble.strategies) == 3
    assert len(ensemble.weights) == 3
    assert np.allclose(ensemble.weights, [1/3, 1/3, 1/3])
    
    # Test with custom weights
    ensemble = StrategyEnsemble([strategy1, strategy2, strategy3], weights=[0.5, 0.3, 0.2])
    assert len(ensemble.strategies) == 3
    assert len(ensemble.weights) == 3
    assert np.allclose(ensemble.weights, [0.5, 0.3, 0.2])
    
    # Test with mismatched weights
    with pytest.raises(ValueError, match="Number of weights must match number of strategies"):
        StrategyEnsemble([strategy1, strategy2, strategy3], weights=[0.5, 0.5])

def test_generate_signals(ensemble, sample_data):
    """Test ensemble signal generation."""
    signals = ensemble.generate_signals(sample_data)
    
    assert isinstance(signals, pd.DataFrame)
    assert 'signal' in signals.columns
    assert signals.index.equals(sample_data.index)
    
    # Check signal values
    assert signals.loc['2020-03-01':'2020-03-15', 'signal'].mean() == 0  # Equal long and short
    assert signals.loc['2020-06-01':'2020-06-15', 'signal'].mean() == 0  # Equal long and short
    
    # Test with custom weights
    ensemble = StrategyEnsemble([strategy1, strategy2, strategy3], weights=[0.6, 0.2, 0.2])
    signals = ensemble.generate_signals(sample_data)
    
    # Check signal values with custom weights
    assert signals.loc['2020-03-01':'2020-03-15', 'signal'].mean() > 0  # More long than short
    assert signals.loc['2020-06-01':'2020-06-15', 'signal'].mean() < 0  # More short than long

def test_get_parameters(ensemble):
    """Test getting ensemble parameters."""
    params = ensemble.get_parameters()
    
    assert isinstance(params, dict)
    assert 'weights' in params
    assert 'strategy_params' in params
    assert len(params['weights']) == 3
    assert len(params['strategy_params']) == 3
    assert np.allclose(params['weights'], [1/3, 1/3, 1/3])

def test_set_parameters(ensemble):
    """Test setting ensemble parameters."""
    # Set weights
    ensemble.set_parameters({'weights': [0.5, 0.3, 0.2]})
    assert np.allclose(ensemble.weights, [0.5, 0.3, 0.2])
    
    # Set strategy parameters
    strategy_params = [
        {'param1': 1, 'param2': 2},
        {'param1': 3, 'param2': 4},
        {'param1': 5, 'param2': 6}
    ]
    ensemble.set_parameters({'strategy_params': strategy_params})
    
    for i, strategy in enumerate(ensemble.strategies):
        assert strategy.parameters == strategy_params[i]
    
    # Test with mismatched weights
    with pytest.raises(ValueError, match="Number of weights must match number of strategies"):
        ensemble.set_parameters({'weights': [0.5, 0.5]})
    
    # Test with mismatched strategy parameters
    with pytest.raises(ValueError, match="Number of strategy parameters must match number of strategies"):
        ensemble.set_parameters({'strategy_params': [{'param1': 1}, {'param2': 2}]})

def test_evaluate(ensemble, sample_data):
    """Test ensemble evaluation."""
    result = ensemble.evaluate(sample_data)
    
    assert result is not None
    assert hasattr(result, 'equity_curve')
    assert hasattr(result, 'drawdown_curve')
    assert hasattr(result, 'metrics')
    
    # Test with custom engine
    engine = StrategyEngine(initial_capital=50000.0)
    result = ensemble.evaluate(sample_data, engine=engine)
    
    assert result is not None
    assert hasattr(result, 'equity_curve')
    assert hasattr(result, 'drawdown_curve')
    assert hasattr(result, 'metrics')

def test_add_strategy(ensemble, sample_data):
    """Test adding a strategy to the ensemble."""
    # Create a new strategy
    new_strategy = MockStrategy('NewStrategy')
    
    # Add the strategy
    ensemble.add_strategy(new_strategy)
    
    assert len(ensemble.strategies) == 4
    assert len(ensemble.weights) == 4
    assert np.allclose(ensemble.weights, [0.25, 0.25, 0.25, 0.25])
    
    # Test with custom weight
    ensemble.add_strategy(MockStrategy('AnotherStrategy'), weight=0.5)
    
    assert len(ensemble.strategies) == 5
    assert len(ensemble.weights) == 5
    # Weights should be [0.2, 0.2, 0.2, 0.2, 0.2] after normalization
    assert np.allclose(ensemble.weights, [0.2, 0.2, 0.2, 0.2, 0.2])

def test_remove_strategy(ensemble):
    """Test removing a strategy from the ensemble."""
    # Remove the first strategy
    ensemble.remove_strategy(0)
    
    assert len(ensemble.strategies) == 2
    assert len(ensemble.weights) == 2
    assert np.allclose(ensemble.weights, [0.5, 0.5])
    
    # Test with invalid index
    with pytest.raises(IndexError, match="Strategy index out of range"):
        ensemble.remove_strategy(10)

def test_clear(ensemble):
    """Test clearing the ensemble."""
    ensemble.clear()
    
    assert len(ensemble.strategies) == 0
    assert len(ensemble.weights) == 0 