"""
Tests for the strategy schema module.

This module contains tests for the strategy interface and schema definitions,
including parameter validation and signal generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy_engine.schema import StrategyInterface, StrategyParameters

class MockStrategy(StrategyInterface):
    """Mock strategy for testing."""
    
    def __init__(self, name, signals=None):
        self.name = name
        self.signals = signals
        self.parameters = {}
        
    def generate_signals(self, data):
        super().generate_signals(data)  # Validate data
        if self.signals is not None:
            return self.signals
        return pd.DataFrame({'signal': [0] * len(data)}, index=data.index)
        
    def get_parameters(self):
        return self.parameters
        
    def set_parameters(self, parameters):
        super().set_parameters(parameters)  # Validate parameters
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
def strategy():
    """Create a mock strategy for testing."""
    return MockStrategy('TestStrategy')

def test_strategy_interface(strategy, sample_data):
    """Test the strategy interface."""
    # Test generate_signals
    signals = strategy.generate_signals(sample_data)
    assert isinstance(signals, pd.DataFrame)
    assert 'signal' in signals.columns
    assert signals.index.equals(sample_data.index)
    
    # Test get_parameters
    params = strategy.get_parameters()
    assert isinstance(params, dict)
    
    # Test set_parameters
    strategy.set_parameters({'param1': 1, 'param2': 2})
    assert strategy.parameters == {'param1': 1, 'param2': 2}

def test_strategy_parameters():
    """Test strategy parameters validation."""
    # Test valid parameters
    params = StrategyParameters(
        lookback_period=20,
        threshold=0.5,
        position_size=1.0
    )
    assert params.lookback_period == 20
    assert params.threshold == 0.5
    assert params.position_size == 1.0
    
    # Test invalid lookback period
    with pytest.raises(ValueError, match="Lookback period must be positive"):
        StrategyParameters(lookback_period=0)
    
    # Test invalid threshold
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        StrategyParameters(threshold=1.5)
    
    # Test invalid position size
    with pytest.raises(ValueError, match="Position size must be positive"):
        StrategyParameters(position_size=0)

def test_strategy_signals(strategy, sample_data):
    """Test strategy signal generation."""
    # Test with custom signals
    signals = pd.DataFrame(index=sample_data.index)
    signals['signal'] = 0
    signals.loc['2020-03-01':'2020-03-15', 'signal'] = 1
    strategy.signals = signals
    
    result = strategy.generate_signals(sample_data)
    assert isinstance(result, pd.DataFrame)
    assert 'signal' in result.columns
    assert result.index.equals(sample_data.index)
    assert (result.loc['2020-03-01':'2020-03-15', 'signal'] == 1).all()
    
    # Test with missing data
    with pytest.raises(ValueError, match="Data must contain required columns"):
        strategy.generate_signals(pd.DataFrame())

def test_strategy_parameter_validation(strategy):
    """Test strategy parameter validation."""
    # Test valid parameters
    strategy.set_parameters({
        'lookback_period': 20,
        'threshold': 0.5,
        'position_size': 1.0
    })
    assert strategy.parameters == {
        'lookback_period': 20,
        'threshold': 0.5,
        'position_size': 1.0
    }
    
    # Test invalid parameters
    with pytest.raises(ValueError, match="Invalid parameter value"):
        strategy.set_parameters({'lookback_period': 'invalid'})
    
    with pytest.raises(ValueError, match="Invalid parameter value"):
        strategy.set_parameters({'lookback_period': -1})

def test_strategy_interface_abstract_methods():
    """Test that StrategyInterface cannot be instantiated directly."""
    with pytest.raises(TypeError):
        StrategyInterface() 