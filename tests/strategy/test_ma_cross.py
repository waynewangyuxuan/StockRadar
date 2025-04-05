import pytest
import pandas as pd
import numpy as np
from core.strategy.ma_cross import MACrossStrategy

@pytest.fixture
def sample_data():
    """Create sample price data with moving averages"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame(index=dates)
    data['MA_10'] = np.random.randn(100).cumsum()  # Simulated MA10
    data['MA_30'] = np.random.randn(100).cumsum()  # Simulated MA30
    return data

@pytest.fixture
def strategy():
    """Create MA crossover strategy instance"""
    return MACrossStrategy(fast_period=10, slow_period=30)

def test_ma_strategy_initialization(strategy):
    """Test MA strategy initialization"""
    assert strategy.name == "MA_Crossover"
    assert strategy.parameters["fast_period"] == 10
    assert strategy.parameters["slow_period"] == 30

def test_validate_parameters():
    """Test parameter validation"""
    # Valid parameters
    strategy = MACrossStrategy(fast_period=10, slow_period=30)
    assert strategy.validate_parameters() is True
    
    # Invalid parameters
    with pytest.raises(ValueError):
        strategy.update_parameters({"fast_period": 40, "slow_period": 30})
    
    with pytest.raises(ValueError):
        strategy.update_parameters({"fast_period": -1})

def test_required_indicators(strategy):
    """Test required indicators list"""
    indicators = strategy.get_required_indicators()
    assert "MA_10" in indicators
    assert "MA_30" in indicators
    assert len(indicators) == 2

def test_calculate_signals(strategy, sample_data):
    """Test signal calculation"""
    signals = strategy.calculate_signals(sample_data)
    
    assert isinstance(signals, pd.DataFrame)
    assert "signal" in signals.columns
    assert signals["signal"].isin([0, 1, -1]).all()
    
    # Test signal generation logic
    fast_ma = sample_data["MA_10"]
    slow_ma = sample_data["MA_30"]
    
    # Check bullish crossover
    bullish_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    assert (signals.loc[bullish_cross, "signal"] == 1).all()
    
    # Check bearish crossover
    bearish_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    assert (signals.loc[bearish_cross, "signal"] == -1).all()

def test_generate_positions(strategy, sample_data):
    """Test position generation"""
    signals = strategy.calculate_signals(sample_data)
    positions = strategy.generate_positions(signals)
    
    assert isinstance(positions, pd.DataFrame)
    assert "position" in positions.columns
    assert positions["position"].isin([0, 1, -1]).all()
    
    # Test position logic
    assert (positions.loc[signals["signal"] == 1, "position"] == 1).all()
    assert (positions.loc[signals["signal"] == -1, "position"] == -1).all()

def test_missing_indicators(strategy):
    """Test handling of missing indicators"""
    invalid_data = pd.DataFrame({
        'MA_10': [1, 2, 3],
        'Price': [10, 11, 12]  # Missing MA_30
    })
    
    with pytest.raises(ValueError, match="Required MA indicators not found in data"):
        strategy.calculate_signals(invalid_data) 