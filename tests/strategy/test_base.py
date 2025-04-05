import pytest
import pandas as pd
from core.strategy.base import BaseStrategy

class MockStrategy(BaseStrategy):
    """Mock strategy for testing BaseStrategy abstract class"""
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mock signals"""
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        return signals
    
    def generate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Generate mock positions"""
        positions = pd.DataFrame(index=signals.index)
        positions["position"] = 0
        return positions

def test_base_strategy_initialization():
    """Test basic initialization of strategy"""
    params = {"param1": 10, "param2": "test"}
    strategy = MockStrategy("test_strategy", params)
    
    assert strategy.name == "test_strategy"
    assert strategy.parameters == params
    assert strategy.signals is None
    assert strategy.positions is None

def test_update_parameters():
    """Test parameter updating functionality"""
    initial_params = {"param1": 10, "param2": "test"}
    strategy = MockStrategy("test_strategy", initial_params)
    
    new_params = {"param1": 20}
    strategy.update_parameters(new_params)
    
    assert strategy.parameters["param1"] == 20
    assert strategy.parameters["param2"] == "test"

def test_get_strategy_info():
    """Test strategy info retrieval"""
    params = {"param1": 10, "param2": "test"}
    strategy = MockStrategy("test_strategy", params)
    
    info = strategy.get_strategy_info()
    assert info["name"] == "test_strategy"
    assert info["parameters"] == params
    assert isinstance(info["required_indicators"], list) 