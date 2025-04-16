import pytest
from typing import Dict, Any
import pandas as pd
from core.strategy_registry import StrategyRegistry
from core.factor_registry import FactorRegistry
from core.schema import StrategyInterface

# Mock classes for testing
class MockStrategy(StrategyInterface):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()
    
    def get_metadata(self) -> Dict[str, Any]:
        return {"name": "mock_strategy"}

class MockFactor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

class TestStrategyRegistry:
    def test_register_strategy(self):
        registry = StrategyRegistry()
        registry.register("mock", MockStrategy)
        assert registry.get("mock") == MockStrategy
    
    def test_register_duplicate_strategy(self):
        registry = StrategyRegistry()
        registry.register("mock", MockStrategy)
        # Should log warning but not raise error
        registry.register("mock", MockStrategy)
        assert registry.get("mock") == MockStrategy
    
    def test_get_nonexistent_strategy(self):
        registry = StrategyRegistry()
        assert registry.get("nonexistent") is None
    
    def test_list_strategies(self):
        registry = StrategyRegistry()
        registry.register("mock1", MockStrategy)
        registry.register("mock2", MockStrategy)
        strategies = registry.list_strategies()
        assert len(strategies) == 2
        assert "mock1" in strategies
        assert "mock2" in strategies
    
    def test_create_strategy(self):
        registry = StrategyRegistry()
        registry.register("mock", MockStrategy)
        config = {"param": "value"}
        strategy = registry.create_strategy("mock", config)
        assert isinstance(strategy, MockStrategy)
        assert strategy.config == config
    
    def test_create_nonexistent_strategy(self):
        registry = StrategyRegistry()
        strategy = registry.create_strategy("nonexistent", {})
        assert strategy is None

class TestFactorRegistry:
    def test_register_factor(self):
        registry = FactorRegistry()
        registry.register("mock", MockFactor)
        assert registry.get("mock") == MockFactor
    
    def test_register_duplicate_factor(self):
        registry = FactorRegistry()
        registry.register("mock", MockFactor)
        # Should log warning but not raise error
        registry.register("mock", MockFactor)
        assert registry.get("mock") == MockFactor
    
    def test_get_nonexistent_factor(self):
        registry = FactorRegistry()
        assert registry.get("nonexistent") is None
    
    def test_list_factors(self):
        registry = FactorRegistry()
        registry.register("mock1", MockFactor)
        registry.register("mock2", MockFactor)
        factors = registry.list_factors()
        assert len(factors) == 2
        assert "mock1" in factors
        assert "mock2" in factors
    
    def test_create_factor(self):
        registry = FactorRegistry()
        registry.register("mock", MockFactor)
        config = {"param": "value"}
        factor = registry.create_factor("mock", config)
        assert isinstance(factor, MockFactor)
        assert factor.config == config
    
    def test_create_nonexistent_factor(self):
        registry = FactorRegistry()
        factor = registry.create_factor("nonexistent", {})
        assert factor is None 