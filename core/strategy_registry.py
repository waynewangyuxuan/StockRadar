"""
Strategy registry module.

This module provides functionality to register, discover, and manage
trading strategies.
"""

from typing import Dict, Type, Any, Optional
from .schema import StrategyInterface

class StrategyRegistry:
    """
    Registry for trading strategies.
    
    This class provides methods to:
    - Register new strategies
    - Discover available strategies
    - Create strategy instances
    - Manage strategy parameters
    """
    
    def __init__(self):
        """Initialize the strategy registry."""
        self._strategies: Dict[str, Type[StrategyInterface]] = {}
        self._parameters: Dict[str, Dict[str, Any]] = {}
        
    def register(self, strategy_class: Type[StrategyInterface], 
                name: Optional[str] = None) -> None:
        """
        Register a new strategy class.
        
        Args:
            strategy_class: The strategy class to register
            name: Optional name for the strategy (defaults to class name)
        """
        if name is None:
            name = strategy_class.__name__
            
        if name in self._strategies:
            raise ValueError(f"Strategy {name} already registered")
            
        self._strategies[name] = strategy_class
        self._parameters[name] = {}
        
    def unregister(self, name: str) -> None:
        """
        Unregister a strategy.
        
        Args:
            name: Name of the strategy to unregister
        """
        if name not in self._strategies:
            raise ValueError(f"Strategy {name} not found")
            
        del self._strategies[name]
        del self._parameters[name]
        
    def get_strategy(self, name: str) -> Type[StrategyInterface]:
        """
        Get a registered strategy class.
        
        Args:
            name: Name of the strategy
            
        Returns:
            The strategy class
        """
        if name not in self._strategies:
            raise ValueError(f"Strategy {name} not found")
            
        return self._strategies[name]
        
    def create_strategy(self, name: str, **kwargs) -> StrategyInterface:
        """
        Create a new strategy instance.
        
        Args:
            name: Name of the strategy
            **kwargs: Additional arguments to pass to strategy constructor
            
        Returns:
            A new strategy instance
        """
        strategy_class = self.get_strategy(name)
        return strategy_class(**kwargs)
        
    def set_parameters(self, name: str, parameters: Dict[str, Any]) -> None:
        """
        Set default parameters for a strategy.
        
        Args:
            name: Name of the strategy
            parameters: Dictionary of parameters
        """
        if name not in self._strategies:
            raise ValueError(f"Strategy {name} not found")
            
        self._parameters[name] = parameters.copy()
        
    def get_parameters(self, name: str) -> Dict[str, Any]:
        """
        Get default parameters for a strategy.
        
        Args:
            name: Name of the strategy
            
        Returns:
            Dictionary of parameters
        """
        if name not in self._strategies:
            raise ValueError(f"Strategy {name} not found")
            
        return self._parameters[name].copy()
        
    def list_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered strategies and their parameters.
        
        Returns:
            Dictionary mapping strategy names to their parameters
        """
        return {name: self.get_parameters(name) 
                for name in self._strategies}
        
    def clear(self) -> None:
        """Clear all registered strategies."""
        self._strategies.clear()
        self._parameters.clear()

# Global registry instance
registry = StrategyRegistry()
