"""
Strategy registry module.

This module provides functionality to register, discover, and manage
trading strategies.
"""

from typing import Dict, Type, Any, Optional
import logging
from .schema import StrategyInterface

class StrategyRegistry:
    """Registry for managing trading strategies."""
    
    def __init__(self):
        """Initialize the strategy registry."""
        self._strategies: Dict[str, Type[StrategyInterface]] = {}
        self._parameters: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
    def register(self, name: str, strategy_class: Type[StrategyInterface]) -> None:
        """Register a strategy class.
        
        Args:
            name: Name of the strategy
            strategy_class: Strategy class to register
        """
        if name in self._strategies:
            self.logger.warning(f"Strategy {name} already registered, overwriting")
        self._strategies[name] = strategy_class
        self.logger.info(f"Registered strategy: {name}")
        
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
        
    def get(self, name: str) -> Optional[Type[StrategyInterface]]:
        """Get a strategy class by name.
        
        Args:
            name: Name of the strategy
            
        Returns:
            Strategy class if found, None otherwise
        """
        return self._strategies.get(name)
        
    def create_strategy(self, name: str, config: dict) -> Optional[StrategyInterface]:
        """Create a strategy instance.
        
        Args:
            name: Name of the strategy
            config: Strategy configuration
            
        Returns:
            Strategy instance if found, None otherwise
        """
        strategy_class = self.get(name)
        if strategy_class is None:
            self.logger.error(f"Strategy {name} not found")
            return None
            
        try:
            return strategy_class(config)
        except Exception as e:
            self.logger.error(f"Failed to create strategy {name}: {str(e)}")
            return None
        
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
        
    def list_strategies(self) -> list[str]:
        """Get list of registered strategy names.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    def get_all(self) -> Dict[str, Type[StrategyInterface]]:
        """Get all registered strategies.

        Returns:
            Dictionary of strategy name to strategy class mappings
        """
        return self._strategies.copy()
        
    def clear(self) -> None:
        """Clear all registered strategies."""
        self._strategies.clear()
        self._parameters.clear()

# Global registry instance
registry = StrategyRegistry()
