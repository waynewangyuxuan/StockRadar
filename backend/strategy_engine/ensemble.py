"""
Strategy ensemble module.

This module provides functionality to combine multiple trading strategies
into an ensemble for improved performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .schema import StrategyInterface
from .base import StrategyEngine

class StrategyEnsemble:
    """
    Ensemble of trading strategies.
    
    This class provides methods to:
    - Combine multiple strategies
    - Weight strategy signals
    - Generate ensemble signals
    - Evaluate ensemble performance
    """
    
    def __init__(self, strategies: List[StrategyInterface],
                 weights: Optional[List[float]] = None):
        """
        Initialize the strategy ensemble.
        
        Args:
            strategies: List of strategy instances
            weights: Optional list of strategy weights (defaults to equal weights)
        """
        self.strategies = strategies
        self.weights = weights if weights is not None else [1.0] * len(strategies)
        
        if len(self.weights) != len(strategies):
            raise ValueError("Number of weights must match number of strategies")
            
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble signals from market data.
        
        Args:
            data: DataFrame with OHLCV data and any required factor columns
            
        Returns:
            DataFrame with ensemble signals
        """
        # Generate signals from each strategy
        strategy_signals = []
        for strategy in self.strategies:
            signals = strategy.generate_signals(data)
            strategy_signals.append(signals)
            
        # Combine signals using weights
        ensemble_signals = pd.DataFrame(0, index=data.index, columns=['signal'])
        
        for signals, weight in zip(strategy_signals, self.weights):
            ensemble_signals['signal'] += signals['signal'] * weight
            
        # Convert weighted signals to discrete signals
        ensemble_signals['signal'] = np.where(ensemble_signals['signal'] > 0.5, 1,
                                            np.where(ensemble_signals['signal'] < -0.5, -1, 0))
        
        return ensemble_signals
        
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get ensemble parameters.
        
        Returns:
            Dictionary of ensemble parameters
        """
        return {
            'weights': self.weights.copy(),
            'strategy_params': [s.get_parameters() for s in self.strategies]
        }
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set ensemble parameters.
        
        Args:
            parameters: Dictionary of ensemble parameters
        """
        if 'weights' in parameters:
            weights = parameters['weights']
            if len(weights) != len(self.strategies):
                raise ValueError("Number of weights must match number of strategies")
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
            
        if 'strategy_params' in parameters:
            strategy_params = parameters['strategy_params']
            if len(strategy_params) != len(self.strategies):
                raise ValueError("Number of strategy parameters must match number of strategies")
            for strategy, params in zip(self.strategies, strategy_params):
                strategy.set_parameters(params)
                
    def evaluate(self, data: pd.DataFrame, engine: Optional[StrategyEngine] = None) -> Any:
        """
        Evaluate ensemble performance.
        
        Args:
            data: DataFrame with OHLCV data
            engine: Optional strategy engine for evaluation
            
        Returns:
            Evaluation results
        """
        if engine is None:
            engine = StrategyEngine()
            
        signals = self.generate_signals(data)
        return engine.evaluate_strategy(self, data)
        
    def add_strategy(self, strategy: StrategyInterface, weight: float = 1.0) -> None:
        """
        Add a strategy to the ensemble.
        
        Args:
            strategy: Strategy to add
            weight: Weight for the strategy
        """
        self.strategies.append(strategy)
        self.weights.append(weight)
        
        # Renormalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    def remove_strategy(self, index: int) -> None:
        """
        Remove a strategy from the ensemble.
        
        Args:
            index: Index of strategy to remove
        """
        if index < 0 or index >= len(self.strategies):
            raise IndexError("Strategy index out of range")
            
        del self.strategies[index]
        del self.weights[index]
        
        # Renormalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    def clear(self) -> None:
        """Clear all strategies from the ensemble."""
        self.strategies.clear()
        self.weights.clear()
