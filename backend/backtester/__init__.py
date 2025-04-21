"""
Backtester module for StockRadar.

This module provides functionality for backtesting trading strategies.
"""

from .models import EvaluationResult
from .evaluator import StrategyEvaluator
from .visualization import BacktestVisualizer

__all__ = [
    'EvaluationResult',
    'StrategyEvaluator',
    'BacktestVisualizer'
]
