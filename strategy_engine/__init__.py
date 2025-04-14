"""
Strategy engine package.

This package provides a framework for developing, evaluating, and combining
trading strategies. It includes:

- Strategy interface and schema definitions
- Strategy evaluation and analysis
- Strategy visualization
- Strategy ensemble support
- Strategy registry and management
"""

from .base import StrategyEngine, StrategyResult
from .schema import StrategyInterface, DataSchema, SignalSchema
from .visualization import StrategyVisualizer
from .ensemble import StrategyEnsemble
from .registry import registry

__all__ = [
    'StrategyEngine',
    'StrategyResult',
    'StrategyInterface',
    'DataSchema',
    'SignalSchema',
    'StrategyVisualizer',
    'StrategyEnsemble',
    'registry'
]
