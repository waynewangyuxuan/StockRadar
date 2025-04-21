"""
Data models for backtesting results.
"""

from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Any

@dataclass
class EvaluationResult:
    """Results of strategy evaluation."""
    equity_curve: pd.DataFrame  # Portfolio value over time
    drawdown_curve: pd.DataFrame  # Drawdown percentage over time
    trades: List[Dict[str, Any]]  # Summary of all trades
    metrics: Dict[str, float]  # Performance metrics
    signals: pd.DataFrame
    positions: pd.DataFrame  # Position history
    portfolio_values: pd.DataFrame 