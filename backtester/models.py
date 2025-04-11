"""
Data models for backtesting results.
"""

from dataclasses import dataclass
import pandas as pd
from typing import Dict, List, Any

@dataclass
class EvaluationResult:
    """Results of strategy evaluation."""
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    trades: List[Dict[str, Any]]
    metrics: Dict[str, float]
    signals: pd.DataFrame
    positions: pd.DataFrame
    portfolio_values: pd.DataFrame 