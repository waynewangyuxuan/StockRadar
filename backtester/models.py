from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class EvaluationResult:
    """Data class to store backtesting results."""
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trades: List[Dict]
    metrics: Dict[str, float]
    signals: Optional[pd.DataFrame] = None
    positions: Optional[pd.DataFrame] = None
    portfolio_values: Optional[pd.DataFrame] = None 