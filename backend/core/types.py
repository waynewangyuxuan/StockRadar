from dataclasses import dataclass
from enum import Enum
import pandas as pd

class FactorType(Enum):
    PRICE_BASED = "price_based"
    VOLUME_BASED = "volume_based"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    CUSTOM = "custom"

@dataclass
class MarketData:
    """Container for market data with validation."""
    data: pd.DataFrame
    
    def __post_init__(self):
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(self.data.index, pd.MultiIndex):
            raise ValueError("data must have a MultiIndex with levels ['date', 'ticker']")
        if not set(self.data.index.names) == {'date', 'ticker'}:
            raise ValueError("data index must have levels ['date', 'ticker']")
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(f"data must contain columns {required_columns}")

@dataclass
class SignalData:
    """Container for trading signals with validation."""
    data: pd.DataFrame
    
    def __post_init__(self):
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(self.data.index, pd.MultiIndex):
            raise ValueError("data must have a MultiIndex with levels ['date', 'ticker']")
        if not set(self.data.index.names) == {'date', 'ticker'}:
            raise ValueError("data index must have levels ['date', 'ticker']")
        if 'signal' not in self.data.columns:
            raise ValueError("data must contain a 'signal' column")
        if not self.data['signal'].isin([-1, 0, 1]).all():
            raise ValueError("signal values must be -1, 0, or 1") 