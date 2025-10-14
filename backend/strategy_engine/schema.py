from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class StrategyInterface(ABC):
    """Base interface for all trading strategies."""
    
    @abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy with configuration.
        
        Args:
            config: Dictionary containing strategy-specific configuration
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from market data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get strategy metadata.
        
        Returns:
            Dictionary with strategy metadata
        """
        pass

class DataSchema:
    """Schema for market data."""
    
    # Required columns
    TICKER = 'ticker'
    DATE = 'date'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VOLUME = 'volume'
    
    # Optional columns
    ADJ_CLOSE = 'adj_close'
    VWAP = 'vwap'
    RETURNS = 'returns'
    VOLATILITY = 'volatility'
    
    @classmethod
    def get_required_columns(cls) -> List[str]:
        """Get list of required columns.
        
        Returns:
            List of required column names
        """
        return [
            cls.TICKER,
            cls.DATE,
            cls.OPEN,
            cls.HIGH,
            cls.LOW,
            cls.CLOSE,
            cls.VOLUME
        ]
    
    @classmethod
    def validate_data(cls, data: pd.DataFrame) -> bool:
        """Validate data against schema.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        required_cols = cls.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return True

class SignalType(Enum):
    """Types of trading signals."""
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class SignalSchema:
    """Schema for trading signals."""
    
    # Required columns
    TICKER = 'ticker'
    DATE = 'date'
    SIGNAL = 'signal'
    
    # Optional columns
    STRENGTH = 'strength'
    STRATEGY = 'strategy'
    PRICE = 'price'
    VOLUME = 'volume'
    
    @classmethod
    def get_required_columns(cls) -> List[str]:
        """Get list of required columns.
        
        Returns:
            List of required column names
        """
        return [
            cls.TICKER,
            cls.DATE,
            cls.SIGNAL
        ]
    
    @classmethod
    def validate_signals(cls, signals: pd.DataFrame) -> bool:
        """Validate signals against schema.
        
        Args:
            signals: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        required_cols = cls.get_required_columns()
        missing_cols = [col for col in required_cols if col not in signals.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Validate signal values
        if not all(signals[cls.SIGNAL].isin([s.value for s in SignalType])):
            raise ValueError(f"Invalid signal values. Must be one of: {[s.value for s in SignalType]}")
            
        return True 