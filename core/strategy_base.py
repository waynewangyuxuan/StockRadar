from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, List
from enum import Enum

class SignalType(Enum):
    """Enum for different types of trading signals."""
    BUY = 1
    SELL = -1
    HOLD = 0

class StrategyBase(ABC):
    """Base class for all trading strategies in StockRadar.
    
    All strategies must inherit from this class and implement its abstract methods.
    This ensures a consistent interface across all strategy implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy with optional configuration.
        
        Args:
            config: Dictionary containing strategy-specific configuration parameters
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame, factor_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate trading signals based on market data and optional factor data.
        
        Args:
            market_data: DataFrame containing market data with at least:
                       ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            factor_data: Optional DataFrame containing pre-computed factor values
        
        Returns:
            DataFrame with at least these columns added:
            - signal: SignalType enum value
            - signal_strength: float between 0 and 1
            - strategy_name: name of the strategy
            - timestamp: when the signal was generated
        """
        pass
    
    @abstractmethod
    def get_required_factors(self) -> List[str]:
        """Get list of factor names required by this strategy.
        
        Returns:
            List of factor names that must be pre-computed
        """
        pass
    
    def validate_input(self, market_data: pd.DataFrame, factor_data: Optional[pd.DataFrame] = None) -> bool:
        """Validate that all required data is present.
        
        Args:
            market_data: Market data DataFrame to validate
            factor_data: Optional factor data DataFrame to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check market data columns
        required_market_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_market = [col for col in required_market_cols if col not in market_data.columns]
        if missing_market:
            raise ValueError(f"Missing required market data columns: {missing_market}")
            
        # Check factor data if required
        required_factors = self.get_required_factors()
        if required_factors:
            if factor_data is None:
                raise ValueError(f"Strategy {self.name} requires factors but none provided: {required_factors}")
            missing_factors = [f for f in required_factors if f not in factor_data.columns]
            if missing_factors:
                raise ValueError(f"Missing required factor columns: {missing_factors}")
                
        return True
    
    def get_signal_columns(self) -> List[str]:
        """Get list of columns this strategy adds to the DataFrame.
        
        Returns:
            List of column names that this strategy computes and adds
        """
        return ['signal', 'signal_strength', 'strategy_name', 'timestamp']
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name}(config={self.config})"
