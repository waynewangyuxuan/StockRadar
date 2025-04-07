from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class FactorBase(ABC):
    """Base class for all factors in StockRadar.
    
    All factors must inherit from this class and implement its abstract methods.
    This ensures a consistent interface across all factor implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the factor with optional configuration.
        
        Args:
            config: Dictionary containing factor-specific configuration parameters
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the factor values for the given market data.
        
        Args:
            data: DataFrame containing market data with at least these columns:
                 ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with factor values added as new columns.
            Must preserve all input columns and add factor-specific columns.
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """Get the list of required input columns for this factor.
        
        Returns:
            List of column names that must be present in the input data.
        """
        pass
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate that the input data has all required columns.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        missing = [col for col in self.get_required_columns() if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns for {self.name}: {missing}")
        return True
    
    def get_factor_columns(self) -> list[str]:
        """Get the list of columns this factor adds to the DataFrame.
        
        Returns:
            List of column names that this factor computes and adds
        """
        return []  # Override in concrete implementations
    
    def __str__(self) -> str:
        """String representation of the factor."""
        return f"{self.name}(config={self.config})"
