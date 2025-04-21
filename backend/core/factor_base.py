from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

class FactorType(Enum):
    """Factor types for C++ translation and optimization."""
    PRICE_BASED = "price_based"
    VOLUME_BASED = "volume_based"
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"

@dataclass
class FactorMetadata:
    """Metadata for factor optimization and C++ translation."""
    name: str
    type: FactorType
    required_columns: List[str]
    output_columns: List[str]
    is_vectorized: bool = True
    supports_batch: bool = True
    memory_efficient: bool = True

class FactorBase(ABC):
    """Base class for all factors in StockRadar.
    
    Design principles:
    1. Performance-first with numpy operations
    2. Clear C++ translation path
    3. Memory efficiency
    4. Batch processing support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the factor with optional configuration.
        
        Args:
            config: Dictionary containing factor-specific configuration parameters
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.metadata = self._get_metadata()
        
    @abstractmethod
    def _get_metadata(self) -> FactorMetadata:
        """Get factor metadata for optimization and C++ translation.
        
        Returns:
            FactorMetadata object with factor characteristics
        """
        pass
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the factor values for the given market data.
        
        Args:
            data: DataFrame containing market data with required columns
        
        Returns:
            DataFrame with factor values added as new columns
        """
        pass
    
    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor values in batch for better performance.
        
        Args:
            data: DataFrame containing market data
            
        Returns:
            DataFrame with factor values
        """
        if not self.metadata.supports_batch:
            return self.calculate(data)
            
        # Default batch implementation using numpy
        result = data.copy()
        for ticker in data['ticker'].unique():
            mask = data['ticker'] == ticker
            ticker_data = data[mask]
            result.loc[mask] = self.calculate(ticker_data)
        return result
    
    def get_required_columns(self) -> List[str]:
        """Get required input columns."""
        return self.metadata.required_columns
    
    def get_factor_columns(self) -> List[str]:
        """Get output factor columns."""
        return self.metadata.output_columns
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        missing = [col for col in self.get_required_columns() if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns for {self.name}: {missing}")
        return True
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Preprocess data for efficient calculation.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (ticker array, column arrays)
        """
        tickers = data['ticker'].values
        columns = {
            col: data[col].values 
            for col in self.get_required_columns()
        }
        return tickers, columns
    
    def __str__(self) -> str:
        """String representation of the factor."""
        return f"{self.name}(type={self.metadata.type.value}, config={self.config})"
