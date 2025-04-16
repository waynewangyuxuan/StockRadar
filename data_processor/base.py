from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

class DataProcessorBase(ABC):
    """Base class for all data processors in StockRadar.
    
    This abstract class defines the interface that all data processors must implement.
    It ensures consistent data processing and validation across different implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data processor with configuration.
        
        Args:
            config: Dictionary containing processor-specific configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def process_data(self, 
                    data: pd.DataFrame,
                    factors: List[str],
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Process market data.
        
        This method should:
        1. Validate input data and parameters
        2. Clean and normalize the data
        3. Calculate basic market metrics
        4. Return processed DataFrame
        
        Args:
            data: DataFrame with market data
            factors: List of factor names to calculate (for interface compatibility)
            start_date: Optional start date for processing
            end_date: Optional end date for processing
            
        Returns:
            DataFrame with processed data and basic market metrics
            
        Raises:
            ValueError: If parameters are invalid
            DataError: If data is invalid or missing
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the input data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if data.empty:
            raise ValueError("DataFrame is empty")
            
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            raise ValueError("Date column must be datetime")
            
        return True
    
    def validate_factors(self, factors: List[str]) -> bool:
        """Validate the list of factors.
        
        Args:
            factors: List of factor names to validate
            
        Returns:
            True if valid, raises ValueError if invalid
            
        Raises:
            ValueError: If factors list is empty or contains invalid factors
        """
        if not factors:
            raise ValueError("Factor list cannot be empty")
            
        if not all(isinstance(f, str) and f.strip() for f in factors):
            raise ValueError("All factors must be non-empty strings")
            
        return True
    
    def validate_dates(self, 
                      start_date: Optional[datetime],
                      end_date: Optional[datetime]) -> bool:
        """Validate the date range.
        
        Args:
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if start_date is not None and not isinstance(start_date, datetime):
            raise ValueError("Start date must be datetime")
            
        if end_date is not None and not isinstance(end_date, datetime):
            raise ValueError("End date must be datetime")
            
        if start_date is not None and end_date is not None:
            if start_date > end_date:
                raise ValueError("Start date must be before end date")
                
        return True
    
    def get_required_metrics(self) -> List[str]:
        """Get the list of basic market metrics that should be calculated.
        
        Returns:
            List of metric names that should be present in the processed data
        """
        return [
            'returns',
            'log_returns',
            'price_change',
            'tr',
            'atr',
            'typical_price',
            'money_flow',
            'momentum',
            'volume_momentum',
            'acceleration',
            'volatility',
            'volume_volatility',
            'price_range',
            'price_range_pct',
            'vwap',
            'relative_volume'
        ]
    
    def __str__(self) -> str:
        """String representation of the data processor."""
        return f"{self.__class__.__name__}(config={self.config})" 