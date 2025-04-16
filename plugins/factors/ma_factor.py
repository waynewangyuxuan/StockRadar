from core.factor_base import FactorBase, FactorType, FactorMetadata
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

class MAFactor(FactorBase):
    """Moving Average Factor.
    
    This factor calculates simple and exponential moving averages for a specified column.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Moving Average factor.
        
        Args:
            config: Dictionary containing factor-specific configuration:
                   - column: Column to calculate moving averages for (default: 'close')
                   - windows: List of window sizes for moving averages (default: [5, 20, 50])
                   - ma_types: List of moving average types ('sma' or 'ema') (default: ['sma'])
        """
        # Set instance variables before calling super().__init__()
        self.config = config or {}
        self.column = self.config.get('column', 'close')
        self.windows = self.config.get('windows', [5, 20, 50])
        self.ma_types = self.config.get('ma_types', ['sma'])
        
        # Validate configuration
        if not isinstance(self.windows, list) or not self.windows:
            raise ValueError("Windows must be a non-empty list")
        if not all(isinstance(w, int) and w > 0 for w in self.windows):
            raise ValueError("All window sizes must be positive integers")
        if not all(t in ['sma', 'ema'] for t in self.ma_types):
            raise ValueError("Moving average types must be 'sma' or 'ema'")
            
        # Initialize base class after setting instance variables
        super().__init__(self.config)
    
    def _get_metadata(self) -> FactorMetadata:
        """Get factor metadata for optimization and C++ translation.
        
        Returns:
            FactorMetadata object with factor characteristics
        """
        output_columns = []
        for window in self.windows:
            for ma_type in self.ma_types:
                output_columns.append(f"{ma_type}_{window}")
                
        return FactorMetadata(
            name="ma_factor",
            type=FactorType.PRICE_BASED,
            required_columns=[self.column],
            output_columns=output_columns,
            is_vectorized=True,
            supports_batch=True,
            memory_efficient=True
        )
            
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with moving average columns added
        """
        # Validate input
        self.validate_input(data)
        
        # Make a copy to avoid modifying input
        result = data.copy()
        
        # Calculate moving averages for each window and type
        for window in self.windows:
            for ma_type in self.ma_types:
                col_name = f"{ma_type}_{window}"
                if ma_type == 'sma':
                    result[col_name] = result.groupby('ticker')[self.column].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                else:  # ema
                    result[col_name] = result.groupby('ticker')[self.column].transform(
                        lambda x: x.ewm(span=window, min_periods=1).mean()
                    )
        
        return result
