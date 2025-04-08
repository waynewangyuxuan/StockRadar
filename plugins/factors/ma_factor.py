from core.factor_base import FactorBase, FactorType, FactorMetadata
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

class MovingAverageFactor(FactorBase):
    """Moving Average factor implementation.
    
    Calculates moving average of price data with:
    1. Configurable window size
    2. Efficient numpy operations
    3. Memory-efficient calculations
    4. Clear C++ translation path
    
    Performance optimizations:
    1. Vectorized numpy operations
    2. Efficient rolling window calculations
    3. Minimal memory allocations
    4. Batch processing support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the moving average factor.
        
        Args:
            config: Dictionary with optional parameters:
                   - window: Rolling window size (default: 20)
                   - price_col: Column to calculate MA on (default: 'close')
        """
        # Set attributes before calling super().__init__
        self.window = int((config or {}).get('window', 20))
        self.price_col = str((config or {}).get('price_col', 'close'))
        
        # Validate config
        self._validate_config()
        
        # Call parent initialization
        super().__init__(config)
        
    def _validate_config(self):
        """Validate factor configuration."""
        if self.window < 1:
            raise ValueError("Window size must be positive")
            
    def _get_metadata(self) -> FactorMetadata:
        """Get factor metadata for optimization."""
        return FactorMetadata(
            name=self.name,
            type=FactorType.PRICE_BASED,
            required_columns=['ticker', self.price_col],
            output_columns=[f'ma_{self.window}'],
            is_vectorized=True,
            supports_batch=True,
            memory_efficient=True
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving average efficiently.
        
        Args:
            data: DataFrame with ticker and price columns
            
        Returns:
            DataFrame with moving average column added
        """
        self.validate_input(data)
        
        # Create a copy to avoid modifying input
        result = data.copy()
        
        # Calculate for each ticker
        for ticker in result['ticker'].unique():
            mask = result['ticker'] == ticker
            ticker_data = result.loc[mask]
            
            # Calculate moving average using pandas rolling
            # This handles NaN values correctly
            ma = ticker_data[self.price_col].rolling(
                window=self.window,
                min_periods=self.window
            ).mean()
            
            # Store results
            result.loc[mask, f'ma_{self.window}'] = ma
            
        return result
    
    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving average in batch for better performance.
        
        Args:
            data: DataFrame with multiple tickers
            
        Returns:
            DataFrame with moving average column added
        """
        # Create a copy to avoid modifying input
        result = data.copy()
        
        # Calculate MA for all tickers at once
        result[f'ma_{self.window}'] = result.groupby('ticker')[self.price_col].transform(
            lambda x: x.rolling(window=self.window, min_periods=self.window).mean()
        )
        
        return result
