from core.factor_base import FactorBase, FactorType, FactorMetadata
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

class VolumeSpikeFactor(FactorBase):
    """Volume Spike factor implementation.
    
    Detects unusual volume spikes by comparing current volume to:
    1. Rolling mean
    2. Rolling standard deviation
    3. Configurable threshold
    
    Performance optimizations:
    1. Vectorized numpy operations
    2. Efficient rolling statistics
    3. Memory-efficient calculations
    4. Clear C++ translation path
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the volume spike factor.
        
        Args:
            config: Dictionary with optional parameters:
                   - window: Rolling window size (default: 20)
                   - threshold: Number of standard deviations (default: 2.0)
        """
        # Set attributes before calling super().__init__
        self.window = int((config or {}).get('window', 20))
        self.threshold = float((config or {}).get('threshold', 2.0))
        
        # Validate config
        self._validate_config()
        
        # Call parent initialization
        super().__init__(config)
        
    def _validate_config(self):
        """Validate factor configuration."""
        if self.window < 1:
            raise ValueError("Window size must be positive")
        if self.threshold <= 0:
            raise ValueError("Threshold must be positive")
            
    def _get_metadata(self) -> FactorMetadata:
        """Get factor metadata for optimization."""
        return FactorMetadata(
            name=self.name,
            type=FactorType.VOLUME_BASED,
            required_columns=['ticker', 'volume'],
            output_columns=['volume_spike'],
            is_vectorized=True,
            supports_batch=True,
            memory_efficient=True
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume spike factor efficiently.
        
        Args:
            data: DataFrame with ticker and volume columns
            
        Returns:
            DataFrame with volume_spike column added
        """
        self.validate_input(data)
        
        # Create a copy to avoid modifying input
        result = data.copy()
        
        # Initialize volume_spike column with False
        result['volume_spike'] = pd.Series(False, index=result.index, dtype=bool)
        
        # Calculate for each ticker
        for ticker in result['ticker'].unique():
            mask = result['ticker'] == ticker
            ticker_data = result.loc[mask]
            
            # Calculate rolling statistics using pandas
            rolling = ticker_data['volume'].rolling(
                window=self.window,
                min_periods=self.window
            )
            rolling_mean = rolling.mean()
            rolling_std = rolling.std()
            
            # Detect spikes
            spikes = ticker_data['volume'] > (
                rolling_mean + self.threshold * rolling_std
            )
            
            # Store results
            result.loc[mask, 'volume_spike'] = spikes.astype(bool)
            
        return result
    
    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume spike factor in batch for better performance.
        
        Args:
            data: DataFrame with multiple tickers
            
        Returns:
            DataFrame with volume_spike column added
        """
        # Create a copy to avoid modifying input
        result = data.copy()
        
        # Initialize volume_spike column with False
        result['volume_spike'] = pd.Series(False, index=result.index, dtype=bool)
        
        # Calculate rolling statistics for all tickers at once
        grouped = result.groupby('ticker')['volume']
        rolling_mean = grouped.transform(
            lambda x: x.rolling(window=self.window, min_periods=self.window).mean()
        )
        rolling_std = grouped.transform(
            lambda x: x.rolling(window=self.window, min_periods=self.window).std()
        )
        
        # Detect spikes
        result['volume_spike'] = (
            result['volume'] > (rolling_mean + self.threshold * rolling_std)
        ).astype(bool)
        
        return result
