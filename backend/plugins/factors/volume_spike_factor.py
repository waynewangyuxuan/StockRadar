from core.factor_base import FactorBase, FactorType, FactorMetadata
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

class VolumeSpikeFactor(FactorBase):
    """Volume Spike Factor.
    
    This factor detects unusual volume spikes by comparing current volume
    to historical average volume.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Volume Spike factor.
        
        Args:
            config: Dictionary containing factor-specific configuration:
                   - window: Window for calculating average volume (default: 20)
                   - threshold: Threshold for spike detection (default: 2.0)
        """
        # Set instance variables before calling super().__init__()
        self.config = config or {}
        self.window = self.config.get('window', 20)
        self.threshold = self.config.get('threshold', 2.0)
        
        # Validate configuration
        if not isinstance(self.window, int) or self.window < 1:
            raise ValueError("Window must be a positive integer")
        if not isinstance(self.threshold, (int, float)) or self.threshold <= 0:
            raise ValueError("Threshold must be a positive number")
            
        # Initialize base class after setting instance variables
        super().__init__(self.config)
            
    def _get_metadata(self) -> FactorMetadata:
        """Get factor metadata for optimization and C++ translation.
        
        Returns:
            FactorMetadata object with factor characteristics
        """
        return FactorMetadata(
            name="volume_spike_factor",
            type=FactorType.VOLUME_BASED,
            required_columns=['volume'],
            output_columns=['volume_spike', 'volume_ratio'],
            is_vectorized=True,
            supports_batch=True,
            memory_efficient=True
        )
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume spike indicators.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with volume spike indicators
        """
        # Validate input
        self.validate_input(data)
        
        # Make a copy to avoid modifying input
        result = data.copy()
        
        # Calculate volume moving average
        result['volume_ma'] = result.groupby('ticker')['volume'].transform(
            lambda x: x.rolling(window=self.window, min_periods=1).mean()
        )
        
        # Calculate volume ratio and detect spikes
        result['volume_ratio'] = result['volume'] / result['volume_ma']
        result['volume_spike'] = (result['volume_ratio'] > self.threshold).astype(int)
        
        return result
