import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from .base import DataProcessorBase

class DataProcessor(DataProcessorBase):
    """Data processor for market data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data processor.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def get_required_metrics(self) -> List[str]:
        """Get list of required metrics."""
        return [
            'returns',
            'volatility',
            'vwap',
            'atr',
            'volume_momentum',
            'volume_volatility',
            'relative_volume',
            'momentum',
            'acceleration'
        ]
    
    def validate_factors(self, factors: List[str]) -> bool:
        """Validate factor list."""
        if not factors:
            raise ValueError("Factor list cannot be empty")
        if not all(isinstance(f, str) and f.strip() for f in factors):
            raise ValueError("All factors must be non-empty strings")
        return True
    
    def process_data(self, data: pd.DataFrame, factors: List[str], start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Process raw market data with comprehensive technical indicators.
        
        Args:
            data: Raw market data DataFrame with MultiIndex (date, ticker)
            factors: List of factors to calculate
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            Processed DataFrame with technical indicators
        """
        # Validate inputs
        self.validate_data(data)
        self.validate_factors(factors)
        self.validate_dates(start_date, end_date)
        
        # Filter by date if specified
        if start_date or end_date:
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask &= data.index.get_level_values('date') >= start_date
            if end_date:
                mask &= data.index.get_level_values('date') <= end_date
            data = data[mask].copy()
        
        # Make a copy to avoid modifying input
        processed = data.copy()
        
        # Calculate for each ticker separately
        for ticker in processed.index.get_level_values('ticker').unique():
            # Create a mask for this ticker using index
            mask = processed.index.get_level_values('ticker') == ticker
            
            # Basic price metrics
            processed.loc[mask, 'returns'] = processed.loc[mask, 'close'].pct_change(fill_method=None)
            processed.loc[mask, 'log_returns'] = np.log1p(processed.loc[mask, 'returns'])
            processed.loc[mask, 'price_change'] = processed.loc[mask, 'close'].diff()
            
            # True Range and ATR
            prev_close = processed.loc[mask, 'close'].shift(1)
            processed.loc[mask, 'tr'] = pd.DataFrame({
                'hl': processed.loc[mask, 'high'] - processed.loc[mask, 'low'],
                'hc': abs(processed.loc[mask, 'high'] - prev_close),
                'lc': abs(processed.loc[mask, 'low'] - prev_close)
            }).max(axis=1)
            processed.loc[mask, 'atr'] = processed.loc[mask, 'tr'].rolling(window=14, min_periods=1).mean()
            
            # Volume metrics
            processed.loc[mask, 'volume_momentum'] = processed.loc[mask, 'volume'].pct_change(periods=5, fill_method=None)
            processed.loc[mask, 'volume_volatility'] = processed.loc[mask, 'volume'].rolling(window=20).std()
            processed.loc[mask, 'relative_volume'] = processed.loc[mask, 'volume'] / processed.loc[mask, 'volume'].rolling(window=20).mean()
            
            # Price momentum and volatility
            processed.loc[mask, 'momentum'] = processed.loc[mask, 'close'].pct_change(periods=5, fill_method=None)
            processed.loc[mask, 'acceleration'] = processed.loc[mask, 'momentum'].diff()
            processed.loc[mask, 'volatility'] = processed.loc[mask, 'returns'].rolling(window=20).std()
            
            # VWAP and money flow
            processed.loc[mask, 'typical_price'] = (
                processed.loc[mask, ['high', 'low', 'close']].mean(axis=1)
            ).clip(
                lower=processed.loc[mask, 'low'],
                upper=processed.loc[mask, 'high']
            )
            processed.loc[mask, 'money_flow'] = processed.loc[mask, 'typical_price'] * processed.loc[mask, 'volume']
            processed.loc[mask, 'vwap'] = (
                processed.loc[mask, 'money_flow'].cumsum() / processed.loc[mask, 'volume'].cumsum()
            ).clip(
                lower=processed.loc[mask, 'low'],
                upper=processed.loc[mask, 'high']
            )
            
            # Price ranges
            processed.loc[mask, 'price_range'] = processed.loc[mask, 'high'] - processed.loc[mask, 'low']
            processed.loc[mask, 'price_range_pct'] = processed.loc[mask, 'price_range'] / processed.loc[mask, 'close']
        
        # Fill missing values
        processed = processed.ffill().bfill().fillna(0)
        
        return processed 