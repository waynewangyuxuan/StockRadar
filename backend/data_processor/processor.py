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
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the input market data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
        
        # Check if we have a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Input data must be a DataFrame, got {type(data)}")
        
        # Check if we have a MultiIndex
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"Input data must have a MultiIndex, got {type(data.index)}")
        
        # Check if the index has the required levels
        required_levels = ['date', 'ticker']
        for level in required_levels:
            if level not in data.index.names:
                raise ValueError(f"Input data index must have level '{level}', got {data.index.names}")
        
        # Check if we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Input data is missing required columns: {missing_columns}")
        
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
        
        # Ensure we have a MultiIndex DataFrame with 'date' and 'ticker' levels
        if not isinstance(data.index, pd.MultiIndex) or 'date' not in data.index.names or 'ticker' not in data.index.names:
            self.logger.error(f"Expected MultiIndex with 'date' and 'ticker', got {data.index.names}")
            raise ValueError("Input data must have a MultiIndex with 'date' and 'ticker' levels")
        
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
        
        # Get unique tickers from the index levels
        unique_tickers = processed.index.get_level_values('ticker').unique()
        
        # Calculate for each ticker separately
        for ticker in unique_tickers:
            # Create a mask for this ticker using index levels
            ticker_mask = processed.index.get_level_values('ticker') == ticker
            
            # Basic price metrics
            processed.loc[ticker_mask, 'returns'] = processed.loc[ticker_mask, 'close'].pct_change(fill_method=None)
            processed.loc[ticker_mask, 'log_returns'] = np.log1p(processed.loc[ticker_mask, 'returns'])
            processed.loc[ticker_mask, 'price_change'] = processed.loc[ticker_mask, 'close'].diff()
            
            # True Range and ATR
            prev_close = processed.loc[ticker_mask, 'close'].shift(1)
            processed.loc[ticker_mask, 'tr'] = pd.DataFrame({
                'hl': processed.loc[ticker_mask, 'high'] - processed.loc[ticker_mask, 'low'],
                'hc': abs(processed.loc[ticker_mask, 'high'] - prev_close),
                'lc': abs(processed.loc[ticker_mask, 'low'] - prev_close)
            }).max(axis=1)
            processed.loc[ticker_mask, 'atr'] = processed.loc[ticker_mask, 'tr'].rolling(window=14, min_periods=1).mean()
            
            # Volume metrics
            processed.loc[ticker_mask, 'volume_momentum'] = processed.loc[ticker_mask, 'volume'].pct_change(periods=5, fill_method=None)
            processed.loc[ticker_mask, 'volume_volatility'] = processed.loc[ticker_mask, 'volume'].rolling(window=20).std()
            processed.loc[ticker_mask, 'relative_volume'] = processed.loc[ticker_mask, 'volume'] / processed.loc[ticker_mask, 'volume'].rolling(window=20).mean()
            
            # Price momentum and volatility
            processed.loc[ticker_mask, 'momentum'] = processed.loc[ticker_mask, 'close'].pct_change(periods=5, fill_method=None)
            processed.loc[ticker_mask, 'acceleration'] = processed.loc[ticker_mask, 'momentum'].diff()
            processed.loc[ticker_mask, 'volatility'] = processed.loc[ticker_mask, 'returns'].rolling(window=20).std()
            
            # VWAP and money flow
            processed.loc[ticker_mask, 'typical_price'] = (
                processed.loc[ticker_mask, ['high', 'low', 'close']].mean(axis=1)
            ).clip(
                lower=processed.loc[ticker_mask, 'low'],
                upper=processed.loc[ticker_mask, 'high']
            )
            processed.loc[ticker_mask, 'money_flow'] = processed.loc[ticker_mask, 'typical_price'] * processed.loc[ticker_mask, 'volume']
            processed.loc[ticker_mask, 'vwap'] = (
                processed.loc[ticker_mask, 'money_flow'].cumsum() / processed.loc[ticker_mask, 'volume'].cumsum()
            ).clip(
                lower=processed.loc[ticker_mask, 'low'],
                upper=processed.loc[ticker_mask, 'high']
            )
            
            # Price ranges
            processed.loc[ticker_mask, 'price_range'] = processed.loc[ticker_mask, 'high'] - processed.loc[ticker_mask, 'low']
            processed.loc[ticker_mask, 'price_range_pct'] = processed.loc[ticker_mask, 'price_range'] / processed.loc[ticker_mask, 'close']
        
        # Fill missing values
        processed = processed.ffill().bfill().fillna(0)
        
        return processed 