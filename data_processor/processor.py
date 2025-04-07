import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from .base import DataProcessorBase

class DataProcessor(DataProcessorBase):
    """Concrete implementation of the data processor.
    
    This class handles core data processing tasks like:
    - Data validation and cleaning
    - Data normalization
    - Data aggregation
    - Basic market metrics calculation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data processor.
        
        Args:
            config: Dictionary containing processor-specific configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def process_data(self, 
                    data: pd.DataFrame,
                    factors: List[str],
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Process market data.
        
        Args:
            data: DataFrame with market data
            factors: List of factor names to calculate
            start_date: Optional start date for processing
            end_date: Optional end date for processing
            
        Returns:
            DataFrame with processed data and basic market metrics
        """
        # Validate inputs
        self.validate_data(data)
        self.validate_factors(factors)
        self.validate_dates(start_date, end_date)
        
        # Make a copy to avoid modifying the input
        data = data.copy()
        
        # Filter by date first if needed (more efficient to do it before setting index)
        if start_date is not None:
            data = data[data['date'] >= start_date]
        if end_date is not None:
            data = data[data['date'] <= end_date]
        
        # Sort by date and ticker for efficiency
        data = data.sort_values(['date', 'ticker'])
        
        # Set index for grouping operations
        data = data.set_index(['ticker', 'date'])  # Ticker first for proper grouping
            
        # Calculate basic metrics
        data = self._calculate_basic_metrics(data)
        
        # Reset index and ensure date order
        data = data.reset_index().sort_values('date')
        
        return data
        
    def _calculate_basic_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic market metrics that are commonly used.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with additional market metrics
        """
        # Calculate daily returns and clip to valid range
        data['returns'] = data.groupby(level='ticker')['close'].pct_change().fillna(0).clip(-1, 1)
        
        # Calculate log returns
        prev_close = data.groupby(level='ticker')['close'].shift(1)
        data['log_returns'] = np.log(data['close'] / prev_close).fillna(0)
        
        # Calculate price changes
        data['price_change'] = data['close'] - prev_close
        
        # Calculate true range (high-low, high-prev_close, low-prev_close)
        data['tr'] = pd.DataFrame({
            'hl': data['high'] - data['low'],
            'hc': abs(data['high'] - prev_close),
            'lc': abs(data['low'] - prev_close)
        }).max(axis=1)
        
        # Calculate average true range (ATR)
        data['atr'] = data.groupby(level='ticker')['tr'].transform(
            lambda x: x.rolling(window=14, min_periods=1).mean()
        )
        
        # Calculate typical price
        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate money flow
        data['money_flow'] = data['typical_price'] * data['volume']
        
        # Calculate momentum (close price change over N periods)
        data['momentum'] = data['close'] - data.groupby(level='ticker')['close'].shift(10)
        
        # Calculate volume momentum
        data['volume_momentum'] = data['volume'] - data.groupby(level='ticker')['volume'].shift(10)
        
        # Calculate acceleration (change in momentum)
        data['acceleration'] = data['momentum'] - data.groupby(level='ticker')['momentum'].shift(1)
        
        # Calculate volatility (rolling standard deviation of returns)
        data['volatility'] = data.groupby(level='ticker')['returns'].transform(
            lambda x: x.rolling(window=20, min_periods=1).std()
        ).fillna(0)  # Fill NaN with 0 for first few days
        
        # Calculate volume volatility
        data['volume_volatility'] = data.groupby(level='ticker')['volume'].transform(
            lambda x: x.rolling(window=20, min_periods=1).std()
        ).fillna(0)  # Fill NaN with 0 for first few days
        
        # Calculate price range
        data['price_range'] = data['high'] - data['low']
        data['price_range_pct'] = data['price_range'] / data['close']
        
        # Calculate VWAP using rolling windows
        rolling_money_flow = data.groupby(level='ticker')['money_flow'].transform(
            lambda x: x.rolling(window=20, min_periods=1).sum()
        )
        rolling_volume = data.groupby(level='ticker')['volume'].transform(
            lambda x: x.rolling(window=20, min_periods=1).sum()
        )
        data['vwap'] = (rolling_money_flow / rolling_volume).clip(data['low'], data['high'])
        
        # Calculate relative volume
        data['relative_volume'] = data['volume'] / data.groupby(level='ticker')['volume'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )
        
        return data 