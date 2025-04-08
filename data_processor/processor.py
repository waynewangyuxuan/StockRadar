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
        # Make a copy to avoid modifying input
        data = data.copy()
        
        # Validate inputs
        self.validate_data(data)
        self.validate_factors(factors)
        self.validate_dates(start_date, end_date)
        
        # Filter data by date range if specified
        if start_date is not None:
            data = data[data['date'] >= start_date]
        if end_date is not None:
            data = data[data['date'] <= end_date]
            
        # Sort data
        data = data.sort_values(['ticker', 'date'])
        
        # Handle missing values in required columns first
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        data = self._handle_missing_values(data, required_cols)
        
        # Calculate basic metrics
        data = self._calculate_basic_metrics(data)
        
        # Fill any remaining NaN values in calculated metrics
        calculated_cols = [
            'returns', 'log_returns', 'price_change', 'tr', 'atr',
            'typical_price', 'money_flow', 'momentum', 'volume_momentum',
            'acceleration', 'volatility', 'volume_volatility',
            'price_range', 'price_range_pct', 'vwap', 'relative_volume'
        ]
        data = self._handle_missing_values(data, calculated_cols, fill_only=True)
        
        # Verify no missing values remain
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            self.logger.warning(f"Found missing values in columns: {missing_cols}")
            # Fill any remaining NaN values with 0 for calculated metrics
            data[calculated_cols] = data[calculated_cols].fillna(0)
        
        return data
        
    def _handle_missing_values(self, 
                             data: pd.DataFrame,
                             columns: List[str],
                             fill_only: bool = False) -> pd.DataFrame:
        """Handle missing values in specified columns.
        
        Args:
            data: DataFrame with market data
            columns: List of columns to handle missing values for
            fill_only: If True, only fill missing values without dropping rows
            
        Returns:
            DataFrame with handled missing values
        """
        # Forward fill within each ticker group
        for col in columns:
            if col in data.columns:
                data[col] = data.groupby('ticker')[col].ffill()
                data[col] = data.groupby('ticker')[col].bfill()
        
        # Drop rows with missing values in specified columns if not in fill_only mode
        if not fill_only:
            data = data.dropna(subset=columns)
        
        return data
        
    def _calculate_basic_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic market metrics that are commonly used.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with additional market metrics
        """
        # Calculate metrics for each ticker separately
        for ticker in data['ticker'].unique():
            mask = data['ticker'] == ticker
            
            # Calculate returns and price changes
            data.loc[mask, 'returns'] = data.loc[mask, 'close'].pct_change()
            data.loc[mask, 'log_returns'] = np.log(data.loc[mask, 'close']).diff()
            data.loc[mask, 'price_change'] = data.loc[mask, 'close'].diff()
            
            # Calculate true range
            prev_close = data.loc[mask, 'close'].shift(1)
            data.loc[mask, 'tr'] = pd.DataFrame({
                'hl': data.loc[mask, 'high'] - data.loc[mask, 'low'],
                'hc': abs(data.loc[mask, 'high'] - prev_close),
                'lc': abs(data.loc[mask, 'low'] - prev_close)
            }).max(axis=1)
            
            # Calculate ATR with minimum periods
            data.loc[mask, 'atr'] = data.loc[mask, 'tr'].rolling(
                window=14,
                min_periods=1
            ).mean()
            
            # Calculate typical price and money flow
            data.loc[mask, 'typical_price'] = (
                data.loc[mask, ['high', 'low', 'close']].mean(axis=1)
            )
            data.loc[mask, 'money_flow'] = (
                data.loc[mask, 'typical_price'] * data.loc[mask, 'volume']
            )
            
            # Calculate momentum and acceleration
            data.loc[mask, 'momentum'] = data.loc[mask, 'close'].diff(10)
            data.loc[mask, 'volume_momentum'] = data.loc[mask, 'volume'].diff(10)
            data.loc[mask, 'acceleration'] = data.loc[mask, 'momentum'].diff()
            
            # Calculate volatilities
            data.loc[mask, 'volatility'] = data.loc[mask, 'returns'].rolling(
                window=20,
                min_periods=1
            ).std()
            data.loc[mask, 'volume_volatility'] = data.loc[mask, 'volume'].rolling(
                window=20,
                min_periods=1
            ).std()
            
            # Calculate price ranges
            data.loc[mask, 'price_range'] = data.loc[mask, 'high'] - data.loc[mask, 'low']
            data.loc[mask, 'price_range_pct'] = data.loc[mask, 'price_range'] / data.loc[mask, 'close']
            
            # Calculate VWAP (daily)
            data.loc[mask, 'vwap'] = (
                data.loc[mask, 'money_flow'] / data.loc[mask, 'volume']
            )
            
            # Calculate relative volume
            data.loc[mask, 'relative_volume'] = data.loc[mask, 'volume'] / data.loc[mask, 'volume'].rolling(
                window=20,
                min_periods=1
            ).mean()
        
        return data 