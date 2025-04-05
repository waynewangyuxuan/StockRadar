import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import DataProcessor

class MarketDataCleaner(DataProcessor):
    """Processor for cleaning and normalizing market data"""
    
    def __init__(
        self,
        name: str = "market_data_cleaner",
        description: Optional[str] = None,
        enabled: bool = True,
        required_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        date_column: str = "Date",
        symbol_column: str = "symbol",
        fill_method: str = "ffill",
        remove_outliers: bool = True,
        outlier_std_threshold: float = 3.0
    ):
        """Initialize market data cleaner
        
        Args:
            name: Name of the cleaner
            description: Optional description
            enabled: Whether the cleaner is enabled
            required_columns: List of required columns
            numeric_columns: List of numeric columns to process
            date_column: Name of the date column
            symbol_column: Name of the symbol column
            fill_method: Method to fill missing values
            remove_outliers: Whether to remove outliers
            outlier_std_threshold: Standard deviation threshold for outliers
        """
        super().__init__(name, description, enabled)
        self.required_columns = required_columns or [
            "Date", "symbol", "open", "high", "low", "close", "volume"
        ]
        self.numeric_columns = numeric_columns or [
            "open", "high", "low", "close", "volume"
        ]
        self.date_column = date_column
        self.symbol_column = symbol_column
        self.fill_method = fill_method
        self.remove_outliers = remove_outliers
        self.outlier_std_threshold = outlier_std_threshold
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate market data
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data[self.date_column]):
            raise ValueError(f"Date column '{self.date_column}' must be datetime type")
        
        # Check for negative values in numeric columns
        for col in self.numeric_columns:
            if (data[col] < 0).any():
                raise ValueError(f"Negative values found in column '{col}'")
        
        # Check for high > low
        if (data['high'] < data['low']).any():
            raise ValueError("High price cannot be less than low price")
        
        return True
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process market data
        
        Args:
            data: Input DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        start_time = datetime.now()
        
        try:
            # Create a copy to avoid modifying the original
            df = data.copy()
            
            # Sort by date and symbol
            df = df.sort_values([self.date_column, self.symbol_column])
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Remove outliers if enabled
            if self.remove_outliers:
                df = self._remove_outliers(df)
            
            # Calculate additional metrics
            df = self._calculate_metrics(df)
            
            # Update metrics
            self.metrics['processed_rows'] = len(df)
            self.metrics['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            return df
            
        except Exception as e:
            self.metrics['errors'] += 1
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # Forward fill missing values within each symbol
        df = df.groupby(self.symbol_column).fillna(method=self.fill_method)
        
        # Backward fill any remaining missing values
        df = df.fillna(method='bfill')
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from numeric columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers removed
        """
        for col in self.numeric_columns:
            # Calculate mean and standard deviation for each symbol
            stats = df.groupby(self.symbol_column)[col].agg(['mean', 'std'])
            
            # Create mask for outliers
            mask = pd.Series(True, index=df.index)
            for symbol in stats.index:
                symbol_mask = df[self.symbol_column] == symbol
                mean = stats.loc[symbol, 'mean']
                std = stats.loc[symbol, 'std']
                
                # Mark values outside threshold as outliers
                mask[symbol_mask] = (
                    (df.loc[symbol_mask, col] >= mean - self.outlier_std_threshold * std) &
                    (df.loc[symbol_mask, col] <= mean + self.outlier_std_threshold * std)
                )
            
            # Replace outliers with NaN
            df.loc[~mask, col] = np.nan
        
        # Handle the NaN values created by outlier removal
        df = self._handle_missing_values(df)
        
        return df
    
    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional market metrics
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional metrics
        """
        # Calculate daily returns
        df['daily_return'] = df.groupby(self.symbol_column)['close'].pct_change()
        
        # Calculate price changes
        df['price_change'] = df['close'] - df['open']
        
        # Calculate trading range
        df['trading_range'] = df['high'] - df['low']
        
        # Calculate volume-weighted average price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df 