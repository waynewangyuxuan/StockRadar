from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

class DataProviderBase(ABC):
    """Base class for all data providers in StockRadar.
    
    This abstract class defines the interface that all data providers must implement.
    It ensures consistent data format and behavior across different data sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data provider with configuration.
        
        Args:
            config: Dictionary containing provider-specific configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def fetch(self, 
              symbols: List[str],
              start_date: datetime,
              end_date: datetime,
              interval: str = "1d") -> pd.DataFrame:
        """Fetch market data for the specified symbols and time range.
        
        Args:
            symbols: List of stock symbols to fetch
            start_date: Start date for the data
            end_date: End date for the data
            interval: Data frequency (e.g., "1d", "1h", "1m")
            
        Returns:
            DataFrame with the following columns:
            - ticker: Stock symbol
            - date: Timestamp
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume
            - provider: Name of the data provider
            - timestamp: When the data was fetched
            
        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If data source is unreachable
            DataError: If data is invalid or missing
        """
        pass
    
    @abstractmethod
    def fetch_historical_data(self, 
                             symbols: List[str],
                             start_date: datetime,
                             end_date: datetime,
                             interval: str = "1d") -> pd.DataFrame:
        """Fetch historical market data for the specified symbols and time range.
        
        Args:
            symbols: List of stock symbols to fetch
            start_date: Start date for the data
            end_date: End date for the data
            interval: Data frequency (e.g., "1d", "1h", "1m")
            
        Returns:
            DataFrame with MultiIndex (date, ticker) and columns:
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume
            
        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If data source is unreachable
            DataError: If data is invalid or missing
        """
        pass
    
    @abstractmethod
    def fetch_live_data(self,
                       symbols: List[str],
                       interval: str = "1m") -> pd.DataFrame:
        """Fetch latest market data for the specified symbols.
        
        Args:
            symbols: List of stock symbols to fetch
            interval: Data frequency (e.g., "1m", "5m", "15m")
            
        Returns:
            DataFrame with MultiIndex (date, ticker) and columns:
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume
            
        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If data source is unreachable
            DataError: If data is invalid or missing
        """
        pass
    
    def validate_symbols(self, symbols: List[str]) -> bool:
        """Validate the list of symbols.
        
        Args:
            symbols: List of stock symbols to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not symbols:
            raise ValueError("Symbols list cannot be empty")
            
        if not all(isinstance(s, str) and s.strip() for s in symbols):
            raise ValueError("All symbols must be non-empty strings")
            
        return True
    
    def validate_dates(self, start_date: datetime, end_date: datetime) -> bool:
        """Validate the date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise ValueError("Dates must be datetime objects") 
            
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
            
        if end_date > datetime.now():
            raise ValueError("End date cannot be in the future")
            
        return True
    
    def validate_interval(self, interval: str) -> bool:
        """Validate the data interval.
        
        Args:
            interval: Data frequency (e.g., "1d", "1h", "1m")
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        valid_intervals = ['1m', '5m', '15m', '30m', '1h', '1d', '1w', '1mo']
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {valid_intervals}")
        return True
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the DataFrame format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Standardized DataFrame with MultiIndex (date, ticker)
        """
        # Ensure required columns exist
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Add provider and timestamp columns
        df['provider'] = self.__class__.__name__
        df['timestamp'] = pd.Timestamp.now()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Set MultiIndex (date, ticker)
        df = df.set_index(['date', 'ticker'])
        
        # Sort index for faster access
        df = df.sort_index()
        
        # Select and order columns
        df = df[['open', 'high', 'low', 'close', 'volume', 'provider', 'timestamp']]
        
        return df
    
    def __str__(self) -> str:
        """String representation of the data provider."""
        return f"{self.__class__.__name__}(config={self.config})"
