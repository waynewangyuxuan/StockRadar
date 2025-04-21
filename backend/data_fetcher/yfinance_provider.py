import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
import time
import logging
from .base import DataProviderBase

class YahooFinanceProvider(DataProviderBase):
    """Yahoo Finance data provider."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Yahoo Finance provider.
        
        Args:
            config: Dictionary containing provider-specific configuration:
                   - retry_count: Number of retries for failed requests (default: 3)
                   - retry_delay: Delay between retries in seconds (default: 1)
                   - timeout: Request timeout in seconds (default: 10)
        """
        super().__init__(config)
        self.retry_count = self.config.get('retry_count', 3)
        self.retry_delay = self.config.get('retry_delay', 1)
        self.timeout = self.config.get('timeout', 10)
        self.logger = logging.getLogger(__name__)
        
    def fetch(self, 
              symbols: List[str],
              start_date: datetime,
              end_date: datetime,
              interval: str = "1d") -> pd.DataFrame:
        """Fetch market data for the specified symbols and time range.
        
        This method is a wrapper around fetch_historical_data to satisfy
        the DataProviderBase abstract class requirement.
        
        Args:
            symbols: List of stock symbols to fetch
            start_date: Start date for the data
            end_date: End date for the data
            interval: Data frequency (e.g., "1d", "1h", "1m")
            
        Returns:
            DataFrame with market data
        """
        return self.fetch_historical_data(symbols, start_date, end_date, interval)
        
    def fetch_historical_data(self, 
                            symbols: List[str],
                            start_date: datetime,
                            end_date: datetime,
                            interval: str = "1d") -> pd.DataFrame:
        """Fetch historical market data from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols to fetch
            start_date: Start date for the data
            end_date: End date for the data
            interval: Data frequency (e.g., "1d", "1h", "1m")
            
        Returns:
            DataFrame with market data
        """
        # Validate inputs
        self.validate_symbols(symbols)
        self.validate_dates(start_date, end_date)
        self.validate_interval(interval)
        
        self.logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Fetch data for each symbol with retries
        dfs = []
        for symbol in symbols:
            df = self._fetch_symbol(symbol, start_date, end_date, interval)
            if df is not None and not df.empty:
                df['ticker'] = symbol
                dfs.append(df)
                
        if not dfs:
            raise ValueError("No data available for any of the requested symbols")
            
        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Standardize the DataFrame
        return self._standardize_dataframe(combined_df)
        
    def fetch_live_data(self,
                       symbols: List[str],
                       interval: str = "1m") -> pd.DataFrame:
        """Fetch latest market data from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols to fetch
            interval: Data frequency (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m")
            
        Returns:
            DataFrame with latest market data
        """
        self.validate_symbols(symbols)
        self.validate_interval(interval)
        
        # For live data, we fetch the last few intervals to ensure we have the latest data
        # Yahoo Finance has different limitations for different intervals
        interval_limits = {
            "1m": 7,      # 7 days
            "2m": 60,     # 60 days
            "5m": 60,     # 60 days
            "15m": 60,    # 60 days
            "30m": 60,    # 60 days
            "60m": 730,   # 730 days (2 years)
            "90m": 60     # 60 days
        }
        
        # Calculate start date based on interval - ensure datetime objects
        days_limit = interval_limits.get(interval, 7)
        start_date = datetime.now() - timedelta(days=days_limit)
        end_date = datetime.now()
        
        # Validate dates before proceeding - this performs type checking too
        self.validate_dates(start_date, end_date)
        
        self.logger.info(f"Fetching live data for {len(symbols)} symbols")
        
        # Fetch recent data for each symbol
        dfs = []
        for symbol in symbols:
            try:
                df = self._fetch_symbol(symbol, start_date, end_date, interval)
                if df is not None and not df.empty:
                    # Get only the latest data point
                    latest_data = df.iloc[[-1]].copy()  # Create a copy to avoid SettingWithCopyWarning
                    latest_data.loc[:, 'ticker'] = symbol  # Use .loc to set values
                    dfs.append(latest_data)
            except Exception as e:
                self.logger.warning(f"Failed to fetch live data for {symbol}: {str(e)}")
                
        if not dfs:
            raise ValueError("No live data available for any of the requested symbols")
            
        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Standardize the DataFrame
        return self._standardize_dataframe(combined_df)
        
    def _fetch_symbol(self,
                     symbol: str,
                     start_date: datetime,
                     end_date: datetime,
                     interval: str) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol with retry logic.
        
        Args:
            symbol: Stock symbol to fetch
            start_date: Start date for the data
            end_date: End date for the data
            interval: Data frequency
            
        Returns:
            DataFrame with market data or None if all retries fail
        """
        # Ensure we have proper datetime objects
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            self.logger.error(f"Invalid date types: start_date={type(start_date)}, end_date={type(end_date)}")
            raise ValueError("Both start_date and end_date must be datetime objects")
        
        for attempt in range(self.retry_count + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    timeout=self.timeout
                )
                
                if df.empty:
                    raise ValueError(f"No data available for symbol {symbol}")
                    
                # Reset index to make date a column
                df = df.reset_index()
                
                # Handle column name changes in yfinance
                df.columns = df.columns.str.lower()
                
                # Check if 'date' column exists, otherwise look for 'datetime' or 'time'
                if 'date' not in df.columns:
                    if 'datetime' in df.columns:
                        df = df.rename(columns={'datetime': 'date'})
                    elif 'time' in df.columns:
                        df = df.rename(columns={'time': 'date'})
                    else:
                        # If no date-like column is found, use the index
                        df = df.reset_index()
                        df = df.rename(columns={'index': 'date'})
                
                # Handle other column name changes
                if 'stock splits' in df.columns:
                    df = df.rename(columns={'stock splits': 'splits'})
                
                # Ensure we have all required columns
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                return df[required_cols]
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < self.retry_count:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to fetch data for {symbol} after {self.retry_count} retries")
                    return None
                    
        return None
        
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the DataFrame format.
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            Standardized DataFrame with MultiIndex (date, ticker)
        """
        # Ensure we have the required columns
        required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Add provider and timestamp columns
        df['provider'] = self.__class__.__name__
        df['timestamp'] = pd.Timestamp.now()
        
        # Set MultiIndex
        df = df.set_index(['date', 'ticker'])
        
        # Sort index
        df = df.sort_index()
        
        # Select and order columns
        df = df[['open', 'high', 'low', 'close', 'volume', 'provider', 'timestamp']]
        
        return df
        
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get additional information about a stock symbol.
        
        Args:
            symbol: Stock symbol to get info for
            
        Returns:
            Dictionary containing symbol information
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'symbol': symbol,
            'name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD'),
            'timezone': info.get('exchangeTimezoneName', 'America/New_York')
        }
        
    def get_sp500_symbols(self) -> List[str]:
        """Get list of S&P 500 symbols.
        
        Returns:
            List of S&P 500 stock symbols
        """
        # This is a placeholder. In a real implementation, you would:
        # 1. Either maintain a local list that's updated periodically
        # 2. Or fetch from a reliable source (e.g., Wikipedia API)
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Example symbols
        
    def get_nasdaq100_symbols(self) -> List[str]:
        """Get list of NASDAQ 100 symbols.
        
        Returns:
            List of NASDAQ 100 stock symbols
        """
        # This is a placeholder. In a real implementation, you would:
        # 1. Either maintain a local list that's updated periodically
        # 2. Or fetch from a reliable source
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Example symbols

    def _ensure_datetime(self, date_value) -> datetime:
        """Ensure a value is a proper datetime object.
        
        Args:
            date_value: Value to convert to datetime
            
        Returns:
            datetime object
        """
        if isinstance(date_value, datetime):
            return date_value
        
        try:
            # Try to convert string to datetime
            if isinstance(date_value, str):
                return pd.to_datetime(date_value).to_pydatetime()
            
            # Try to convert timestamp to datetime
            if isinstance(date_value, pd.Timestamp):
                return date_value.to_pydatetime()
            
            # Try to convert date to datetime
            if isinstance(date_value, date):
                return datetime.combine(date_value, datetime.min.time())
            
            # Try to convert numeric value to datetime
            if isinstance(date_value, (int, float)):
                return datetime.fromtimestamp(date_value)
            
            raise ValueError(f"Cannot convert {type(date_value)} to datetime")
        except Exception as e:
            self.logger.error(f"Failed to convert {date_value} to datetime: {str(e)}")
            raise ValueError(f"Invalid date value: {date_value}")

    def validate_dates(self, start_date, end_date) -> bool:
        """Validate the date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # First ensure we have proper datetime objects
        try:
            start_date = self._ensure_datetime(start_date)
            end_date = self._ensure_datetime(end_date)
        except ValueError as e:
            self.logger.error(f"Date validation error: {str(e)}")
            raise
        
        # Now validate with base class method
        return super().validate_dates(start_date, end_date)
