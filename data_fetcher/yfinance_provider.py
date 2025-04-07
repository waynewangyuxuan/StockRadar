import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
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
        
    def fetch(self, 
              symbols: List[str],
              start_date: datetime,
              end_date: datetime,
              interval: str = "1d") -> pd.DataFrame:
        """Fetch market data from Yahoo Finance.
        
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
        
        self.logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
        
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
                df.columns = df.columns.str.lower()
                df = df.rename(columns={'date': 'date', 'stock splits': 'splits'})
                
                return df[['date', 'open', 'high', 'low', 'close', 'volume']]
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < self.retry_count:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"Failed to fetch data for {symbol} after {self.retry_count} retries")
                    
        return None
        
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
