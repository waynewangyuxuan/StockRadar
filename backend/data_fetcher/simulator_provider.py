import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from .base import DataProviderBase

class SimulatorProvider(DataProviderBase):
    """Simulator data provider for testing.
    
    This provider generates synthetic market data with configurable parameters.
    It's useful for testing strategies and factors without real market data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the simulator provider.
        
        Args:
            config: Dictionary containing provider-specific configuration:
                   - volatility: Price volatility (default: 0.02)
                   - trend: Price trend (-1 for down, 0 for flat, 1 for up) (default: 0)
                   - volume_scale: Volume scaling factor (default: 1000000)
                   - random_seed: Random seed for reproducibility (default: 42)
        """
        super().__init__(config)
        self.volatility = self.config.get('volatility', 0.02)
        self.trend = self.config.get('trend', 0)
        self.volume_scale = self.config.get('volume_scale', 1000000)
        self.random_seed = self.config.get('random_seed', 42)
        
    def fetch(self, 
              symbols: List[str],
              start_date: datetime,
              end_date: datetime,
              interval: str = "1d") -> pd.DataFrame:
        """Generate synthetic market data.
        
        Args:
            symbols: List of stock symbols to simulate
            start_date: Start date for the data
            end_date: End date for the data
            interval: Data frequency (e.g., "1d", "1h", "1m")
            
        Returns:
            DataFrame with synthetic market data
        """
        # Validate inputs
        self.validate_symbols(symbols)
        self.validate_dates(start_date, end_date)
        self.validate_interval(interval)
        
        self.logger.info(f"Generating synthetic data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Reset random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Generate data for each symbol
        dfs = []
        for symbol in symbols:
            df = self._generate_symbol_data(symbol, start_date, end_date, interval)
            dfs.append(df)
            
        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Standardize the DataFrame
        return self._standardize_dataframe(combined_df)
        
    def _generate_symbol_data(self,
                            symbol: str,
                            start_date: datetime,
                            end_date: datetime,
                            interval: str) -> pd.DataFrame:
        """Generate synthetic data for a single symbol.
        
        Args:
            symbol: Stock symbol to generate data for
            start_date: Start date for the data
            end_date: End date for the data
            interval: Data frequency
            
        Returns:
            DataFrame with synthetic market data
        """
        # Calculate number of periods
        delta = end_date - start_date
        if interval == "1d":
            periods = delta.days + 1
        elif interval == "1h":
            periods = delta.days * 24 + 1
        elif interval == "1m":
            periods = delta.days * 24 * 60 + 1
        else:
            raise ValueError(f"Unsupported interval: {interval}")
            
        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, freq=interval)
        
        # Generate random walk for prices with trend
        drift = self.trend * 0.001  # Small daily drift based on trend
        returns = np.random.normal(
            loc=drift,
            scale=self.volatility,
            size=periods
        )
        
        # Calculate cumulative returns
        cum_returns = np.cumsum(returns)
        
        # Add trend component
        trend_component = np.linspace(0, self.trend * 0.1, periods)  # Linear trend
        cum_returns += trend_component
        
        # Calculate prices starting at $100
        prices = 100 * np.exp(cum_returns)
        
        # Generate volumes with lognormal distribution
        volumes = np.random.lognormal(
            mean=np.log(self.volume_scale),
            sigma=0.5,
            size=periods
        ).astype(int)
        
        # Create price variations around close
        spreads = np.abs(np.random.normal(0, self.volatility, periods))
        highs = prices * (1 + spreads)
        lows = prices * (1 - spreads)
        opens = prices + np.random.normal(0, self.volatility, periods)
        
        # Ensure price relationships are maintained
        opens = np.clip(opens, lows, highs)
        
        # Create DataFrame
        df = pd.DataFrame({
            'ticker': symbol,
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
        
        return df
        
    def set_random_seed(self, seed: int) -> None:
        """Set a new random seed for reproducibility.
        
        Args:
            seed: New random seed
        """
        self.random_seed = seed
