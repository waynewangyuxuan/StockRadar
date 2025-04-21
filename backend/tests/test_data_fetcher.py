import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_fetcher.base import DataProviderBase
from data_fetcher.yfinance_provider import YahooFinanceProvider
from data_fetcher.simulator_provider import SimulatorProvider

class TestDataProvider(DataProviderBase):
    """Concrete implementation of DataProviderBase for testing."""
    
    def fetch(self, symbols, start_date, end_date, interval="1d"):
        """Dummy implementation of fetch method."""
        return pd.DataFrame()

class TestDataProviderBase(unittest.TestCase):
    """Test the base data provider class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.provider = TestDataProvider()
        
    def test_validate_symbols(self):
        """Test symbol validation."""
        # Valid symbols
        self.assertTrue(self.provider.validate_symbols(['AAPL', 'MSFT', 'GOOGL']))
        
        # Empty list
        with self.assertRaises(ValueError):
            self.provider.validate_symbols([])
            
        # Invalid types
        with self.assertRaises(ValueError):
            self.provider.validate_symbols(['AAPL', 123])
            
        # Empty strings
        with self.assertRaises(ValueError):
            self.provider.validate_symbols(['AAPL', ''])
            
    def test_validate_dates(self):
        """Test date validation."""
        now = datetime.now()
        past = now - timedelta(days=10)
        future = now + timedelta(days=10)
        
        # Valid dates
        self.assertTrue(self.provider.validate_dates(past, now))
        
        # Start after end
        with self.assertRaises(ValueError):
            self.provider.validate_dates(now, past)
            
        # Future end date
        with self.assertRaises(ValueError):
            self.provider.validate_dates(past, future)
            
        # Invalid types
        with self.assertRaises(ValueError):
            self.provider.validate_dates('2023-01-01', now)
            
    def test_validate_interval(self):
        """Test interval validation."""
        # Valid intervals
        valid_intervals = ['1m', '5m', '15m', '30m', '1h', '1d', '1w', '1mo']
        for interval in valid_intervals:
            self.assertTrue(self.provider.validate_interval(interval))
            
        # Invalid interval
        with self.assertRaises(ValueError):
            self.provider.validate_interval('invalid')
            
    def test_standardize_dataframe(self):
        """Test DataFrame standardization."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=3)
        df = pd.DataFrame({
            'ticker': ['AAPL'] * 3,
            'date': dates,
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000000, 1100000, 1200000]
        })
        
        # Standardize
        std_df = self.provider._standardize_dataframe(df)
        
        # Check columns
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'provider', 'timestamp']
        self.assertEqual(set(std_df.columns), set(required_cols))
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(std_df['date']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(std_df['timestamp']))
        
        # Check sorting
        self.assertTrue(std_df['date'].is_monotonic_increasing)
        
class TestSimulatorProvider(unittest.TestCase):
    """Test the simulator data provider."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.provider = SimulatorProvider({})  # Pass empty dict instead of None
        self.symbols = ['AAPL', 'MSFT']
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 10)
        
    def test_fetch(self):
        """Test data generation."""
        df = self.provider.fetch(self.symbols, self.start_date, self.end_date)
        
        # Check basic structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df['ticker'].unique()), len(self.symbols))
        self.assertEqual(len(df['date'].unique()), 10)  # 10 days
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(df['open']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['high']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['low']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['close']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['volume']))
        
        # Check price relationships
        self.assertTrue((df['high'] >= df['open']).all())
        self.assertTrue((df['high'] >= df['close']).all())
        self.assertTrue((df['low'] <= df['open']).all())
        self.assertTrue((df['low'] <= df['close']).all())
        
    def test_reproducibility(self):
        """Test that data generation is reproducible."""
        # Generate data twice with same seed
        df1 = self.provider.fetch(self.symbols, self.start_date, self.end_date)
        df2 = self.provider.fetch(self.symbols, self.start_date, self.end_date)
        
        # Drop timestamp column as it will be different
        df1 = df1.drop('timestamp', axis=1)
        df2 = df2.drop('timestamp', axis=1)
        
        # Data should be identical
        pd.testing.assert_frame_equal(df1, df2)
        
        # Change seed and generate again
        self.provider.set_random_seed(43)
        df3 = self.provider.fetch(self.symbols, self.start_date, self.end_date)
        df3 = df3.drop('timestamp', axis=1)
        
        # Data should be different
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(df1, df3)
            
    def test_trend(self):
        """Test price trend generation."""
        # Generate data with upward trend
        up_provider = SimulatorProvider({'trend': 1})
        up_df = up_provider.fetch(self.symbols, self.start_date, self.end_date)
        
        # Generate data with downward trend
        down_provider = SimulatorProvider({'trend': -1})
        down_df = down_provider.fetch(self.symbols, self.start_date, self.end_date)
        
        # Check trend direction
        up_trend = up_df.groupby('ticker')['close'].apply(lambda x: x.iloc[-1] > x.iloc[0]).all()
        down_trend = down_df.groupby('ticker')['close'].apply(lambda x: x.iloc[-1] < x.iloc[0]).all()
        
        self.assertTrue(up_trend)
        self.assertTrue(down_trend)
        
    def test_volatility(self):
        """Test volatility parameter."""
        # Generate data with different volatility levels
        low_vol = SimulatorProvider({'volatility': 0.01})
        high_vol = SimulatorProvider({'volatility': 0.05})
        
        low_df = low_vol.fetch(self.symbols, self.start_date, self.end_date)
        high_df = high_vol.fetch(self.symbols, self.start_date, self.end_date)
        
        # Calculate price changes
        low_changes = low_df.groupby('ticker')['close'].pct_change().std()
        high_changes = high_df.groupby('ticker')['close'].pct_change().std()
        
        # High volatility should have larger price changes
        self.assertTrue(high_changes.mean() > low_changes.mean())

class TestYahooFinanceProvider(unittest.TestCase):
    """Test the Yahoo Finance data provider."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.provider = YahooFinanceProvider({})  # Pass empty dict instead of None
        self.symbols = ['AAPL', 'MSFT']
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 10)
        
    def test_fetch(self):
        """Test data fetching."""
        df = self.provider.fetch(self.symbols, self.start_date, self.end_date)
        
        # Check basic structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df['ticker'].unique()), len(self.symbols))
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(df['open']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['high']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['low']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['close']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['volume']))
        
        # Check price relationships
        self.assertTrue((df['high'] >= df['open']).all())
        self.assertTrue((df['high'] >= df['close']).all())
        self.assertTrue((df['low'] <= df['open']).all())
        self.assertTrue((df['low'] <= df['close']).all())
        
    def test_get_symbol_info(self):
        """Test symbol information retrieval."""
        info = self.provider.get_symbol_info('AAPL')
        
        # Check required fields
        self.assertIn('symbol', info)
        self.assertIn('name', info)
        self.assertIn('sector', info)
        self.assertIn('industry', info)
        self.assertIn('market_cap', info)
        self.assertIn('currency', info)
        self.assertIn('timezone', info)
        
        # Check data types
        self.assertIsInstance(info['symbol'], str)
        self.assertIsInstance(info['name'], str)
        self.assertIsInstance(info['market_cap'], (int, float))
        
    def test_retry_mechanism(self):
        """Test retry mechanism for failed requests."""
        # Create provider with short timeout to force retries
        provider = YahooFinanceProvider({
            'timeout': 0.001,
            'retry_count': 2,
            'retry_delay': 0.1
        })
        
        # This should raise an exception after retries
        with self.assertRaises(Exception):
            provider.fetch(self.symbols, self.start_date, self.end_date)

if __name__ == '__main__':
    unittest.main() 