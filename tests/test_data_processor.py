import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_processor.base import DataProcessorBase
from data_processor.processor import DataProcessor

class TestDataProcessorBase(unittest.TestCase):
    """Test the base data processor class."""
    
    class ConcreteDataProcessor(DataProcessorBase):
        """Concrete implementation for testing the base class."""
        def process_data(self, data, factors, start_date=None, end_date=None):
            return data
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = self.ConcreteDataProcessor()
        
    def test_validate_data(self):
        """Test data validation."""
        # Create valid data
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
        
        # Valid data should pass
        self.assertTrue(self.processor.validate_data(df))
        
        # Missing columns should fail
        with self.assertRaises(ValueError):
            self.processor.validate_data(df.drop('close', axis=1))
            
        # Empty DataFrame should fail
        with self.assertRaises(ValueError):
            self.processor.validate_data(pd.DataFrame())
            
        # Invalid date type should fail
        df['date'] = df['date'].astype(str)
        with self.assertRaises(ValueError):
            self.processor.validate_data(df)
            
    def test_validate_factors(self):
        """Test factor validation."""
        # Valid factors
        self.assertTrue(self.processor.validate_factors(['returns', 'volatility', 'rsi']))
        
        # Empty list
        with self.assertRaises(ValueError):
            self.processor.validate_factors([])
            
        # Invalid types
        with self.assertRaises(ValueError):
            self.processor.validate_factors(['returns', 123])
            
        # Empty strings
        with self.assertRaises(ValueError):
            self.processor.validate_factors(['returns', ''])
            
    def test_validate_dates(self):
        """Test date validation."""
        now = datetime.now()
        past = now - timedelta(days=10)
        future = now + timedelta(days=10)
        
        # Valid dates
        self.assertTrue(self.processor.validate_dates(past, now))
        self.assertTrue(self.processor.validate_dates(None, now))
        self.assertTrue(self.processor.validate_dates(past, None))
        self.assertTrue(self.processor.validate_dates(None, None))
        
        # Start after end
        with self.assertRaises(ValueError):
            self.processor.validate_dates(now, past)
            
        # Invalid types
        with self.assertRaises(ValueError):
            self.processor.validate_dates('2023-01-01', now)
            
    def test_get_required_metrics(self):
        """Test required metrics list."""
        metrics = self.processor.get_required_metrics()
        self.assertIsInstance(metrics, list)
        self.assertTrue(len(metrics) > 0)
        self.assertTrue(all(isinstance(m, str) for m in metrics))
        
        # Check for essential metrics
        essential_metrics = ['returns', 'volatility', 'vwap']
        for metric in essential_metrics:
            self.assertIn(metric, metrics)
            
class TestDataProcessor(unittest.TestCase):
    """Test the concrete data processor implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
        self.symbols = ['AAPL', 'MSFT']
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 1, 10)
        
        # Create sample data
        dates = pd.date_range(start=self.start_date, end=self.end_date)
        data = []
        for symbol in self.symbols:
            base_price = 100
            for date in dates:
                # Generate daily price movement
                daily_volatility = 0.02  # 2% daily volatility
                price_change = np.random.normal(0, daily_volatility)
                
                # Calculate OHLC ensuring proper ordering
                base_open = base_price * (1 + price_change)
                intraday_moves = np.random.uniform(-0.01, 0.01, 3)  # 1% max intraday move
                high = base_open * (1 + max(intraday_moves))
                low = base_open * (1 + min(intraday_moves))
                close = base_open * (1 + intraday_moves[-1])
                
                # Ensure OHLC relationship
                high = max(high, base_open, close)
                low = min(low, base_open, close)
                
                # Generate realistic volume (log-normal distribution)
                volume = int(np.exp(np.random.normal(14, 0.5)))  # Centers around 1.2M with positive skew
                
                data.append({
                    'ticker': symbol,
                    'date': date,
                    'open': float(base_open),
                    'high': float(high),
                    'low': float(low),
                    'close': float(close),
                    'volume': volume
                })
                
                # Update base price for next day
                base_price = close
                
        self.data = pd.DataFrame(data)
        
        # Sort by date and ticker for consistency
        self.data = self.data.sort_values(['date', 'ticker']).reset_index(drop=True)
        
    def test_process_data(self):
        """Test data processing."""
        # Process data
        processed_data = self.processor.process_data(
            self.data,
            factors=['returns', 'volatility'],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Check basic validation
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertFalse(processed_data.empty)
        self.assertTrue(self.processor.validate_data(processed_data))
        
        # Check required metrics
        required_metrics = self.processor.get_required_metrics()
        for metric in required_metrics:
            self.assertIn(metric, processed_data.columns)
            
        # Check data cleaning
        self.assertFalse(processed_data.isnull().any().any())  # No missing values
        self.assertEqual(len(processed_data), len(processed_data.drop_duplicates()))  # No duplicates
        
        # Check data organization
        # Data should be sorted by ticker first, then date
        for ticker in processed_data['ticker'].unique():
            ticker_data = processed_data[processed_data['ticker'] == ticker]
            self.assertTrue(ticker_data['date'].is_monotonic_increasing)  # Dates should be increasing within each ticker
        
        # Check metric calculations
        # Returns should be between -1 and 1
        self.assertTrue((processed_data['returns'] >= -1).all())
        self.assertTrue((processed_data['returns'] <= 1).all())
        
        # Volatility should be positive
        self.assertTrue((processed_data['volatility'] >= 0).all())
        
        # VWAP should be between high and low
        self.assertTrue((processed_data['vwap'] >= processed_data['low']).all())
        self.assertTrue((processed_data['vwap'] <= processed_data['high']).all())
        
        # ATR should be positive
        self.assertTrue((processed_data['atr'] >= 0).all())
        
    def test_process_data_with_missing_values(self):
        """Test handling of missing values."""
        # Add some missing values
        data_with_missing = self.data.copy()
        data_with_missing.loc[0, 'close'] = np.nan
        data_with_missing.loc[5, 'volume'] = np.nan
        
        # Process data
        processed_data = self.processor.process_data(
            data_with_missing,
            factors=['returns', 'volatility'],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Check that missing values are handled
        self.assertFalse(processed_data.isnull().any().any())
        
    def test_process_data_with_date_filtering(self):
        """Test date filtering."""
        mid_date = self.start_date + (self.end_date - self.start_date) / 2
        
        # Process data with date filtering
        processed_data = self.processor.process_data(
            self.data,
            factors=['returns', 'volatility'],
            start_date=mid_date,
            end_date=self.end_date
        )
        
        # Check date filtering
        self.assertTrue((processed_data['date'] >= mid_date).all())
        self.assertTrue((processed_data['date'] <= self.end_date).all())
        
    def test_process_data_with_invalid_input(self):
        """Test handling of invalid input."""
        # Invalid data structure
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.processor.process_data(invalid_data, factors=['returns'])
            
        # Invalid date range
        with self.assertRaises(ValueError):
            self.processor.process_data(
                self.data,
                factors=['returns'],
                start_date=self.end_date,
                end_date=self.start_date
            )
            
        # Invalid factors
        with self.assertRaises(ValueError):
            self.processor.process_data(self.data, factors=[])
            
if __name__ == '__main__':
    unittest.main() 