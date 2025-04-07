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
        
        # Create sample data with controlled price changes
        dates = pd.date_range(start=self.start_date, end=self.end_date)
        data = []
        
        for symbol in self.symbols:
            # Start with base price and add small random changes
            base_price = 100.0
            for date in dates:
                # Random price change between -5% and +5%
                price_change_pct = np.random.uniform(-0.05, 0.05)
                close = base_price * (1 + price_change_pct)
                
                data.append({
                    'ticker': symbol,
                    'date': date,
                    'open': close * (1 - np.random.uniform(0, 0.01)),  # Slightly lower
                    'high': close * (1 + np.random.uniform(0, 0.02)),  # Slightly higher
                    'low': close * (1 - np.random.uniform(0, 0.02)),   # Slightly lower
                    'close': close,
                    'volume': int(1000000 * (1 + np.random.uniform(-0.1, 0.1)))  # ±10% volume variation
                })
                
                # Update base price for next day
                base_price = close
                
        self.data = pd.DataFrame(data)
        
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
            
        # Check data organization
        self.assertTrue(processed_data['date'].is_monotonic_increasing)  # Sorted by date
        
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