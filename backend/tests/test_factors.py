import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from plugins.factors.ma_factor import MAFactor
from plugins.factors.volume_spike_factor import VolumeSpikeFactor

def create_test_data(n_tickers=2, n_days=100):
    """Create test data for factor testing."""
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    tickers = [f'TICKER_{i}' for i in range(n_tickers)]
    
    data = []
    for ticker in tickers:
        # Generate random price data
        prices = np.random.normal(100, 10, n_days).cumsum() + 1000
        volumes = np.random.normal(1000000, 100000, n_days)
        
        for date, price, volume in zip(dates, prices, volumes):
            data.append({
                'ticker': ticker,
                'date': date,
                'open': price + np.random.normal(0, 1),
                'high': price + abs(np.random.normal(0, 2)),
                'low': price - abs(np.random.normal(0, 2)),
                'close': price,
                'volume': max(volume, 0)  # Ensure positive volume
            })
    
    return pd.DataFrame(data)

class TestMAFactor:
    """Test the Moving Average factor."""
    
    @pytest.fixture
    def factor(self):
        """Create a factor instance for testing."""
        return MAFactor({
            'column': 'close',
            'windows': [5, 20],
            'ma_types': ['sma', 'ema']
        })
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        return create_test_data()
    
    def test_initialization(self, factor):
        """Test factor initialization."""
        assert factor.column == 'close'
        assert factor.windows == [5, 20]
        assert factor.ma_types == ['sma', 'ema']
    
    def test_invalid_window(self):
        """Test invalid window configuration."""
        with pytest.raises(ValueError):
            MAFactor({'windows': []})
    
    def test_calculate(self, factor, data):
        """Test factor calculation."""
        result = factor.calculate(data)
        
        # Check that the result has the expected columns
        expected_columns = ['sma_5', 'sma_20', 'ema_5', 'ema_20']
        for col in expected_columns:
            assert col in result.columns
        
        # Check that the values are not NaN
        for col in expected_columns:
            assert not result[col].isna().all()
    
    def test_calculate_batch(self, factor, data):
        """Test factor calculation with multiple tickers."""
        result = factor.calculate(data)
        
        # Check that we have data for all tickers
        assert len(result['ticker'].unique()) == len(data['ticker'].unique())
    
    def test_performance(self, factor):
        """Test factor calculation performance."""
        data = create_test_data(n_tickers=10, n_days=1000)
        
        import time
        start_time = time.time()
        result = factor.calculate(data)
        end_time = time.time()
        
        # Should complete in less than 1 second
        assert end_time - start_time < 1.0

class TestVolumeSpikeFactor:
    """Test the Volume Spike factor."""
    
    @pytest.fixture
    def factor(self):
        """Create a factor instance for testing."""
        return VolumeSpikeFactor({
            'window': 20,
            'threshold': 2.0
        })
    
    @pytest.fixture
    def data(self):
        """Create test data."""
        return create_test_data()
    
    def test_initialization(self, factor):
        """Test factor initialization."""
        assert factor.window == 20
        assert factor.threshold == 2.0
    
    def test_invalid_config(self):
        """Test invalid configuration."""
        with pytest.raises(ValueError):
            VolumeSpikeFactor({'window': 0})
        
        with pytest.raises(ValueError):
            VolumeSpikeFactor({'threshold': 0})
    
    def test_calculate(self, factor, data):
        """Test factor calculation."""
        result = factor.calculate(data)
        
        # Check that the result has the expected columns
        expected_columns = ['volume_ma', 'volume_std', 'volume_ratio', 'volume_spike']
        for col in expected_columns:
            assert col in result.columns
        
        # Check that the values are not NaN
        for col in expected_columns:
            assert not result[col].isna().all()
        
        # Check that volume_spike is binary
        assert result['volume_spike'].isin([0, 1]).all()
    
    def test_calculate_batch(self, factor, data):
        """Test factor calculation with multiple tickers."""
        result = factor.calculate(data)
        
        # Check that we have data for all tickers
        assert len(result['ticker'].unique()) == len(data['ticker'].unique())
    
    def test_performance(self, factor):
        """Test factor calculation performance."""
        data = create_test_data(n_tickers=10, n_days=1000)
        
        import time
        start_time = time.time()
        result = factor.calculate(data)
        end_time = time.time()
        
        # Should complete in less than 1 second
        assert end_time - start_time < 1.0 