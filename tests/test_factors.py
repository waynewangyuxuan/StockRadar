import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from plugins.factors import MovingAverageFactor, VolumeSpikeFactor

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

class TestMovingAverageFactor:
    @pytest.fixture
    def factor(self):
        return MovingAverageFactor(config={'window': 20})
    
    @pytest.fixture
    def data(self):
        return create_test_data()
    
    def test_initialization(self, factor):
        """Test factor initialization."""
        assert factor.window == 20
        assert factor.price_col == 'close'
        
    def test_invalid_window(self):
        """Test invalid window size."""
        with pytest.raises(ValueError):
            MovingAverageFactor(config={'window': 0})
    
    def test_calculate(self, factor, data):
        """Test basic calculation."""
        result = factor.calculate(data)
        
        # Check output column exists
        assert f'ma_{factor.window}' in result.columns
        
        # Check first window-1 values are NaN
        assert result[f'ma_{factor.window}'].iloc[:factor.window-1].isna().all()
        
        # Check values are within price range
        price_col = factor.price_col
        assert result[f'ma_{factor.window}'].min() >= result[price_col].min()
        assert result[f'ma_{factor.window}'].max() <= result[price_col].max()
        
        # Check calculation is correct for one ticker
        ticker_data = result[result['ticker'] == 'TICKER_0']
        manual_ma = ticker_data[price_col].rolling(window=factor.window).mean()
        pd.testing.assert_series_equal(
            ticker_data[f'ma_{factor.window}'],
            manual_ma,
            check_names=False
        )
    
    def test_calculate_batch(self, factor, data):
        """Test batch calculation."""
        result = factor.calculate_batch(data)
        
        # Check output matches single calculation
        single_result = factor.calculate(data)
        pd.testing.assert_frame_equal(result, single_result)
    
    def test_performance(self, factor):
        """Test performance with large dataset."""
        large_data = create_test_data(n_tickers=10, n_days=1000)
        
        import time
        start_time = time.time()
        result = factor.calculate(large_data)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 1.0  # 1 second

class TestVolumeSpikeFactor:
    @pytest.fixture
    def factor(self):
        return VolumeSpikeFactor(config={'window': 20, 'threshold': 2.0})
    
    @pytest.fixture
    def data(self):
        return create_test_data()
    
    def test_initialization(self, factor):
        """Test factor initialization."""
        assert factor.window == 20
        assert factor.threshold == 2.0
        
    def test_invalid_config(self):
        """Test invalid configuration."""
        with pytest.raises(ValueError):
            VolumeSpikeFactor(config={'window': 0})
        with pytest.raises(ValueError):
            VolumeSpikeFactor(config={'threshold': 0})
    
    def test_calculate(self, factor, data):
        """Test basic calculation."""
        result = factor.calculate(data)
        
        # Check output column exists
        assert 'volume_spike' in result.columns
        
        # Check first window-1 values are False
        assert not result['volume_spike'].iloc[:factor.window-1].any()
        
        # Check values are boolean
        assert result['volume_spike'].dtype == bool
        
        # Check calculation is correct for one ticker
        ticker_data = result[result['ticker'] == 'TICKER_0']
        volume = ticker_data['volume']
        
        # Manual calculation for verification
        rolling_mean = volume.rolling(window=factor.window).mean()
        rolling_std = volume.rolling(window=factor.window).std()
        manual_spikes = volume > (rolling_mean + factor.threshold * rolling_std)
        
        pd.testing.assert_series_equal(
            ticker_data['volume_spike'],
            manual_spikes,
            check_names=False
        )
    
    def test_calculate_batch(self, factor, data):
        """Test batch calculation."""
        result = factor.calculate_batch(data)
        
        # Check output matches single calculation
        single_result = factor.calculate(data)
        pd.testing.assert_frame_equal(result, single_result)
    
    def test_performance(self, factor):
        """Test performance with large dataset."""
        large_data = create_test_data(n_tickers=10, n_days=1000)
        
        import time
        start_time = time.time()
        result = factor.calculate(large_data)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 1.0  # 1 second 