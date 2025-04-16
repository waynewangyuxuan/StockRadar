import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from core.schema import DataSchema, SignalSchema, SignalType

class TestDataSchema:
    @pytest.fixture
    def valid_data(self):
        """Create valid market data for testing."""
        return pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'date': [datetime.now(), datetime.now()],
            'open': [150.0, 300.0],
            'high': [155.0, 305.0],
            'low': [145.0, 295.0],
            'close': [152.0, 302.0],
            'volume': [1000000, 500000]
        })
    
    def test_get_required_columns(self):
        """Test getting required columns."""
        required_cols = DataSchema.get_required_columns()
        assert len(required_cols) == 7
        assert all(col in required_cols for col in [
            'ticker', 'date', 'open', 'high', 'low', 'close', 'volume'
        ])
    
    def test_validate_valid_data(self, valid_data):
        """Test validating valid data."""
        assert DataSchema.validate_data(valid_data) is True
    
    def test_validate_missing_columns(self, valid_data):
        """Test validating data with missing columns."""
        invalid_data = valid_data.drop(['volume'], axis=1)
        with pytest.raises(ValueError) as exc_info:
            DataSchema.validate_data(invalid_data)
        assert "Missing required columns" in str(exc_info.value)
    
    def test_validate_empty_data(self):
        """Test validating empty data."""
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError) as exc_info:
            DataSchema.validate_data(empty_data)
        assert "Missing required columns" in str(exc_info.value)

class TestSignalSchema:
    @pytest.fixture
    def valid_signals(self):
        """Create valid signals for testing."""
        return pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'date': [datetime.now(), datetime.now()],
            'signal': [SignalType.BUY.value, SignalType.SELL.value],
            'strength': [0.8, 0.9],
            'strategy': ['ma_cross', 'ma_cross']
        })
    
    def test_get_required_columns(self):
        """Test getting required columns."""
        required_cols = SignalSchema.get_required_columns()
        assert len(required_cols) == 3
        assert all(col in required_cols for col in [
            'ticker', 'date', 'signal'
        ])
    
    def test_validate_valid_signals(self, valid_signals):
        """Test validating valid signals."""
        assert SignalSchema.validate_signals(valid_signals) is True
    
    def test_validate_missing_columns(self, valid_signals):
        """Test validating signals with missing columns."""
        invalid_signals = valid_signals.drop(['signal'], axis=1)
        with pytest.raises(ValueError) as exc_info:
            SignalSchema.validate_signals(invalid_signals)
        assert "Missing required columns" in str(exc_info.value)
    
    def test_validate_invalid_signal_values(self, valid_signals):
        """Test validating signals with invalid values."""
        invalid_signals = valid_signals.copy()
        invalid_signals.loc[0, 'signal'] = 999  # Invalid signal value
        with pytest.raises(ValueError) as exc_info:
            SignalSchema.validate_signals(invalid_signals)
        assert "Invalid signal values" in str(exc_info.value)
    
    def test_validate_empty_signals(self):
        """Test validating empty signals."""
        empty_signals = pd.DataFrame()
        with pytest.raises(ValueError) as exc_info:
            SignalSchema.validate_signals(empty_signals)
        assert "Missing required columns" in str(exc_info.value)
    
    def test_signal_type_values(self):
        """Test signal type enumeration values."""
        assert SignalType.BUY.value == 1
        assert SignalType.SELL.value == -1
        assert SignalType.HOLD.value == 0 