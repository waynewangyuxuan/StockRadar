import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from data_layer.monitoring.metrics import DefaultMetricsCollector
from data_layer.monitoring.alerts import AlertManager, ConsoleAlertNotifier
from data_layer.monitoring.lineage import LineageTracker

@pytest.fixture
def sample_market_data():
    """Generate market data for testing"""
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n)
    
    # Generate simulated data
    np.random.seed(42)
    close_prices = np.random.normal(100, 10, n).cumsum()  # Random walk
    data = pd.DataFrame({
        'date': dates,
        'symbol': 'AAPL',
        'open': close_prices * (1 + np.random.normal(0, 0.02, n)),
        'high': close_prices * (1 + np.random.normal(0.02, 0.02, n)),
        'low': close_prices * (1 + np.random.normal(-0.02, 0.02, n)),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, n)
    })
    
    return data

@pytest.fixture
def metrics_collector():
    """Create metrics collector"""
    return DefaultMetricsCollector()

@pytest.fixture
def alert_manager():
    """Create alert manager"""
    manager = AlertManager()
    manager.add_notifier(ConsoleAlertNotifier())
    return manager

@pytest.fixture
def lineage_tracker():
    """Create lineage tracker"""
    return LineageTracker()

@pytest.fixture
def date_range():
    """Create test date range"""
    return (
        datetime(2023, 1, 1),
        datetime(2023, 12, 31)
    ) 