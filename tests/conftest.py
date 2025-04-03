import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from monitoring.metrics.collector import DefaultMetricsCollector
from monitoring.alerts.alert_manager import AlertManager, ConsoleAlertNotifier
from monitoring.lineage.tracker import LineageTracker

@pytest.fixture
def sample_market_data():
    """生成用于测试的市场数据"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    data = []
    for symbol in symbols:
        # 生成模拟数据
        n = len(dates)
        close_prices = np.random.normal(100, 10, n).cumsum()  # 随机游走
        
        for i, date in enumerate(dates):
            data.append({
                'symbol': symbol,
                'date': date,
                'open': close_prices[i] * (1 + np.random.normal(0, 0.01)),
                'high': close_prices[i] * (1 + np.random.normal(0.01, 0.01)),
                'low': close_prices[i] * (1 - np.random.normal(0.01, 0.01)),
                'close': close_prices[i],
                'volume': np.random.randint(1000000, 10000000)
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def metrics_collector():
    """创建指标收集器"""
    return DefaultMetricsCollector()

@pytest.fixture
def alert_manager():
    """创建告警管理器"""
    manager = AlertManager()
    manager.add_notifier(ConsoleAlertNotifier())
    return manager

@pytest.fixture
def lineage_tracker():
    """创建血缘追踪器"""
    return LineageTracker()

@pytest.fixture
def date_range():
    """创建测试用的日期范围"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    return start_date, end_date 