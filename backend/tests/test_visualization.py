"""
Tests for the visualization module.
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import shutil
from pathlib import Path

from backtester.models import EvaluationResult
from backtester.visualization import BacktestVisualizer

@pytest.fixture
def sample_result():
    """Create sample evaluation result for testing."""
    # Create sample dates
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create sample equity curve (growing with some volatility)
    initial_value = 100000
    returns = np.random.normal(0.0005, 0.01, len(dates))  # Daily returns
    equity_curve = pd.Series(initial_value * (1 + returns).cumprod(), index=dates)
    
    # Create sample drawdown curve
    rolling_max = equity_curve.expanding().max()
    drawdown_curve = (equity_curve - rolling_max) / rolling_max
    
    # Create sample trades
    trades = [
        {
            'ticker': 'AAPL',
            'entry_date': dates[10],
            'exit_date': dates[15],
            'position': 1,
            'entry_price': 150.0,
            'exit_price': 155.0,
            'pnl': 5.0,
            'return_pct': 3.33
        },
        {
            'ticker': 'GOOGL',
            'entry_date': dates[20],
            'exit_date': dates[25],
            'position': -1,
            'entry_price': 2800.0,
            'exit_price': 2750.0,
            'pnl': 50.0,
            'return_pct': 1.79
        }
    ]
    
    # Create sample metrics
    metrics = {
        'total_return': 15.5,
        'sharpe_ratio': 1.8,
        'max_drawdown': -8.5,
        'win_rate': 60.0,
        'total_trades': 10,
        'avg_return': 1.55,
        'avg_win': 2.5,
        'avg_loss': -1.2
    }
    
    # Create sample signals and positions
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    signals = pd.DataFrame(np.random.choice([-1, 0, 1], size=(len(dates), len(tickers))),
                         index=dates, columns=tickers)
    positions = pd.DataFrame(np.random.choice([-1, 0, 1], size=(len(dates), len(tickers))),
                           index=dates, columns=tickers)
    
    # Create sample portfolio values
    portfolio_values = pd.DataFrame(
        np.random.normal(100000, 1000, size=(len(dates), len(tickers))),
        index=dates, columns=tickers
    )
    
    return EvaluationResult(
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        trades=trades,
        metrics=metrics,
        signals=signals,
        positions=positions,
        portfolio_values=portfolio_values
    )

@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory for test results."""
    return str(tmp_path / "test_results")

class TestBacktestVisualizer:
    """Test suite for BacktestVisualizer class."""
    
    def test_initialization(self, output_dir):
        """Test visualizer initialization."""
        visualizer = BacktestVisualizer(output_dir)
        assert os.path.exists(output_dir)
        assert visualizer.output_dir == output_dir
    
    def test_plot_equity_curve(self, sample_result, output_dir):
        """Test equity curve plotting."""
        visualizer = BacktestVisualizer(output_dir)
        visualizer.plot_equity_curve(sample_result)
        
        # Check if plot was saved
        assert os.path.exists(os.path.join(output_dir, 'equity_curve.png'))
    
    def test_plot_trade_distribution(self, sample_result, output_dir):
        """Test trade distribution plotting."""
        visualizer = BacktestVisualizer(output_dir)
        visualizer.plot_trade_distribution(sample_result)
        
        # Check if plot was saved
        assert os.path.exists(os.path.join(output_dir, 'trade_distribution.png'))
    
    def test_plot_monthly_returns(self, sample_result, output_dir):
        """Test monthly returns plotting."""
        visualizer = BacktestVisualizer(output_dir)
        visualizer.plot_monthly_returns(sample_result)
        
        # Check if plot was saved
        assert os.path.exists(os.path.join(output_dir, 'monthly_returns.png'))
    
    def test_export_to_excel(self, sample_result, output_dir):
        """Test Excel export functionality."""
        visualizer = BacktestVisualizer(output_dir)
        visualizer.export_to_excel(sample_result)
        
        excel_file = os.path.join(output_dir, 'backtest_results.xlsx')
        assert os.path.exists(excel_file)
        
        # Verify Excel file contents
        with pd.ExcelFile(excel_file) as xls:
            assert 'Equity Curve' in xls.sheet_names
            assert 'Trades' in xls.sheet_names
            assert 'Metrics' in xls.sheet_names
            assert 'Positions' in xls.sheet_names
            assert 'Signals' in xls.sheet_names
    
    def test_plot_without_saving(self, sample_result, output_dir):
        """Test plotting without saving to file."""
        visualizer = BacktestVisualizer(output_dir)
        
        # Plot without saving
        visualizer.plot_equity_curve(sample_result, save=False)
        visualizer.plot_trade_distribution(sample_result, save=False)
        visualizer.plot_monthly_returns(sample_result, save=False)
        
        # Check that no files were created
        assert len(os.listdir(output_dir)) == 0
    
    def test_empty_trades(self, output_dir):
        """Test visualization with empty trades list."""
        result = EvaluationResult(
            equity_curve=pd.Series(),
            drawdown_curve=pd.Series(),
            trades=[],
            metrics={},
            signals=None,
            positions=None,
            portfolio_values=None
        )
        
        visualizer = BacktestVisualizer(output_dir)
        
        # Should not raise errors
        visualizer.plot_trade_distribution(result)
        visualizer.export_to_excel(result)
    
    def test_cleanup(self, output_dir):
        """Test cleanup of output directory."""
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        assert not os.path.exists(output_dir)
        
    def test_plot_price_action(self, sample_result, output_dir):
        """Test price action plotting."""
        visualizer = BacktestVisualizer(output_dir)
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        price_data = pd.Series(np.random.normal(100, 10, len(dates)), index=dates)
        
        visualizer.plot_price_action(sample_result, price_data)
        
        # Check if plot was saved
        assert os.path.exists(os.path.join(output_dir, 'price_action.png'))
    
    def test_plot_volume_profile(self, sample_result, output_dir):
        """Test volume profile plotting."""
        visualizer = BacktestVisualizer(output_dir)
        
        # Create sample price and volume data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        price_data = pd.Series(np.random.normal(100, 10, len(dates)), index=dates)
        volume_data = pd.Series(np.random.randint(1000, 10000, len(dates)), index=dates)
        
        visualizer.plot_volume_profile(sample_result, price_data, volume_data)
        
        # Check if plot was saved
        assert os.path.exists(os.path.join(output_dir, 'volume_profile.png')) 