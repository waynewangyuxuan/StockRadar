"""
Tests for the strategy visualization module.

This module contains tests for the visualization functionality of the strategy engine,
including plotting equity curves, drawdowns, and trade analysis.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from strategy_engine.base import StrategyResult
from strategy_engine.visualization import StrategyVisualizer

@pytest.fixture
def sample_result():
    """Create a sample strategy result for testing."""
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    
    # Create equity curve
    equity_curve = pd.DataFrame({
        'value': np.random.randn(len(dates)).cumsum() + 100000
    }, index=dates)
    
    # Create drawdown curve
    drawdown_curve = pd.DataFrame({
        'value': np.random.randn(len(dates)).cumsum() / 10
    }, index=dates)
    drawdown_curve['value'] = drawdown_curve['value'].clip(upper=0)
    
    # Create trade summary
    trade_dates = dates[::30]  # Every 30 days
    trade_summary = pd.DataFrame({
        'entry_date': trade_dates[:-1],
        'exit_date': trade_dates[1:],
        'entry_price': np.random.randn(len(trade_dates)-1) * 10 + 100,
        'exit_price': np.random.randn(len(trade_dates)-1) * 10 + 100,
        'position': np.random.choice([-1, 1], len(trade_dates)-1),
        'pnl': np.random.randn(len(trade_dates)-1) * 1000,
        'duration': pd.Series([timedelta(days=30)] * (len(trade_dates)-1), index=trade_dates[:-1])
    })
    
    # Create position summary
    position_summary = pd.DataFrame({
        'position': np.random.choice([-1, 0, 1], len(dates)),
        'position_value': np.random.randn(len(dates)) * 10000
    }, index=dates)
    
    # Create signals
    signals = pd.DataFrame({
        'signal': np.random.choice([-1, 0, 1], len(dates))
    }, index=dates)
    
    # Create positions
    positions = pd.DataFrame({
        'position': np.random.choice([-1, 0, 1], len(dates))
    }, index=dates)
    
    # Create portfolio values
    portfolio_values = pd.DataFrame({
        'value': np.random.randn(len(dates)).cumsum() + 100000
    }, index=dates)
    
    # Create metrics
    metrics = {
        'total_return': 0.15,
        'annualized_return': 0.12,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.1,
        'win_rate': 0.6,
        'profit_factor': 1.8
    }
    
    return StrategyResult(
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        trade_summary=trade_summary,
        position_summary=position_summary,
        metrics=metrics,
        signals=signals,
        positions=positions,
        portfolio_values=portfolio_values
    )

@pytest.fixture
def visualizer(sample_result):
    """Create a strategy visualizer instance for testing."""
    return StrategyVisualizer(sample_result)

def test_visualizer_initialization(sample_result):
    """Test visualizer initialization."""
    visualizer = StrategyVisualizer(sample_result)
    assert visualizer.result == sample_result
    assert visualizer.fig is None
    assert visualizer.axes is None

def test_plot_results(visualizer, tmp_path):
    """Test plotting strategy results."""
    # Test without saving
    visualizer.plot_results()
    assert visualizer.fig is not None
    assert visualizer.axes is not None
    assert len(visualizer.axes) == 3
    plt.close(visualizer.fig)
    
    # Test with saving
    save_path = tmp_path / "test_plot.png"
    visualizer.plot_results(save_path=str(save_path))
    assert save_path.exists()
    plt.close(visualizer.fig)

def test_plot_equity_curve(visualizer):
    """Test plotting equity curve."""
    fig, ax = plt.subplots()
    visualizer._plot_equity_curve(ax)
    
    # Check that the plot has the expected elements
    assert len(ax.lines) > 0
    assert ax.get_title() == 'Equity Curve'
    assert ax.get_xlabel() == ''
    assert ax.get_ylabel() == ''
    
    plt.close(fig)

def test_plot_drawdown(visualizer):
    """Test plotting drawdown."""
    fig, ax = plt.subplots()
    visualizer._plot_drawdown(ax)
    
    # Check that the plot has the expected elements
    assert len(ax.patches) > 0
    assert ax.get_title() == 'Drawdown'
    assert ax.get_xlabel() == ''
    assert ax.get_ylabel() == ''
    
    plt.close(fig)

def test_plot_positions(visualizer):
    """Test plotting positions."""
    fig, ax = plt.subplots()
    visualizer._plot_positions(ax)
    
    # Check that the plot has the expected elements
    assert len(ax.lines) > 0
    assert ax.get_title() == 'Positions'
    assert ax.get_xlabel() == ''
    assert ax.get_ylabel() == ''
    
    plt.close(fig)

def test_plot_trade_analysis(visualizer, tmp_path):
    """Test plotting trade analysis."""
    # Test without saving
    visualizer.plot_trade_analysis()
    plt.close('all')
    
    # Test with saving
    save_path = tmp_path / "test_trade_analysis.png"
    visualizer.plot_trade_analysis(save_path=str(save_path))
    assert save_path.exists()
    plt.close('all')

def test_plot_pnl_distribution(visualizer):
    """Test plotting PnL distribution."""
    fig, ax = plt.subplots()
    visualizer._plot_pnl_distribution(ax)
    
    # Check that the plot has the expected elements
    assert len(ax.patches) > 0
    assert ax.get_title() == 'Trade PnL Distribution'
    
    plt.close(fig)

def test_plot_cumulative_pnl(visualizer):
    """Test plotting cumulative PnL."""
    fig, ax = plt.subplots()
    visualizer._plot_cumulative_pnl(ax)
    
    # Check that the plot has the expected elements
    assert len(ax.lines) > 0
    assert ax.get_title() == 'Cumulative PnL'
    
    plt.close(fig)

def test_plot_duration_vs_pnl(visualizer):
    """Test plotting duration vs PnL."""
    fig, ax = plt.subplots()
    visualizer._plot_duration_vs_pnl(ax)
    
    # Check that the plot has the expected elements
    assert len(ax.collections) > 0
    assert ax.get_title() == 'Trade Duration vs PnL'
    
    plt.close(fig)

def test_plot_monthly_win_rate(visualizer):
    """Test plotting monthly win rate."""
    fig, ax = plt.subplots()
    visualizer._plot_monthly_win_rate(ax)
    
    # Check that the plot has the expected elements
    assert len(ax.lines) > 0
    assert ax.get_title() == 'Monthly Win Rate'
    
    plt.close(fig) 