"""
Tests for the backtester metrics module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtester.metrics import (
    PerformanceMetrics,
    TradeMetrics,
    ReturnMetrics,
    PositionMetrics
)

@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    base_time = datetime(2023, 1, 1)
    return [
        {
            'timestamp': base_time,
            'ticker': 'AAPL',
            'side': 1,
            'quantity': 100,
            'price': 150.0,
            'value': 15000.0,
            'cost': 15.0
        },
        {
            'timestamp': base_time + timedelta(days=1),
            'ticker': 'AAPL',
            'side': -1,
            'quantity': 100,
            'price': 155.0,
            'value': 15500.0,
            'cost': 15.5
        },
        {
            'timestamp': base_time + timedelta(days=2),
            'ticker': 'GOOGL',
            'side': 1,
            'quantity': 50,
            'price': 2800.0,
            'value': 140000.0,
            'cost': 140.0
        },
        {
            'timestamp': base_time + timedelta(days=3),
            'ticker': 'GOOGL',
            'side': -1,
            'quantity': 50,
            'price': 2750.0,
            'value': 137500.0,
            'cost': 137.5
        }
    ]

@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    return {
        'AAPL': {
            'quantity': 100,
            'average_price': 150.0,
            'current_price': 155.0,
            'unrealized_pnl': 500.0,
            'realized_pnl': 0.0
        },
        'GOOGL': {
            'quantity': -50,
            'average_price': 2800.0,
            'current_price': 2750.0,
            'unrealized_pnl': 2500.0,
            'realized_pnl': 0.0
        }
    }

@pytest.fixture
def sample_portfolio_values():
    """Create sample portfolio values for testing."""
    # Create a series of portfolio values with some volatility
    initial_value = 100000.0
    num_days = 252  # One year of trading days
    
    # Generate daily returns with positive drift
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, num_days)  # Mean: 5bps, Std: 1%
    
    # Calculate portfolio values
    portfolio_values = initial_value * np.cumprod(1 + returns)
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='B')
    
    return portfolio_values, dates

class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics class."""
    
    def test_initialization(self):
        """Test metrics calculator initialization."""
        calculator = PerformanceMetrics()
        assert calculator.risk_free_rate == 0.02
        
        calculator = PerformanceMetrics(risk_free_rate=0.03)
        assert calculator.risk_free_rate == 0.03
    
    def test_trade_metrics_calculation(self, sample_trades):
        """Test trade metrics calculation."""
        calculator = PerformanceMetrics()
        metrics = calculator._calculate_trade_metrics(sample_trades)
        
        assert isinstance(metrics, TradeMetrics)
        assert metrics.num_trades == 4
        assert 0.0 <= metrics.win_rate <= 1.0
        assert metrics.profit_factor > 0
        assert metrics.avg_trade_duration > 0
    
    def test_return_metrics_calculation(self, sample_portfolio_values):
        """Test return metrics calculation."""
        calculator = PerformanceMetrics()
        portfolio_values, _ = sample_portfolio_values
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = calculator._calculate_return_metrics(returns, portfolio_values)
        
        assert isinstance(metrics, ReturnMetrics)
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.annual_return, float)
        assert isinstance(metrics.annual_volatility, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.max_drawdown <= 0  # Drawdown should be negative or zero
    
    def test_position_metrics_calculation(self, sample_positions, sample_trades):
        """Test position metrics calculation."""
        calculator = PerformanceMetrics()
        metrics = calculator._calculate_position_metrics(sample_positions, sample_trades)
        
        assert isinstance(metrics, PositionMetrics)
        assert metrics.avg_position_size > 0
        assert metrics.position_turnover >= 0
        assert metrics.avg_holding_period > 0
        assert metrics.max_position_size > 0
        assert 0 <= metrics.position_concentration <= 1
    
    def test_full_metrics_calculation(self, sample_portfolio_values, sample_trades, sample_positions):
        """Test full metrics calculation."""
        calculator = PerformanceMetrics()
        portfolio_values, dates = sample_portfolio_values
        
        results = {
            'portfolio_values': portfolio_values,
            'timestamps': dates,
            'trades': sample_trades,
            'positions': sample_positions
        }
        
        metrics = calculator.calculate(results)
        
        assert 'trade_metrics' in metrics
        assert 'return_metrics' in metrics
        assert 'position_metrics' in metrics
        
        assert isinstance(metrics['trade_metrics'], TradeMetrics)
        assert isinstance(metrics['return_metrics'], ReturnMetrics)
        assert isinstance(metrics['position_metrics'], PositionMetrics)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        calculator = PerformanceMetrics()
        
        # Empty trades
        trade_metrics = calculator._calculate_trade_metrics([])
        assert trade_metrics.num_trades == 0
        assert trade_metrics.win_rate == 0.0
        
        # Empty returns
        return_metrics = calculator._calculate_return_metrics(np.array([]), np.array([100000.0]))
        assert return_metrics.total_return == 0.0
        assert return_metrics.annual_return == 0.0
        
        # Empty positions
        position_metrics = calculator._calculate_position_metrics({}, [])
        assert position_metrics.avg_position_size == 0.0
        assert position_metrics.position_turnover == 0.0
    
    def test_edge_cases(self):
        """Test edge cases and potential error conditions."""
        calculator = PerformanceMetrics()
        
        # Single trade
        single_trade = [{
            'timestamp': datetime.now(),
            'ticker': 'AAPL',
            'side': 1,
            'quantity': 100,
            'price': 150.0,
            'value': 15000.0,
            'cost': 15.0
        }]
        metrics = calculator._calculate_trade_metrics(single_trade)
        assert metrics.num_trades == 1
        
        # Single position
        single_position = {
            'AAPL': {
                'quantity': 100,
                'average_price': 150.0,
                'current_price': 155.0,
                'unrealized_pnl': 500.0,
                'realized_pnl': 0.0
            }
        }
        metrics = calculator._calculate_position_metrics(single_position, single_trade)
        assert metrics.avg_position_size > 0
        
        # Constant portfolio value (no returns)
        constant_value = np.array([100000.0] * 10)
        returns = np.zeros(9)
        metrics = calculator._calculate_return_metrics(returns, constant_value)
        assert metrics.total_return == 0.0
        assert metrics.annual_volatility == 0.0
    
    def test_metric_consistency(self, sample_portfolio_values, sample_trades):
        """Test consistency of metric calculations."""
        calculator = PerformanceMetrics()
        portfolio_values, dates = sample_portfolio_values
        
        # Calculate returns in two ways
        returns1 = np.diff(portfolio_values) / portfolio_values[:-1]
        returns2 = np.log(portfolio_values[1:]) - np.log(portfolio_values[:-1])
        
        # Metrics should be similar for both return calculations
        metrics1 = calculator._calculate_return_metrics(returns1, portfolio_values)
        metrics2 = calculator._calculate_return_metrics(returns2, portfolio_values)
        
        # Compare key metrics (should be close but not exactly equal)
        assert np.abs(metrics1.annual_return - metrics2.annual_return) < 0.01
        assert np.abs(metrics1.annual_volatility - metrics2.annual_volatility) < 0.01
        assert np.abs(metrics1.sharpe_ratio - metrics2.sharpe_ratio) < 0.1
    
    def test_risk_metrics(self, sample_portfolio_values):
        """Test risk-related metrics calculation."""
        calculator = PerformanceMetrics()
        portfolio_values, _ = sample_portfolio_values
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = calculator._calculate_return_metrics(returns, portfolio_values)
        
        # Verify risk metrics
        assert metrics.max_drawdown <= 0  # Drawdown should be negative or zero
        assert metrics.annual_volatility >= 0  # Volatility should be positive
        assert metrics.drawdown_duration >= 0  # Duration should be positive
        
        # Verify Sortino ratio uses only negative returns
        negative_returns = returns[returns < 0]
        assert len(negative_returns) > 0  # Ensure we have some negative returns
        
        # Sortino ratio should be different from Sharpe ratio
        assert metrics.sortino_ratio != metrics.sharpe_ratio 