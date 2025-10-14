"""
Tests for the backtester evaluator module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtester.evaluator import StrategyEvaluator, EvaluationResult
from core.strategy_base import StrategyBase, SignalType
from typing import List, Optional

class MockStrategy(StrategyBase):
    """Mock strategy for testing."""
    
    def __init__(self):
        """Initialize mock strategy."""
        super().__init__()
        self.name = "MockStrategy"
        self.description = "A mock strategy for testing"
    
    def get_required_factors(self) -> List[str]:
        """Get list of factor names required by this strategy."""
        return []  # No factors required for this mock strategy
    
    def generate_signals(self, market_data: pd.DataFrame, factor_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate mock signals."""
        # Ensure market data has proper index
        if not isinstance(market_data.index, pd.MultiIndex):
            market_data.index = pd.MultiIndex.from_product(
                [market_data.index, ['GOOGL']],
                names=['date', 'ticker']
            )
        
        # Get unique dates and tickers
        dates = market_data.index.get_level_values('date').unique()
        tickers = market_data.index.get_level_values('ticker').unique()
        
        # Generate random signals
        signals_data = []
        for date in dates:
            for ticker in tickers:
                signal = int(np.random.choice([1, -1]))  # Buy or sell signal as int
                signals_data.append({
                    'date': date,
                    'ticker': ticker,
                    'signal': signal,
                    'signal_strength': np.random.random(),
                    'strategy_name': self.name,
                    'timestamp': pd.Timestamp.now()
                })
        
        # Create signals DataFrame with proper index
        signals = pd.DataFrame(signals_data)
        signals.set_index(['date', 'ticker'], inplace=True)
        signals['signal'] = signals['signal'].astype(int)  # Ensure int type
        
        return signals

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    # Create date range
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    tickers = ['AAPL', 'GOOGL']
    
    # Create multi-index
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    # Generate OHLCV data
    np.random.seed(42)
    n = len(index)
    
    # Generate prices with trend and volatility
    base_price = 100
    trend = np.linspace(0, 20, n)  # Upward trend
    volatility = np.random.normal(0, 2, n)
    prices = base_price + trend + np.cumsum(volatility)
    
    # Ensure prices are positive and create OHLCV data
    prices = np.maximum(prices, 1)
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.5, n),
        'high': prices + np.random.normal(1, 0.5, n),
        'low': prices - np.random.normal(1, 0.5, n),
        'close': prices + np.random.normal(0, 0.5, n),
        'volume': np.random.randint(1000, 10000, n)
    }, index=index)
    
    # Ensure high > open/close > low
    data['high'] = data[['open', 'high', 'close']].max(axis=1) + 0.1
    data['low'] = data[['open', 'low', 'close']].min(axis=1) - 0.1
    
    return data

@pytest.fixture
def evaluator():
    """Create strategy evaluator instance."""
    strategy = MockStrategy()
    return StrategyEvaluator(strategy=strategy, initial_capital=100000.0)

class TestStrategyEvaluator:
    """Test suite for StrategyEvaluator class."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        strategy = MockStrategy()
        evaluator = StrategyEvaluator(strategy=strategy, initial_capital=100000.0)
        
        assert evaluator.strategy == strategy
        assert evaluator.initial_capital == 100000.0
        assert evaluator.visualizer is not None
    
    def test_position_calculation(self, evaluator, sample_data):
        """Test position calculation from signals."""
        signals = evaluator.strategy.generate_signals(sample_data)
        positions = evaluator._calculate_positions(signals, sample_data)
        
        assert isinstance(positions, pd.DataFrame)
        assert positions.index.equals(sample_data.index)
        assert all(pos in [-1, 0, 1] for pos in positions.values.flatten())
    
    def test_portfolio_value_calculation(self, evaluator, sample_data):
        """Test portfolio value calculation."""
        signals = evaluator.strategy.generate_signals(sample_data)
        positions = evaluator._calculate_positions(signals, sample_data)
        portfolio_values = evaluator._calculate_portfolio_values(positions, sample_data)
        
        assert isinstance(portfolio_values, pd.DataFrame)
        assert portfolio_values.index.equals(sample_data.index)
        assert portfolio_values.iloc[0].sum() == pytest.approx(evaluator.initial_capital)
    
    def test_equity_curve_calculation(self, evaluator, sample_data):
        """Test equity curve calculation."""
        signals = evaluator.strategy.generate_signals(sample_data)
        positions = evaluator._calculate_positions(signals, sample_data)
        portfolio_values = evaluator._calculate_portfolio_values(positions, sample_data)
        equity_curve = evaluator._calculate_equity_curve(portfolio_values)
        
        assert isinstance(equity_curve, pd.DataFrame)
        assert equity_curve.index.equals(sample_data.index)
        assert equity_curve['value'].iloc[0] == pytest.approx(evaluator.initial_capital)
        assert all(equity_curve['value'] > 0)  # Equity should always be positive
    
    def test_drawdown_calculation(self, evaluator, sample_data):
        """Test drawdown calculation."""
        signals = evaluator.strategy.generate_signals(sample_data)
        positions = evaluator._calculate_positions(signals, sample_data)
        portfolio_values = evaluator._calculate_portfolio_values(positions, sample_data)
        equity_curve = evaluator._calculate_equity_curve(portfolio_values)
        drawdowns = evaluator._calculate_drawdowns(equity_curve)
        
        assert isinstance(drawdowns, pd.DataFrame)
        assert drawdowns.index.equals(sample_data.index)
        assert all(drawdowns['value'] <= 0)  # Drawdowns should always be negative or zero
        assert drawdowns['value'].min() >= -1  # Maximum drawdown cannot exceed -100%
    
    def test_trade_extraction(self, evaluator, sample_data):
        """Test trade extraction from positions."""
        signals = evaluator.strategy.generate_signals(sample_data)
        positions = evaluator._calculate_positions(signals, sample_data)
        trades = evaluator._extract_trades(positions, sample_data)
        
        assert isinstance(trades, list)
        for trade in trades:
            assert isinstance(trade, dict)
            assert all(key in trade for key in [
                'ticker', 'entry_date', 'exit_date', 'position',
                'entry_price', 'exit_price', 'pnl', 'return_pct'
            ])
            assert trade['entry_date'] < trade['exit_date']
            assert trade['position'] in [-1, 1]
    
    def test_metrics_calculation(self, evaluator, sample_data):
        """Test performance metrics calculation."""
        signals = evaluator.strategy.generate_signals(sample_data)
        positions = evaluator._calculate_positions(signals, sample_data)
        portfolio_values = evaluator._calculate_portfolio_values(positions, sample_data)
        equity_curve = evaluator._calculate_equity_curve(portfolio_values)
        trades = evaluator._extract_trades(positions, sample_data)
        metrics = evaluator._calculate_metrics(equity_curve, trades)
        
        assert isinstance(metrics, dict)
        assert all(key in metrics for key in [
            'total_return', 'sharpe_ratio', 'max_drawdown',
            'win_rate', 'total_trades', 'avg_return',
            'avg_win', 'avg_loss'
        ])
        assert metrics['max_drawdown'] <= 0
        assert 0 <= metrics['win_rate'] <= 100
        assert metrics['total_trades'] >= 0
    
    def test_full_evaluation(self, evaluator, sample_data):
        """Test full strategy evaluation."""
        result = evaluator.evaluate(sample_data)
        
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.equity_curve, pd.DataFrame)
        assert isinstance(result.drawdown_curve, pd.DataFrame)
        assert isinstance(result.trades, list)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.signals, pd.DataFrame)
        assert isinstance(result.positions, pd.DataFrame)
        assert isinstance(result.portfolio_values, pd.DataFrame)
    
    def test_edge_cases(self, evaluator):
        """Test edge cases and error handling."""
        # Empty data
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        with pytest.raises(ValueError):
            evaluator.evaluate(empty_data)
        
        # Single day of data - should work but with no returns
        single_day = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100],
            'volume': [1000]
        }, index=[datetime.now()])
        
        # Set MultiIndex for single day data
        single_day.index = pd.MultiIndex.from_product(
            [single_day.index, ['GOOGL']],
            names=['date', 'ticker']
        )
        
        result = evaluator.evaluate(single_day)
        assert result.metrics['total_return'] == 0
        assert result.metrics['sharpe_ratio'] == 0
        
        # Missing columns
        invalid_data = pd.DataFrame({'close': [100, 101, 102]})
        with pytest.raises(ValueError):
            evaluator.evaluate(invalid_data)
    
    def test_visualization_methods(self, evaluator, sample_data):
        """Test visualization method calls."""
        result = evaluator.evaluate(sample_data)
        
        # Test all plotting methods
        evaluator.plot_equity_curve(result, save=False)
        evaluator.plot_drawdown_curve(result, save=False)
        evaluator.plot_trade_distribution(result, save=False)
        evaluator.plot_monthly_returns(result, save=False)
        evaluator.plot_position_concentration(result, save=False)
        evaluator.plot_performance_dashboard(result, save=False)
    
    def test_report_generation(self, evaluator, sample_data):
        """Test report generation methods."""
        result = evaluator.evaluate(sample_data)
        
        # Test report generation
        report = evaluator.generate_report(result)
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Test metrics printing
        evaluator.print_metrics(result)  # Should not raise any errors
        
        # Test Excel export
        filename = "test_results.xlsx"
        export_path = evaluator.export_to_excel(result, filename)
        assert isinstance(export_path, str)
        assert export_path.endswith(".xlsx") 