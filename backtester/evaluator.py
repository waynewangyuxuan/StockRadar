"""
Strategy evaluator for backtesting results.

This module provides functionality to evaluate trading strategies
based on backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from .models import EvaluationResult
from .metrics import PerformanceMetrics
from .visualization import BacktestVisualizer
from core.schema import StrategyInterface, DataSchema, SignalSchema

class StrategyEvaluator:
    """
    Evaluates trading strategies based on backtesting results.
    """
    
    def __init__(self, strategy: StrategyInterface, initial_capital: float = 100000.0):
        """
        Initialize the strategy evaluator.
        
        Args:
            strategy: The trading strategy to evaluate
            initial_capital: Initial capital for the strategy
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.metrics = PerformanceMetrics()
        self.visualizer = BacktestVisualizer()
    
    def evaluate(self, data: pd.DataFrame) -> EvaluationResult:
        """
        Evaluate the strategy on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            EvaluationResult containing performance metrics and analysis
        """
        # Validate data
        if data.empty:
            raise ValueError("Empty data provided")
            
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check for multi-index
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have a MultiIndex with levels ['date', 'ticker']")
            
        # Get strategy signals
        signals = self.strategy.generate_signals(data)
        
        # Calculate positions and portfolio values
        positions = self._calculate_positions(signals, data)
        portfolio_values = self._calculate_portfolio_values(positions, data)
        
        # Calculate equity curve and drawdowns
        equity_curve = self._calculate_equity_curve(portfolio_values)
        drawdown_curve = self._calculate_drawdowns(equity_curve)
        
        # Extract trades
        trades = self._extract_trades(positions, data)
        
        # Calculate metrics
        metrics = self.metrics.calculate(equity_curve, drawdown_curve, trades)
        
        return EvaluationResult(
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            trades=trades,
            metrics=metrics,
            signals=signals,
            positions=positions,
            portfolio_values=portfolio_values
        )
    
    def _calculate_positions(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate positions from signals."""
        positions = pd.DataFrame(index=data.index)
        positions['position'] = signals['signal'].astype(int)
        return positions
    
    def _calculate_portfolio_values(self, positions: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate portfolio values from positions."""
        portfolio = pd.DataFrame(index=data.index)
        
        # Initialize with initial capital
        portfolio['value'] = self.initial_capital
        
        # Calculate returns based on close prices
        returns = data['close'].pct_change()
        
        # Calculate position returns
        position_returns = positions['position'].shift(1) * returns
        
        # Calculate portfolio values
        portfolio['value'] = (1 + position_returns).cumprod() * self.initial_capital
        
        # Fill NaN values with initial capital
        portfolio['value'] = portfolio['value'].fillna(self.initial_capital)
        
        return portfolio
    
    def _calculate_equity_curve(self, portfolio_values: pd.DataFrame) -> pd.DataFrame:
        """Calculate equity curve from portfolio values."""
        equity_curve = pd.DataFrame(index=portfolio_values.index)
        equity_curve['value'] = portfolio_values['value']
        return equity_curve
    
    def _calculate_drawdowns(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """Calculate drawdown series from equity curve."""
        drawdowns = pd.DataFrame(index=equity_curve.index)
        rolling_max = equity_curve['value'].expanding().max()
        drawdowns['value'] = (equity_curve['value'] - rolling_max) / rolling_max
        return drawdowns
    
    def _extract_trades(self, positions: pd.DataFrame, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract trades from position changes."""
        trades = []
        position_changes = positions['position'].diff()
        
        # Group by ticker
        for ticker in data.index.get_level_values('ticker').unique():
            ticker_mask = data.index.get_level_values('ticker') == ticker
            ticker_positions = positions.loc[ticker_mask]
            ticker_changes = position_changes.loc[ticker_mask]
            ticker_data = data.loc[ticker_mask]
            
            # Find entry and exit points
            change_points = ticker_changes[ticker_changes != 0].index
            
            for i in range(len(change_points) - 1):
                entry_date = change_points[i]
                exit_date = change_points[i + 1]
                
                position = ticker_positions.loc[entry_date, 'position']
                if position == 0:  # Skip if no position taken
                    continue
                
                entry_price = ticker_data.loc[entry_date, 'close']
                exit_price = ticker_data.loc[exit_date, 'close']
                
                pnl = (exit_price - entry_price) * position
                return_pct = (pnl / entry_price) * 100
                
                trade = {
                    'ticker': ticker,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return_pct': return_pct
                }
                trades.append(trade)
        
        return trades
    
    def _calculate_metrics(self, equity_curve: pd.Series, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        annualized_return = ((1 + total_return/100) ** (252/len(returns)) - 1) * 100
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        max_drawdown = self._calculate_drawdowns(equity_curve).min() * 100
        
        # Trade metrics
        total_trades = len(trades)
        if total_trades > 0:
            win_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = (win_trades / total_trades) * 100
            
            # Calculate average returns
            returns = [t['return_pct'] for t in trades]
            avg_return = np.mean(returns) if returns else 0
            
            # Calculate average win/loss
            win_returns = [r for r in returns if r > 0]
            loss_returns = [r for r in returns if r <= 0]
            avg_win = np.mean(win_returns) if win_returns else 0
            avg_loss = np.mean(loss_returns) if loss_returns else 0
            
            # Calculate profit factor
            gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_return = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        metrics = {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': total_trades,
            'avg_return': float(avg_return),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor)
        }
        
        return metrics
    
    def plot_equity_curve(self, result: EvaluationResult, save: bool = False) -> None:
        """Plot equity curve."""
        self.visualizer.plot_equity_curve(result.equity_curve, save)
    
    def plot_drawdown_curve(self, result: EvaluationResult, save: bool = False) -> None:
        """Plot drawdown curve."""
        self.visualizer.plot_drawdown_curve(result.drawdown_curve, save)
    
    def plot_trade_distribution(self, result: EvaluationResult, save: bool = False) -> None:
        """Plot trade distribution."""
        self.visualizer.plot_trade_distribution(result.trades, save)
    
    def plot_monthly_returns(self, result: EvaluationResult, save: bool = False) -> None:
        """Plot monthly returns."""
        self.visualizer.plot_monthly_returns(result.equity_curve, save)
    
    def plot_position_concentration(self, result: EvaluationResult, save: bool = False) -> None:
        """Plot position concentration."""
        self.visualizer.plot_position_concentration(result.positions, save)
    
    def plot_performance_dashboard(self, result: EvaluationResult, save: bool = False) -> None:
        """Plot performance dashboard."""
        self.visualizer.plot_performance_dashboard(result, save)
    
    def generate_report(self, result: EvaluationResult) -> str:
        """Generate performance report."""
        report = []
        report.append("Performance Report")
        report.append("=================")
        report.append("")
        
        # Add metrics section
        report.append("Performance Metrics")
        report.append("-----------------")
        for key, value in result.metrics.items():
            report.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
        report.append("")
        
        # Add trade statistics
        report.append("Trade Statistics")
        report.append("---------------")
        report.append(f"Total Trades: {len(result.trades)}")
        if result.trades:
            win_trades = len([t for t in result.trades if t['pnl'] > 0])
            win_rate = (win_trades / len(result.trades)) * 100
            report.append(f"Win Rate: {win_rate:.2f}%")
            
            avg_pnl = np.mean([t['pnl'] for t in result.trades])
            report.append(f"Average P&L: {avg_pnl:.2f}")
            
            avg_duration = np.mean([
                (t['exit_date'][0] - t['entry_date'][0]).days 
                for t in result.trades
            ])
            report.append(f"Average Trade Duration: {avg_duration:.1f} days")
        report.append("")
        
        return "\n".join(report)
    
    def print_metrics(self, result: EvaluationResult) -> None:
        """Print performance metrics."""
        print(self.generate_report(result))
    
    def export_to_excel(self, result: EvaluationResult, filename: str) -> str:
        """Export results to Excel."""
        with pd.ExcelWriter(filename) as writer:
            # Write equity curve
            result.equity_curve.to_excel(writer, sheet_name='Equity Curve')
            
            # Write drawdown curve
            result.drawdown_curve.to_excel(writer, sheet_name='Drawdowns')
            
            # Write trades
            trades_df = pd.DataFrame(result.trades)
            trades_df.to_excel(writer, sheet_name='Trades')
            
            # Write positions
            result.positions.to_excel(writer, sheet_name='Positions')
            
            # Write metrics
            metrics_df = pd.DataFrame([result.metrics])
            metrics_df.to_excel(writer, sheet_name='Metrics')
        
        return filename
