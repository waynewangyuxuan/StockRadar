"""
Strategy evaluator for backtesting results.

This module provides functionality to evaluate trading strategies
based on backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from .visualization import BacktestVisualizer
from .models import EvaluationResult
from core.strategy_base import StrategyBase

@dataclass
class EvaluationResult:
    """Results of strategy evaluation."""
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    trades: List[Dict]
    metrics: Dict[str, float]
    signals: pd.DataFrame
    positions: pd.DataFrame
    portfolio_values: pd.DataFrame

class StrategyEvaluator:
    """
    Evaluates trading strategies based on backtesting results.
    
    This class provides methods to evaluate trading strategies, including:
    - Performance analysis
    - Risk assessment
    - Trade analysis
    - Position analysis
    - Visualization
    """
    
    def __init__(self, strategy: StrategyBase, initial_capital: float = 100000.0):
        """
        Initialize the strategy evaluator.
        
        Args:
            strategy: The trading strategy
            initial_capital: Initial capital for the strategy
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.visualizer = BacktestVisualizer()
    
    def evaluate(self, data: pd.DataFrame) -> EvaluationResult:
        """
        Evaluate the strategy on historical data.
        
        Args:
            data: DataFrame with OHLCV data and any required factor columns
            
        Returns:
            EvaluationResult containing equity curve, drawdowns, trades and metrics
        """
        if data.empty:
            raise ValueError("Empty data provided")
            
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
            
        # Check for multi-index
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have a multi-index with 'date' and 'ticker' levels")
            
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        # Calculate positions and portfolio values
        positions = self._calculate_positions(signals, data)
        portfolio_values = self._calculate_portfolio_values(positions, data)
        
        # Calculate equity curve and drawdowns
        equity_curve = pd.DataFrame(portfolio_values['value'])
        drawdown_curve = pd.DataFrame(self._calculate_drawdowns(portfolio_values['value']))
        
        # Extract trades and calculate metrics
        trades = self._extract_trades(positions, data)
        metrics = self._calculate_metrics(portfolio_values['value'], trades)
        
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
        """Calculate position sizes based on signals."""
        # Initialize positions DataFrame with same index as signals
        positions = pd.DataFrame(0, index=signals.index, columns=['position'])
        
        # Convert signals to positions (-1, 0, 1)
        signal_values = signals['signal'].astype(int)
        positions.loc[signal_values > 0, 'position'] = 1
        positions.loc[signal_values < 0, 'position'] = -1
        
        return positions
    
    def _calculate_portfolio_values(self, positions: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate portfolio value over time based on positions."""
        # Calculate returns based on close prices
        returns = data['close'].pct_change()
        
        # Calculate position returns
        position_returns = positions['position'].shift(1) * returns
        
        # Calculate portfolio values
        portfolio_values = pd.DataFrame(index=positions.index)
        portfolio_values['value'] = (1 + position_returns).cumprod() * self.initial_capital
        
        # Fill NaN values with initial capital
        portfolio_values['value'] = portfolio_values['value'].fillna(self.initial_capital)
        
        return portfolio_values
    
    def _calculate_equity_curve(self, portfolio_values: pd.DataFrame) -> pd.Series:
        """Calculate equity curve from portfolio values."""
        return portfolio_values['value']
    
    def _calculate_drawdowns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        return drawdowns
    
    def _extract_trades(self, positions: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """Extract individual trades from position changes."""
        trades = []
        position_changes = positions['position'].diff()
        
        # Group by ticker to handle trades for each ticker separately
        for ticker in data.index.get_level_values('ticker').unique():
            ticker_mask = data.index.get_level_values('ticker') == ticker
            ticker_positions = positions.loc[ticker_mask]
            ticker_changes = position_changes.loc[ticker_mask]
            
            # Find entry and exit points
            entries = ticker_changes[ticker_changes != 0].index
            
            for i in range(len(entries) - 1):
                entry_date = entries[i]
                exit_date = entries[i + 1]
                
                position_size = ticker_positions.loc[entry_date, 'position']
                if position_size == 0:  # Skip if no position taken
                    continue
                    
                entry_price = data.loc[entry_date, 'close']
                exit_price = data.loc[exit_date, 'close']
                
                pnl = (exit_price - entry_price) * position_size
                return_pct = (pnl / entry_price) * 100
                
                trade = {
                    'ticker': ticker,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'position': position_size,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return_pct': return_pct
                }
                trades.append(trade)
                
        return trades
    
    def _calculate_metrics(self, equity_curve: pd.Series, trades: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        max_drawdown = self._calculate_drawdowns(equity_curve).min() * 100
        
        # Trade metrics
        win_trades = len([t for t in trades if t['pnl'] > 0])
        total_trades = len(trades)
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_return = np.mean([t['return_pct'] for t in trades]) if trades else 0
        avg_win = np.mean([t['return_pct'] for t in trades if t['pnl'] > 0]) if win_trades > 0 else 0
        avg_loss = np.mean([t['return_pct'] for t in trades if t['pnl'] < 0]) if (total_trades - win_trades) > 0 else 0
        
        metrics = {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': total_trades,
            'avg_return': float(avg_return),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss)
        }
        
        return metrics
    
    def plot_equity_curve(self, evaluation_result: EvaluationResult, save: bool = True) -> None:
        """
        Plot equity curve.
        
        Args:
            evaluation_result: EvaluationResult object
            save: Whether to save the plot
        """
        self.visualizer.plot_equity_curve(evaluation_result, save)
    
    def plot_drawdown_curve(self, evaluation_result: EvaluationResult, save: bool = True) -> None:
        """
        Plot drawdown curve.
        
        Args:
            evaluation_result: EvaluationResult object
            save: Whether to save the plot
        """
        self.visualizer.plot_drawdown_curve(evaluation_result, save)
    
    def plot_trade_distribution(self, evaluation_result: EvaluationResult, save: bool = True) -> None:
        """
        Plot trade distribution.
        
        Args:
            evaluation_result: EvaluationResult object
            save: Whether to save the plot
        """
        self.visualizer.plot_trade_distribution(evaluation_result, save)
    
    def plot_monthly_returns(self, evaluation_result: EvaluationResult, save: bool = True) -> None:
        """
        Plot monthly returns heatmap.
        
        Args:
            evaluation_result: EvaluationResult object
            save: Whether to save the plot
        """
        self.visualizer.plot_monthly_returns(evaluation_result, save)
    
    def plot_position_concentration(self, evaluation_result: EvaluationResult, save: bool = True) -> None:
        """
        Plot position concentration.
        
        Args:
            evaluation_result: EvaluationResult object
            save: Whether to save the plot
        """
        self.visualizer.plot_position_concentration(evaluation_result, save)
    
    def plot_performance_dashboard(self, evaluation_result: EvaluationResult, save: bool = True) -> None:
        """
        Plot a comprehensive performance dashboard.
        
        Args:
            evaluation_result: EvaluationResult object
            save: Whether to save the plot
        """
        self.visualizer.plot_performance_dashboard(evaluation_result, save)
    
    def export_to_excel(self, evaluation_result: EvaluationResult, filename: str = "backtest_results.xlsx") -> str:
        """
        Export backtest results to Excel.
        
        Args:
            evaluation_result: EvaluationResult object
            filename: Name of the Excel file
            
        Returns:
            Path to the saved Excel file
        """
        return self.visualizer.export_to_excel(evaluation_result, filename)
    
    def print_metrics(self, evaluation_result: EvaluationResult) -> None:
        """
        Print performance metrics.
        
        Args:
            evaluation_result: EvaluationResult object
        """
        metrics = evaluation_result.metrics
        
        print("=== Performance Metrics ===")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['total_return'] / 252:.2%}")
        print(f"Annual Volatility: {metrics['sharpe_ratio'] * np.sqrt(252):.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        print("\n=== Trade Metrics ===")
        print(f"Number of Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['total_return'] / -metrics['avg_loss']:.2f}")
        print(f"Average Win: {metrics['avg_win']:.2f}")
        print(f"Average Loss: {metrics['avg_loss']:.2f}")
    
    def generate_report(self, evaluation_result: EvaluationResult) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_result: EvaluationResult object
            
        Returns:
            String containing the evaluation report
        """
        metrics = evaluation_result.metrics
        
        report = []
        report.append("# Strategy Evaluation Report")
        report.append("")
        
        report.append("## Performance Metrics")
        report.append(f"- Total Return: {metrics['total_return']:.2%}")
        report.append(f"- Annual Return: {metrics['total_return'] / 252:.2%}")
        report.append(f"- Annual Volatility: {metrics['sharpe_ratio'] * np.sqrt(252):.2%}")
        report.append(f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append(f"- Max Drawdown: {metrics['max_drawdown']:.2%}")
        report.append("")
        
        report.append("## Trade Metrics")
        report.append(f"- Number of Trades: {metrics['total_trades']}")
        report.append(f"- Win Rate: {metrics['win_rate']:.2%}")
        report.append(f"- Profit Factor: {metrics['total_return'] / -metrics['avg_loss']:.2f}")
        report.append(f"- Average Win: {metrics['avg_win']:.2f}")
        report.append(f"- Average Loss: {metrics['avg_loss']:.2f}")
        report.append("")
        
        report.append("## Equity Curve")
        report.append("```")
        report.append(evaluation_result.equity_curve.to_string())
        report.append("```")
        report.append("")
        
        report.append("## Drawdown Curve")
        report.append("```")
        report.append(evaluation_result.drawdown_curve.to_string())
        report.append("```")
        report.append("")
        
        report.append("## Trade Summary")
        report.append("```")
        report.append(pd.DataFrame(evaluation_result.trades).to_string())
        report.append("```")
        report.append("")
        
        return "\n".join(report)
