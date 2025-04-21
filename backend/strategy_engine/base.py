"""
Base strategy engine implementation.

This module provides the core functionality for strategy evaluation,
analysis, and visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StrategyResult:
    """Results of strategy evaluation."""
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    trade_summary: pd.DataFrame
    position_summary: pd.DataFrame
    metrics: Dict[str, float]
    signals: pd.DataFrame
    positions: pd.DataFrame
    portfolio_values: pd.DataFrame

class StrategyEngine:
    """
    Core engine for strategy evaluation and analysis.
    
    This class provides methods to:
    - Evaluate trading strategies
    - Calculate performance metrics
    - Generate trade and position analysis
    - Visualize results
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the strategy engine.
        
        Args:
            initial_capital: Initial capital for portfolio calculations
        """
        self.initial_capital = initial_capital
        
    def evaluate_strategy(self, strategy, data: pd.DataFrame) -> StrategyResult:
        """
        Evaluate a trading strategy on historical data.
        
        Args:
            strategy: The trading strategy to evaluate
            data: DataFrame with OHLCV data and any required factor columns
            
        Returns:
            StrategyResult containing evaluation results
        """
        if data.empty:
            raise ValueError("Empty data provided")
            
        # Generate signals from strategy
        signals = strategy.generate_signals(data)
        
        # Calculate positions and portfolio values
        positions = self._calculate_positions(signals, data)
        portfolio_values = self._calculate_portfolio_values(positions, data)
        
        # Calculate equity curve and drawdowns
        equity_curve = self._calculate_equity_curve(portfolio_values)
        drawdown_curve = self._calculate_drawdowns(equity_curve)
        
        # Extract trades and calculate metrics
        trades = self._extract_trades(positions, data)
        metrics = self._calculate_metrics(equity_curve, drawdown_curve, trades)
        
        # Generate summaries
        trade_summary = self._generate_trade_summary(trades)
        position_summary = self._generate_position_summary(positions)
        
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
    
    def _calculate_positions(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate position sizes based on signals."""
        positions = pd.DataFrame(0, index=signals.index, columns=['position'])
        signal_values = signals['signal'].astype(int)
        positions.loc[signal_values > 0, 'position'] = 1
        positions.loc[signal_values < 0, 'position'] = -1
        return positions
    
    def _calculate_portfolio_values(self, positions: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate portfolio value over time based on positions."""
        portfolio_values = pd.DataFrame(index=positions.index)
        returns = data['close'].pct_change()
        position_returns = positions['position'].shift(1) * returns
        portfolio_values['value'] = self.initial_capital * (1 + position_returns).cumprod()
        return portfolio_values
    
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
        """Extract individual trades from position changes."""
        trades = []
        position_changes = positions['position'].diff()
        
        for idx in position_changes[position_changes != 0].index:
            trade = {
                'entry_date': idx,
                'entry_price': data.loc[idx, 'close'],
                'position': positions.loc[idx, 'position'],
                'size': abs(positions.loc[idx, 'position'])
            }
            trades.append(trade)
            
        return trades
    
    def _calculate_metrics(self, equity_curve: pd.DataFrame, drawdown_curve: pd.DataFrame, 
                         trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics."""
        returns = equity_curve['value'].pct_change()
        
        metrics = {
            'total_return': (equity_curve['value'].iloc[-1] / self.initial_capital) - 1,
            'annualized_return': self._calculate_annualized_return(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': drawdown_curve['value'].min(),
            'win_rate': self._calculate_win_rate(trades),
            'profit_factor': self._calculate_profit_factor(trades)
        }
        
        return metrics
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        total_days = len(returns)
        total_return = (1 + returns).prod() - 1
        return (1 + total_return) ** (252 / total_days) - 1
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return winning_trades / len(trades)
    
    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor from trades."""
        gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _generate_trade_summary(self, trades: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate summary statistics for trades."""
        if not trades:
            return pd.DataFrame()
            
        summary = pd.DataFrame(trades)
        summary['duration'] = summary['exit_date'] - summary['entry_date']
        return summary
    
    def _generate_position_summary(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for positions."""
        summary = pd.DataFrame(index=positions.index)
        summary['position'] = positions['position']
        summary['position_value'] = summary['position'] * positions.index.get_level_values('close')
        return summary
