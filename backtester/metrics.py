"""
Performance metrics for backtesting results.

This module provides functions to calculate various performance metrics
for evaluating trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

@dataclass
class TradeMetrics:
    """Metrics related to individual trades."""
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float

@dataclass
class ReturnMetrics:
    """Metrics related to returns."""
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: float

@dataclass
class PositionMetrics:
    """Metrics related to positions."""
    avg_position_size: float
    position_turnover: float
    avg_holding_period: float
    max_position_size: float
    position_concentration: float

class PerformanceMetrics:
    """
    Calculates performance metrics for backtesting results.
    
    This class provides methods to calculate various performance metrics
    for evaluating trading strategies, including:
    - Return metrics (Sharpe ratio, Sortino ratio, etc.)
    - Risk metrics (drawdown, volatility, etc.)
    - Trade metrics (win rate, profit factor, etc.)
    - Position metrics (turnover, concentration, etc.)
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for calculating risk-adjusted returns
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate(self, equity_curve: pd.DataFrame, drawdown_curve: pd.DataFrame, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            equity_curve: Portfolio equity curve
            drawdown_curve: Portfolio drawdown curve
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns
        returns = equity_curve['value'].pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve['value'].iloc[-1] / equity_curve['value'].iloc[0] - 1) * 100 if len(equity_curve) > 0 else 0
        
        # Handle edge cases for annualized return
        if len(returns) > 0:
            annualized_return = ((1 + total_return/100) ** (252/len(returns)) - 1) * 100
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        else:
            annualized_return = 0
            sharpe_ratio = 0
            
        max_drawdown = drawdown_curve['value'].min() * 100 if len(drawdown_curve) > 0 else 0
        
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
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> TradeMetrics:
        """
        Calculate metrics related to individual trades.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            TradeMetrics object containing trade-related metrics
        """
        if not trades:
            return TradeMetrics(
                num_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                avg_trade_duration=0.0
            )
        
        # Convert trades to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades)
        
        # Calculate trade P&L
        trades_df['pnl'] = trades_df['value'] * trades_df['side']
        
        # Separate winning and losing trades
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        # Calculate metrics
        num_trades = len(trades_df)
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0
        
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0
        
        largest_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0.0
        largest_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0.0
        
        # Calculate average trade duration
        if 'timestamp' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            avg_duration = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).total_seconds() / num_trades
        else:
            avg_duration = 0.0
        
        return TradeMetrics(
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_duration
        )
    
    def _calculate_return_metrics(
        self,
        returns: np.ndarray,
        portfolio_values: np.ndarray
    ) -> ReturnMetrics:
        """
        Calculate metrics related to returns.
        
        Args:
            returns: Array of returns
            portfolio_values: Array of portfolio values
            
        Returns:
            ReturnMetrics object containing return-related metrics
        """
        if len(returns) == 0:
            return ReturnMetrics(
                total_return=0.0,
                annual_return=0.0,
                annual_volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                avg_drawdown=0.0,
                drawdown_duration=0.0
            )
        
        # Calculate total and annual returns
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = total_return * (252 / len(returns))  # Assuming daily data
        
        # Calculate volatility
        annual_volatility = np.std(returns) * np.sqrt(252)
        
        # Calculate Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        
        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1.0
        sortino_ratio = np.sqrt(252) * np.mean(excess_returns) / downside_std if downside_std > 0 else 0.0
        
        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()
        avg_drawdown = np.mean(drawdown[drawdown < 0]) if len(drawdown[drawdown < 0]) > 0 else 0.0
        
        # Calculate drawdown duration
        drawdown_periods = np.sum(drawdown < 0)
        drawdown_duration = drawdown_periods / len(drawdown)
        
        return ReturnMetrics(
            total_return=total_return,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            drawdown_duration=drawdown_duration
        )
    
    def _calculate_position_metrics(
        self,
        positions: Dict[str, Any],
        trades: List[Dict[str, Any]]
    ) -> PositionMetrics:
        """
        Calculate metrics related to positions.
        
        Args:
            positions: Dictionary of positions
            trades: List of trade dictionaries
            
        Returns:
            PositionMetrics object containing position-related metrics
        """
        if not positions:
            return PositionMetrics(
                avg_position_size=0.0,
                position_turnover=0.0,
                avg_holding_period=0.0,
                max_position_size=0.0,
                position_concentration=0.0
            )
        
        # Calculate position sizes
        position_sizes = [abs(pos['quantity'] * pos['current_price']) for pos in positions.values()]
        total_position_size = sum(position_sizes)
        
        # Calculate average and max position size
        avg_position_size = total_position_size / len(positions)
        max_position_size = max(position_sizes)
        
        # Calculate position concentration (Herfindahl-Hirschman Index)
        position_weights = [size / total_position_size for size in position_sizes]
        position_concentration = sum(w * w for w in position_weights)
        
        # Calculate position turnover
        if not trades:
            position_turnover = 0.0
        else:
            trade_values = sum(abs(trade['value']) for trade in trades)
            avg_portfolio_value = total_position_size  # Simplified assumption
            position_turnover = trade_values / (2 * avg_portfolio_value)  # Divide by 2 to avoid double counting
        
        # Calculate average holding period
        if not trades:
            avg_holding_period = 0.0
        else:
            trade_dates = [pd.to_datetime(trade['timestamp']) for trade in trades]
            if len(trade_dates) > 1:
                holding_periods = [(trade_dates[i+1] - trade_dates[i]).days 
                                 for i in range(len(trade_dates)-1)]
                avg_holding_period = sum(holding_periods) / len(holding_periods)
            else:
                avg_holding_period = 0.0
        
        return PositionMetrics(
            avg_position_size=float(avg_position_size),
            position_turnover=float(position_turnover),
            avg_holding_period=float(avg_holding_period),
            max_position_size=float(max_position_size),
            position_concentration=float(position_concentration)
        )
