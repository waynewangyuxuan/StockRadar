"""
Visualization module for strategy results.

This module provides functionality to visualize strategy evaluation results,
including equity curves, drawdowns, trades, and positions.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from .base import StrategyResult

class StrategyVisualizer:
    """
    Visualizer for strategy evaluation results.
    
    This class provides methods to create various visualizations of strategy
    performance, including:
    - Equity curves
    - Drawdown charts
    - Trade analysis
    - Position analysis
    """
    
    def __init__(self, result: StrategyResult):
        """
        Initialize the visualizer with strategy results.
        
        Args:
            result: StrategyResult containing evaluation results
        """
        self.result = result
        self.fig = None
        self.axes = None
        
    def plot_results(self, figsize: tuple = (15, 10), 
                    save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive visualization of strategy results.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
        """
        self.fig, self.axes = plt.subplots(3, 1, figsize=figsize, 
                                         gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot equity curve
        self._plot_equity_curve(self.axes[0])
        
        # Plot drawdown
        self._plot_drawdown(self.axes[1])
        
        # Plot positions
        self._plot_positions(self.axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def _plot_equity_curve(self, ax: plt.Axes) -> None:
        """Plot equity curve with annotations."""
        equity = self.result.equity_curve['value']
        ax.plot(equity.index, equity.values, label='Equity Curve')
        
        # Add annotations for key points
        max_value = equity.max()
        min_value = equity.min()
        final_value = equity.iloc[-1]
        
        ax.annotate(f'Max: {max_value:,.0f}',
                   xy=(equity.idxmax(), max_value),
                   xytext=(10, 10), textcoords='offset points')
        
        ax.annotate(f'Min: {min_value:,.0f}',
                   xy=(equity.idxmin(), min_value),
                   xytext=(10, -10), textcoords='offset points')
        
        ax.annotate(f'Final: {final_value:,.0f}',
                   xy=(equity.index[-1], final_value),
                   xytext=(10, 10), textcoords='offset points')
        
        ax.set_title('Equity Curve')
        ax.grid(True)
        ax.legend()
        
    def _plot_drawdown(self, ax: plt.Axes) -> None:
        """Plot drawdown chart."""
        drawdown = self.result.drawdown_curve['value']
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                       color='red', alpha=0.3, label='Drawdown')
        
        # Add annotation for maximum drawdown
        max_dd = drawdown.min()
        ax.annotate(f'Max DD: {max_dd:.1%}',
                   xy=(drawdown.idxmin(), max_dd),
                   xytext=(10, -10), textcoords='offset points')
        
        ax.set_title('Drawdown')
        ax.grid(True)
        ax.legend()
        
    def _plot_positions(self, ax: plt.Axes) -> None:
        """Plot position changes."""
        positions = self.result.positions['position']
        ax.plot(positions.index, positions.values, 
                drawstyle='steps-post', label='Position')
        
        # Add annotations for position changes
        position_changes = positions.diff()
        for idx in position_changes[position_changes != 0].index:
            ax.annotate(f'{positions[idx]:.0f}',
                       xy=(idx, positions[idx]),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center')
        
        ax.set_title('Positions')
        ax.grid(True)
        ax.legend()
        
    def plot_trade_analysis(self, figsize: tuple = (15, 10),
                          save_path: Optional[str] = None) -> None:
        """
        Create a visualization of trade analysis.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
        """
        if self.result.trade_summary.empty:
            print("No trades to analyze")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot trade PnL distribution
        self._plot_pnl_distribution(axes[0, 0])
        
        # Plot cumulative PnL
        self._plot_cumulative_pnl(axes[0, 1])
        
        # Plot trade duration vs PnL
        self._plot_duration_vs_pnl(axes[1, 0])
        
        # Plot win rate by month
        self._plot_monthly_win_rate(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def _plot_pnl_distribution(self, ax: plt.Axes) -> None:
        """Plot distribution of trade PnL."""
        pnl = self.result.trade_summary['pnl']
        ax.hist(pnl, bins=50, alpha=0.7)
        ax.set_title('Trade PnL Distribution')
        ax.grid(True)
        
    def _plot_cumulative_pnl(self, ax: plt.Axes) -> None:
        """Plot cumulative PnL over time."""
        pnl = self.result.trade_summary['pnl'].cumsum()
        ax.plot(pnl.index, pnl.values)
        ax.set_title('Cumulative PnL')
        ax.grid(True)
        
    def _plot_duration_vs_pnl(self, ax: plt.Axes) -> None:
        """Plot trade duration vs PnL."""
        ax.scatter(self.result.trade_summary['duration'],
                  self.result.trade_summary['pnl'])
        ax.set_title('Trade Duration vs PnL')
        ax.grid(True)
        
    def _plot_monthly_win_rate(self, ax: plt.Axes) -> None:
        """Plot monthly win rate."""
        monthly_wins = self.result.trade_summary.groupby(
            self.result.trade_summary.index.to_period('M'))['pnl'].apply(
                lambda x: (x > 0).mean())
        ax.plot(monthly_wins.index.astype(str), monthly_wins.values)
        ax.set_title('Monthly Win Rate')
        ax.grid(True)
        plt.xticks(rotation=45) 