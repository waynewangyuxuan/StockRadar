"""
Visualization module for backtesting results.

This module provides functionality to visualize backtesting results,
including equity curves, drawdown curves, trade distributions,
and performance metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from matplotlib.dates import date2num
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from .models import EvaluationResult

class BacktestVisualizer:
    """
    Visualizes backtesting results.
    
    This class provides methods to create various visualizations
    of backtesting results, including:
    - Equity curves
    - Drawdown curves
    - Trade distributions
    - Monthly returns heatmaps
    - Position concentration
    - Performance dashboards
    """
    
    def __init__(self, output_dir: str = "backtest_results"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        sns.set_palette("husl")
        
        # Set non-interactive backend for testing
        if os.environ.get('PYTEST_CURRENT_TEST'):
            plt.switch_backend('Agg')
        
        # Create a figure with multiple pages
        self.fig = plt.figure(figsize=(15, 10))
        self.current_page = 0
        self.total_pages = 0
        self.pages = []
    
    def _add_page(self, title: str) -> GridSpec:
        """Add a new page to the figure."""
        self.total_pages += 1
        gs = GridSpec(1, 1, figure=self.fig)
        self.pages.append((gs, title))
        return gs
        
    def _show_page(self, page_num: int) -> None:
        """Show a specific page of the figure."""
        if 0 <= page_num < self.total_pages:
            self.current_page = page_num
            plt.clf()
            gs, title = self.pages[page_num]
            self.fig.add_subplot(gs[0, 0])
            plt.title(title)
            plt.tight_layout()
            plt.draw()
            
    def _show_or_close(self, save: bool) -> None:
        """Helper method to either show or close the plot based on context."""
        if save:
            plt.close()
        elif os.environ.get('PYTEST_CURRENT_TEST'):
            plt.close()
        else:
            plt.show()
    
    def plot_all(self, result: EvaluationResult, price_data: pd.Series, volume_data: pd.Series, save: bool = True) -> None:
        """
        Create all plots in a single figure with subplots.
        
        Args:
            result: Evaluation result containing all data
            price_data: Price series data
            volume_data: Volume series data
            save: Whether to save the plots
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot equity curve
        ax_equity = fig.add_subplot(gs[0, 0])
        result.equity_curve.plot(ax=ax_equity, label='Portfolio Value')
        ax_equity.set_title('Equity Curve')
        ax_equity.set_xlabel('Date')
        ax_equity.set_ylabel('Portfolio Value')
        ax_equity.grid(True)
        ax_equity.legend()
        
        # Plot trade distribution
        ax_trades = fig.add_subplot(gs[0, 1])
        if result.trades:
            returns = [trade['return_pct'] for trade in result.trades]
            sns.histplot(returns, kde=True, ax=ax_trades)
            ax_trades.set_title('Trade Return Distribution')
            ax_trades.set_xlabel('Return %')
            ax_trades.set_ylabel('Count')
            ax_trades.grid(True)
        else:
            ax_trades.text(0.5, 0.5, "No trades to plot", ha='center', va='center')
        
        # Plot monthly returns
        ax_monthly = fig.add_subplot(gs[1, :])
        monthly_returns = result.equity_curve.resample('M').last().pct_change() * 100
        monthly_returns_matrix = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        sns.heatmap(monthly_returns_matrix, annot=True, fmt='.1f', center=0, cmap='RdYlGn', ax=ax_monthly)
        ax_monthly.set_title('Monthly Returns (%)')
        ax_monthly.set_xlabel('Month')
        ax_monthly.set_ylabel('Year')
        
        # Plot price action with volume
        ax_price = fig.add_subplot(gs[2, 0])
        ax_volume = fig.add_subplot(gs[2, 1])
        
        # Plot price
        ax_price.plot(price_data.index, price_data.values, label='Price', color='gray', alpha=0.7)
        
        # Convert trades to DataFrame if it's a list
        if isinstance(result.trades, list):
            trades_df = pd.DataFrame(result.trades)
        else:
            trades_df = result.trades
        
        if not trades_df.empty:
            # Plot buy/sell signals
            buy_points = trades_df[trades_df['position'] > 0]
            sell_points = trades_df[trades_df['position'] < 0]
            
            if not buy_points.empty:
                ax_price.scatter(buy_points['entry_date'].values, 
                               [price_data[d] for d in buy_points['entry_date']], 
                               color='green', marker='^', s=100, label='Buy', alpha=0.7)
            
            if not sell_points.empty:
                ax_price.scatter(sell_points['entry_date'].values, 
                               [price_data[d] for d in sell_points['entry_date']], 
                               color='red', marker='v', s=100, label='Sell', alpha=0.7)
        
        ax_price.set_title('Price Action with Trade Signals')
        ax_price.set_xlabel('Date')
        ax_price.set_ylabel('Price')
        ax_price.grid(True)
        ax_price.legend()
        
        # Plot volume
        volume_bars = ax_volume.bar(volume_data.index, volume_data.values, 
                                  color='blue', alpha=0.5, label='Volume')
        
        # Color volume bars based on price movement
        for i in range(1, len(price_data)):
            color = 'green' if price_data.iloc[i] > price_data.iloc[i-1] else 'red'
            volume_bars[i-1].set_facecolor(color)
        volume_bars[0].set_facecolor('gray')  # First bar color
        
        ax_volume.set_title('Volume Profile')
        ax_volume.set_xlabel('Date')
        ax_volume.set_ylabel('Volume')
        ax_volume.grid(True)
        ax_volume.legend()
        
        # Format dates and adjust layout
        fig.autofmt_xdate()
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'backtest_summary.png'))
        
        self._show_or_close(save)
        
    def plot_equity_curve(self, result: EvaluationResult, save: bool = True) -> None:
        """
        Plot equity curve and drawdowns.
        
        Args:
            result: EvaluationResult containing equity curve and drawdowns
            save: Whether to save the plot to file
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
        
        # Plot equity curve
        result.equity_curve.plot(ax=ax1, label='Portfolio Value')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True)
        ax1.legend()
        
        # Plot drawdowns
        result.drawdown_curve.plot(ax=ax2, color='red', label='Drawdown')
        ax2.set_title('Drawdowns')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'equity_curve.png'))
        self._show_or_close(save)
    
    def plot_drawdown_curve(self, result: EvaluationResult, save: bool = True) -> None:
        """
        Plot drawdown curve.
        
        Args:
            result: EvaluationResult containing drawdown curve
            save: Whether to save the plot to file
        """
        plt.figure(figsize=(12, 6))
        result.drawdown_curve.plot(color='red', label='Drawdown')
        plt.title('Drawdown Curve')
        plt.xlabel('Date')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        plt.legend()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'drawdown_curve.png'))
        self._show_or_close(save)
    
    def plot_trade_distribution(self, result: EvaluationResult, save: bool = True) -> None:
        """
        Plot trade return distribution.
        
        Args:
            result: EvaluationResult containing trades
            save: Whether to save the plot to file
        """
        if not result.trades:
            print("No trades to plot")
            return
        
        returns = [trade['return_pct'] for trade in result.trades]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(returns, kde=True)
        plt.title('Trade Return Distribution')
        plt.xlabel('Return %')
        plt.ylabel('Count')
        plt.grid(True)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'trade_distribution.png'))
        self._show_or_close(save)
    
    def plot_monthly_returns(self, result: EvaluationResult, save: bool = True) -> None:
        """
        Plot monthly returns heatmap.
        
        Args:
            result: EvaluationResult containing equity curve
            save: Whether to save the plot to file
        """
        # Reset index to get date as a column
        equity_curve = result.equity_curve.reset_index()
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        
        # Calculate monthly returns
        monthly_returns = equity_curve.groupby([equity_curve['date'].dt.year, equity_curve['date'].dt.month])['value'].last().pct_change() * 100
        monthly_returns_matrix = monthly_returns.unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns_matrix, annot=True, fmt='.1f', center=0, cmap='RdYlGn')
        plt.title('Monthly Returns (%)')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'monthly_returns.png'))
        self._show_or_close(save)
    
    def plot_position_concentration(self, result: EvaluationResult, save: bool = True) -> None:
        """
        Plot position concentration.
        
        Args:
            result: EvaluationResult containing positions
            save: Whether to save the plot to file
        """
        if result.positions.empty:
            print("No positions to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Calculate position concentration
        position_values = result.positions['position'].value_counts()
        position_values = position_values / position_values.sum() * 100
        
        plt.pie(
            position_values,
            labels=position_values.index,
            autopct='%1.1f%%',
            startangle=90
        )
        
        plt.title('Position Concentration')
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'position_concentration.png'))
        self._show_or_close(save)
    
    def plot_performance_dashboard(self, result: EvaluationResult, save: bool = True) -> None:
        """
        Plot a comprehensive performance dashboard.
        
        Args:
            result: EvaluationResult containing all results
            save: Whether to save the plot
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # Reset index to get date as a column
        equity_curve = result.equity_curve.reset_index()
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(
            equity_curve['date'].unique(),
            equity_curve.groupby('date')['value'].sum(),
            label='Portfolio Value'
        )
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True)
        ax1.legend()
        
        # Reset index for drawdown curve
        drawdown_curve = result.drawdown_curve.reset_index()
        drawdown_curve['date'] = pd.to_datetime(drawdown_curve['date'])
        
        # Drawdown curve
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(
            drawdown_curve['date'].unique(),
            drawdown_curve.groupby('date')['value'].sum(),
            label='Drawdown',
            color='red'
        )
        ax2.set_title('Drawdown Curve')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        ax2.legend()
        
        # Trade distribution
        ax3 = fig.add_subplot(gs[2, 0])
        if result.trades:
            returns = [trade['return_pct'] for trade in result.trades]
            sns.histplot(returns, bins=20, kde=True, ax=ax3)
        ax3.set_title('Trade Return Distribution')
        ax3.set_xlabel('Return %')
        ax3.set_ylabel('Frequency')
        
        # Position concentration
        ax4 = fig.add_subplot(gs[2, 1])
        if not result.positions.empty:
            position_values = result.positions['position'].value_counts()
            position_values = position_values / position_values.sum() * 100
            ax4.pie(
                position_values,
                labels=position_values.index,
                autopct='%1.1f%%',
                startangle=90
            )
        ax4.set_title('Position Concentration')
        
        # Adjust layout
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'performance_dashboard.png'))
        self._show_or_close(save)
    
    def plot_price_action(self, result: EvaluationResult, price_data: pd.Series, save: bool = True) -> None:
        """
        Plot price action with buy/sell signals.
        
        Args:
            result: Evaluation result containing trade information
            price_data: Price series data (e.g., close prices)
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Plot price
        ax.plot(price_data.index, price_data.values, label='Price', color='gray', alpha=0.7)
        
        # Convert trades to DataFrame if it's a list
        if isinstance(result.trades, list):
            trades_df = pd.DataFrame(result.trades)
        else:
            trades_df = result.trades
        
        if not trades_df.empty:
            # Plot buy/sell signals
            buy_points = trades_df[trades_df['position'] > 0]
            sell_points = trades_df[trades_df['position'] < 0]
            
            if not buy_points.empty:
                ax.scatter(buy_points['entry_date'].values, 
                          [price_data[d] for d in buy_points['entry_date']], 
                          color='green', marker='^', s=100, label='Buy', alpha=0.7)
            
            if not sell_points.empty:
                ax.scatter(sell_points['entry_date'].values, 
                          [price_data[d] for d in sell_points['entry_date']], 
                          color='red', marker='v', s=100, label='Sell', alpha=0.7)
            
            # Add annotations for significant trades
            mean_pnl = trades_df['pnl'].mean()
            for _, trade in trades_df.iterrows():
                if abs(trade['pnl']) > mean_pnl * 2:  # Highlight significant trades
                    ax.annotate(f"P&L: ${trade['pnl']:.2f}", 
                              (trade['entry_date'], price_data[trade['entry_date']]),
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_title('Price Action with Trade Signals')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.legend()
        
        # Format x-axis to show dates nicely
        fig.autofmt_xdate()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'price_action.png'))
        self._show_or_close(save)

    def plot_volume_profile(self, result: EvaluationResult, price_data: pd.Series, volume_data: pd.Series, save: bool = True) -> None:
        """
        Plot volume profile with trade markers.
        
        Args:
            result: Evaluation result containing trade information
            price_data: Price series data
            volume_data: Volume series data
            save: Whether to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
        
        # Plot price and volume
        ax1.plot(price_data.index, price_data.values, color='gray', alpha=0.7, label='Price')
        ax2.bar(volume_data.index, volume_data.values, color='blue', alpha=0.5, label='Volume')
        
        # Convert trades to DataFrame if it's a list
        if isinstance(result.trades, list):
            trades_df = pd.DataFrame(result.trades)
        else:
            trades_df = result.trades
            
        if not trades_df.empty:
            # Add buy/sell markers
            buy_points = trades_df[trades_df['position'] > 0]
            sell_points = trades_df[trades_df['position'] < 0]
            
            # Plot markers on price chart
            if not buy_points.empty:
                ax1.scatter(buy_points['entry_date'].values,
                          [price_data[d] for d in buy_points['entry_date']],
                          color='green', marker='^', s=100, label='Buy', alpha=0.7)
                ax2.scatter(buy_points['entry_date'].values,
                          [volume_data[d] for d in buy_points['entry_date']],
                          color='green', marker='^', s=100, label='Buy', alpha=0.7)
            
            if not sell_points.empty:
                ax1.scatter(sell_points['entry_date'].values,
                          [price_data[d] for d in sell_points['entry_date']],
                          color='red', marker='v', s=100, label='Sell', alpha=0.7)
                ax2.scatter(sell_points['entry_date'].values,
                          [volume_data[d] for d in sell_points['entry_date']],
                          color='red', marker='v', s=100, label='Sell', alpha=0.7)
        
        # Add volume bars coloring based on price movement
        for i in range(1, len(price_data)):
            color = 'green' if price_data.iloc[i] > price_data.iloc[i-1] else 'red'
            ax2.patches[i-1].set_facecolor(color)
        ax2.patches[0].set_facecolor('gray')  # First bar color
        
        ax1.set_title('Price and Volume Profile with Trade Markers')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        ax2.legend()
        
        # Format x-axis to show dates nicely
        fig.autofmt_xdate()
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'volume_profile.png'))
        self._show_or_close(save)
    
    def export_to_excel(self, result: EvaluationResult, filename: str = "backtest_results.xlsx") -> str:
        """
        Export results to Excel file.
        
        Args:
            result: EvaluationResult containing all results
            filename: Name of the Excel file
            
        Returns:
            Path to the saved Excel file
        """
        output_file = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(output_file) as writer:
            # Write equity curve
            result.equity_curve.to_excel(writer, sheet_name='Equity Curve')
            
            # Write drawdown curve
            result.drawdown_curve.to_excel(writer, sheet_name='Drawdown Curve')
            
            # Write trades
            pd.DataFrame(result.trades).to_excel(writer, sheet_name='Trades')
            
            # Write metrics
            pd.Series(result.metrics).to_excel(writer, sheet_name='Metrics')
            
            # Write positions
            result.positions.to_excel(writer, sheet_name='Positions')
            
            # Write signals
            result.signals.to_excel(writer, sheet_name='Signals')
        
        return output_file 