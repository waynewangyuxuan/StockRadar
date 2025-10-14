#!/usr/bin/env python
"""
Utility for accessing and analyzing stored trading data
"""
import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage_integration import StorageManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load the configuration file"""
    config_path = Path(__file__).parent.parent / "config" / "api_config.yaml"
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_storage():
    """Initialize the storage manager from config"""
    config = load_config()
    return StorageManager(config.get('storage', {}))

def list_available_datasets(storage_manager, data_type=None):
    """List all available datasets in storage"""
    datasets = storage_manager.list_datasets()
    
    if data_type:
        datasets = [ds for ds in datasets if data_type in ds]
    
    if not datasets:
        print(f"No datasets found{' for type: ' + data_type if data_type else ''}.")
        return []
    
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
    
    return datasets

def get_backtest_results(storage_manager, strategy_name=None, date_range=None):
    """
    Retrieve backtest results for a specific strategy
    
    Args:
        storage_manager: Initialized storage manager
        strategy_name: Name of the strategy to filter by
        date_range: Tuple of (start_date, end_date) as strings in YYYY-MM-DD format
    
    Returns:
        DataFrame with backtest results
    """
    datasets = list_available_datasets(storage_manager, "backtest_results")
    
    if not datasets:
        return None
    
    # Filter by strategy name if provided
    if strategy_name:
        strategy_datasets = [ds for ds in datasets if strategy_name in ds]
        if strategy_datasets:
            datasets = strategy_datasets
        else:
            print(f"No datasets found for strategy: {strategy_name}")
            return None
    
    # Get the most recent dataset if multiple exist
    dataset_name = datasets[-1]
    print(f"\nRetrieving dataset: {dataset_name}")
    
    data = storage_manager.get_data(dataset_name)
    if data is None or data.empty:
        print(f"No data found for dataset: {dataset_name}")
        return None
    
    # Filter by date range if provided
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        if 'date' in data.columns:
            data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        elif 'timestamp' in data.columns:
            data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
    
    return data

def plot_equity_curve(backtest_data, save_path=None):
    """
    Plot the equity curve from backtest results
    
    Args:
        backtest_data: DataFrame with backtest results
        save_path: Path to save the plot image
    """
    if backtest_data is None or backtest_data.empty:
        print("No data available to plot equity curve")
        return
    
    # Check if portfolio value column exists
    if 'portfolio_value' not in backtest_data.columns:
        print("Portfolio value column not found in backtest data")
        return
    
    # Ensure we have a date/timestamp column
    date_col = 'date' if 'date' in backtest_data.columns else 'timestamp'
    if date_col not in backtest_data.columns:
        print(f"Date column not found in backtest data")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_data[date_col], backtest_data['portfolio_value'])
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Equity curve saved to {save_path}")
    else:
        plt.show()

def analyze_trades(backtest_data):
    """
    Analyze trades from backtest results
    
    Args:
        backtest_data: DataFrame with backtest results
    """
    if backtest_data is None or backtest_data.empty:
        print("No data available to analyze trades")
        return
    
    required_cols = ['ticker', 'action', 'price', 'quantity', 'timestamp']
    missing_cols = [col for col in required_cols if col not in backtest_data.columns]
    
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return
    
    trades = backtest_data[backtest_data['action'].isin(['BUY', 'SELL'])]
    
    if trades.empty:
        print("No trades found in the backtest data")
        return
    
    # Calculate trade statistics
    trade_count = len(trades)
    buy_trades = trades[trades['action'] == 'BUY']
    sell_trades = trades[trades['action'] == 'SELL']
    
    print("\nTrade Analysis:")
    print(f"Total trades: {trade_count}")
    print(f"Buy trades: {len(buy_trades)}")
    print(f"Sell trades: {len(sell_trades)}")
    
    # Calculate profit/loss if we have the necessary data
    if 'profit_loss' in trades.columns:
        total_pl = trades['profit_loss'].sum()
        winning_trades = trades[trades['profit_loss'] > 0]
        losing_trades = trades[trades['profit_loss'] < 0]
        
        print(f"Total profit/loss: {total_pl:.2f}")
        print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/trade_count*100:.2f}%)")
        print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/trade_count*100:.2f}%)")
        
        if len(winning_trades) > 0:
            print(f"Average winning trade: {winning_trades['profit_loss'].mean():.2f}")
        
        if len(losing_trades) > 0:
            print(f"Average losing trade: {losing_trades['profit_loss'].mean():.2f}")
    
    # Group trades by ticker
    ticker_stats = trades.groupby('ticker').size().sort_values(ascending=False)
    print("\nTrades by ticker:")
    for ticker, count in ticker_stats.items():
        print(f"{ticker}: {count} trades ({count/trade_count*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Utility for accessing and analyzing stored trading data')
    parser.add_argument('action', choices=['list', 'results', 'plot', 'analyze'],
                        help='Action to perform: list datasets, show results, plot equity curve, or analyze trades')
    parser.add_argument('--type', help='Filter datasets by type')
    parser.add_argument('--strategy', help='Filter by strategy name')
    parser.add_argument('--start-date', help='Start date for filtering (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for filtering (YYYY-MM-DD)')
    parser.add_argument('--save', help='Save output to file (for plots)')
    
    args = parser.parse_args()
    
    try:
        storage_manager = initialize_storage()
        
        if args.action == 'list':
            list_available_datasets(storage_manager, args.type)
        
        elif args.action == 'results':
            date_range = None
            if args.start_date and args.end_date:
                date_range = (args.start_date, args.end_date)
            
            results = get_backtest_results(storage_manager, args.strategy, date_range)
            if results is not None and not results.empty:
                print("\nBacktest Results Summary:")
                print(results.describe())
                print("\nFirst few rows:")
                print(results.head())
        
        elif args.action == 'plot':
            date_range = None
            if args.start_date and args.end_date:
                date_range = (args.start_date, args.end_date)
            
            results = get_backtest_results(storage_manager, args.strategy, date_range)
            save_path = None
            if args.save:
                # Create save directory if it doesn't exist
                save_dir = Path(__file__).parent.parent / "config" / "output" / "analysis"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / args.save
            
            plot_equity_curve(results, save_path)
        
        elif args.action == 'analyze':
            date_range = None
            if args.start_date and args.end_date:
                date_range = (args.start_date, args.end_date)
            
            results = get_backtest_results(storage_manager, args.strategy, date_range)
            analyze_trades(results)
    
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 