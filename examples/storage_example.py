#!/usr/bin/env python
"""
Example demonstrating the usage of the consolidated StorageManager.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_storage import StorageManager

def generate_sample_data(n_rows=100, n_tickers=3):
    """Generate sample market data for demonstration."""
    tickers = [f"TICKER{i}" for i in range(1, n_tickers + 1)]
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_rows)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    
    data_list = []
    for ticker in tickers:
        # Generate random price data
        base_price = np.random.uniform(50, 200)
        prices = np.random.normal(loc=0, scale=1, size=n_rows).cumsum() + base_price
        prices = np.maximum(prices, 1)  # Ensure prices are positive
        
        # Create DataFrame for each ticker
        df = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'open': prices,
            'high': prices * np.random.uniform(1.01, 1.03, n_rows),
            'low': prices * np.random.uniform(0.97, 0.99, n_rows),
            'close': prices * np.random.uniform(0.98, 1.02, n_rows),
            'volume': np.random.randint(100000, 1000000, n_rows),
            'vwap': prices * np.random.uniform(0.99, 1.01, n_rows),
        })
        data_list.append(df)
    
    # Combine data for all tickers
    result = pd.concat(data_list, ignore_index=True)
    result['date'] = pd.to_datetime(result['date'])
    return result

def generate_sample_backtest_results(n_rows=30):
    """Generate sample backtest results for demonstration."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_rows)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    
    # Generate equity curve
    equity_curve = pd.DataFrame({
        'date': dates,
        'portfolio_value': 100000 * (1 + np.random.normal(0.001, 0.01, n_rows).cumsum()),
        'benchmark_value': 100000 * (1 + np.random.normal(0.0005, 0.008, n_rows).cumsum())
    })
    
    # Generate trade list
    trades = []
    for i in range(20):
        trade_date = start_date + timedelta(days=np.random.randint(0, n_rows))
        ticker = f"TICKER{np.random.randint(1, 4)}"
        is_buy = np.random.random() > 0.5
        price = np.random.uniform(50, 200)
        quantity = np.random.randint(10, 100)
        
        trades.append({
            'date': trade_date,
            'ticker': ticker,
            'action': 'BUY' if is_buy else 'SELL',
            'price': price,
            'quantity': quantity,
            'value': price * quantity,
            'commission': price * quantity * 0.001
        })
    
    # Generate performance metrics
    metrics = {
        'total_return': 0.15,
        'annualized_return': 0.12,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.08,
        'win_rate': 0.65,
        'profit_factor': 2.1,
        'num_trades': len(trades)
    }
    
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'metrics': metrics
    }

def main():
    # Initialize storage manager
    print("Initializing storage manager...")
    storage_mgr = StorageManager()
    
    # Generate and save market data
    print("\nGenerating sample market data...")
    market_data = generate_sample_data()
    print(f"Sample market data shape: {market_data.shape}")
    print(market_data.head())
    
    # Save market data
    print("\nSaving market data...")
    market_data_id = storage_mgr.store_market_data(
        data=market_data,
        metadata={'source': 'example', 'generated_at': str(datetime.now())}
    )
    print(f"Market data saved with ID: {market_data_id}")
    
    # Generate and save backtest results
    print("\nGenerating sample backtest results...")
    backtest_results = generate_sample_backtest_results()
    print(f"Sample backtest results contain:")
    print(f"  - Equity curve: {backtest_results['equity_curve'].shape[0]} days")
    print(f"  - Trades: {len(backtest_results['trades'])}")
    print(f"  - Metrics: {len(backtest_results['metrics'])} performance metrics")
    
    # Save backtest results
    print("\nSaving backtest results...")
    backtest_id = storage_mgr.store_backtest_results(
        results=backtest_results,
        strategy_name='sample_strategy',
        metadata={'author': 'example_user', 'parameters': {'param1': 0.1, 'param2': 0.2}}
    )
    print(f"Backtest results saved with ID: {backtest_id}")
    
    # List available datasets
    print("\nListing available datasets:")
    datasets = storage_mgr.list_datasets()
    for ds in datasets:
        print(f"  - {ds}")
    
    # Get versions for market data
    print("\nListing market data versions:")
    versions = storage_mgr.get_versions("market_data")
    for version in versions:
        print(f"  - {version['version_id']} (created: {version['created_at']})")
    
    # Load the latest market data
    print("\nLoading market data:")
    loaded_market_data = storage_mgr.get_market_data()
    print(f"Loaded market data shape: {loaded_market_data.shape}")
    print(loaded_market_data.head())
    
    # Load the backtest results
    print("\nLoading backtest results:")
    loaded_backtest = storage_mgr.get_backtest_results("sample_strategy")
    if loaded_backtest:
        print(f"Loaded backtest metrics:")
        for key, value in loaded_backtest['metrics'].items():
            print(f"  - {key}: {value}")
    
    print("\nStorage example completed successfully!")

if __name__ == "__main__":
    main() 