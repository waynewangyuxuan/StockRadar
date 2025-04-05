"""Comprehensive demo of data layer functionalities"""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from data_layer.fetcher.yfinance_provider import YFinanceProvider
from data_layer.processor.market_data_processor import MarketDataProcessor
from data_layer.monitoring.metrics import DefaultMetricsCollector
from data_layer.monitoring.alerts import AlertManager, AlertSeverity, ConsoleAlertNotifier
from data_layer.monitoring.lineage import LineageTracker

def print_section(title: str) -> None:
    """Print section title"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_data_info(data: Any, title: str = "Data") -> None:
    """Print data information"""
    print(f"\n{title}:")
    if isinstance(data, pd.DataFrame):
        print(f"Shape: {data.shape}")
        print("\nColumns:", ", ".join(data.columns))
        print("\nSample data (first 5 rows):")
        print(data.head().to_string())
        print("\nData Types:")
        print(data.dtypes)
        print("\nSummary Statistics:")
        print(data.describe())
    elif isinstance(data, dict):
        print(f"Number of symbols: {len(data)}")
        for symbol, symbol_data in data.items():
            print(f"\n{symbol}:")
            if isinstance(symbol_data, pd.Series):
                print(symbol_data.to_string())
            else:
                print(symbol_data)
    else:
        print(data)

def main() -> None:
    """Main function"""
    # Initialize monitoring components
    metrics_collector = DefaultMetricsCollector()
    alert_manager = AlertManager()
    alert_manager.add_notifier(ConsoleAlertNotifier())
    lineage_tracker = LineageTracker()
    
    # Initialize data provider
    print_section("Initializing Data Provider")
    yfinance_provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    
    # Test symbol validation
    print_section("Testing Symbol Validation")
    valid_symbols = ['AAPL', 'MSFT', 'GOOGL']
    invalid_symbols = ['INVALID_SYMBOL']
    all_symbols = valid_symbols + invalid_symbols
    
    try:
        validated_symbols = yfinance_provider.validate_symbols(all_symbols)
        print("Valid symbols:", validated_symbols)
        print("Invalid symbols:", set(all_symbols) - set(validated_symbols))
    except Exception as e:
        print(f"Error validating symbols: {str(e)}")
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Test with all fields
        print_section("Fetching Historical Data (All Fields)")
        historical_data = yfinance_provider.get_historical_data(
            symbols=valid_symbols,
            start_date=start_date,
            end_date=end_date
        )
        print_data_info(historical_data['data'], "Historical Data (All Fields)")
        
        # Process complete data
        print_section("Processing Complete Data")
        processor = MarketDataProcessor(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            lineage_tracker=lineage_tracker
        )
        processed_data = processor.process(historical_data)
        print_data_info(processed_data['data'], "Processed Data (All Fields)")
        print("\nAdded Features:", processed_data['metadata']['added_features'])
        
        # Test with specific fields
        print_section("Fetching Historical Data (Filtered Fields)")
        historical_data_filtered = yfinance_provider.get_historical_data(
            symbols=valid_symbols,
            start_date=start_date,
            end_date=end_date,
            fields=['open', 'close', 'volume']
        )
        print_data_info(historical_data_filtered['data'], "Historical Data (Filtered Fields)")
        
        # Process filtered data
        print_section("Processing Filtered Data")
        processed_data_filtered = processor.process(historical_data_filtered)
        print_data_info(processed_data_filtered['data'], "Processed Data (Filtered Fields)")
        print("\nAdded Features:", processed_data_filtered['metadata']['added_features'])
        
        # Get latest data with field filtering
        print_section("Fetching Latest Data")
        latest_data = yfinance_provider.get_latest_data(
            symbols=valid_symbols,
            fields=['open', 'close', 'volume']
        )
        print_data_info(latest_data['data'], "Latest Data")
        
        # View monitoring metrics
        print_section("Monitoring Metrics")
        metrics = metrics_collector.get_metrics()
        for metric in metrics:
            print(f"{metric.name}: {metric.value}")
        
        # View lineage relationships
        print_section("Viewing Lineage Relationships")
        lineage_graph = lineage_tracker.export_graph()
        print("\nData Lineage Graph:")
        print(lineage_graph)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 