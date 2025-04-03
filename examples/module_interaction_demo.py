from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any

from data_layer.fetcher.yfinance_provider import YFinanceProvider
from data_layer.processor.market_data_processor import MarketDataProcessor
from monitoring.metrics.collector import DefaultMetricsCollector
from monitoring.alerts.alert_manager import AlertManager, ConsoleAlertNotifier
from monitoring.lineage.tracker import LineageTracker

def print_section(title: str) -> None:
    """Print section title"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}\n")

def print_data_info(data: pd.DataFrame) -> None:
    """Print data information"""
    print("Shape:", data.shape)
    print("\nColumns:", list(data.columns))
    print("\nSample data:")
    print(data.head())
    print("\nData types:")
    print(data.dtypes)
    print("\nSummary statistics:")
    print(data.describe())

def main():
    """Main function"""
    # Initialize monitoring components
    metrics_collector = DefaultMetricsCollector()
    alert_manager = AlertManager()
    alert_manager.add_notifier(ConsoleAlertNotifier())
    lineage_tracker = LineageTracker()

    print_section("1. Initialize Components")
    print("Monitoring components initialized successfully.")

    # Initialize data providers
    print_section("2. Initialize Data Providers")
    provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    print("Provider:", provider.__class__.__name__)
    print("Source Node ID:", provider.source_node.id if provider.source_node else "None")

    # Get historical data
    print_section("3. Get Historical Data")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    try:
        historical_data = provider.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        print_data_info(historical_data['data'])
    except Exception as e:
        print(f"Error fetching historical data: {str(e)}")

    # Get latest data
    print_section("4. Get Latest Data")
    try:
        latest_data = provider.get_latest_data(symbols=symbols)
        for symbol, data in latest_data['data'].items():
            print(f"\n{symbol} latest data:")
            print(data)
    except Exception as e:
        print(f"Error fetching latest data: {str(e)}")

    # Initialize data processor
    print_section("5. Initialize Data Processor")
    processor = MarketDataProcessor(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    print("Processor:", processor.__class__.__name__)

    # Process data
    print_section("6. Process Data")
    if 'historical_data' in locals() and not historical_data['data'].empty:
        try:
            processed_data = processor.process_data(historical_data['data'])
            print_data_info(processed_data['data'])
        except Exception as e:
            print(f"Error processing data: {str(e)}")

    # View lineage relationships
    print_section("7. Data Lineage")
    graph = lineage_tracker.export_graph()
    print("\nNodes:")
    for node in graph['nodes']:
        print(f"- {node.name} ({node.type})")
    print("\nEdges:")
    for edge in graph['edges']:
        print(f"- {edge[0]} -> {edge[1]}")

    # View metrics
    print_section("8. Performance Metrics")
    if metrics_collector.metrics:
        for metric_name, metric_data in metrics_collector.metrics.items():
            print(f"\n{metric_name}:")
            print(f"  Count: {metric_data['count']}")
            print(f"  Total: {metric_data['total']}")
            print(f"  Average: {metric_data['average']:.2f}")
    else:
        print("No metrics available")

if __name__ == "__main__":
    main() 