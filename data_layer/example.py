from datetime import datetime, timedelta
from typing import Dict, Any

from monitoring.metrics.collector import DefaultMetricsCollector
from monitoring.alerts.alert_manager import AlertManager, ConsoleAlertNotifier
from monitoring.lineage.tracker import LineageTracker

def demo_data_layer():
    # Initialize monitoring components
    metrics_collector = DefaultMetricsCollector()
    alert_manager = AlertManager()
    alert_manager.add_notifier(ConsoleAlertNotifier())
    lineage_tracker = LineageTracker()

    # Initialize data providers and processors
    try:
        from data_layer.fetcher.yfinance_fetcher import YFinanceFetcher
        from data_layer.processor.stock_processor import StockDataProcessor
    except ImportError:
        print("Please install required packages: pip install yfinance pandas")
        return

    fetcher = YFinanceFetcher(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    processor = StockDataProcessor()

    # Get historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    symbols = ["AAPL", "GOOGL", "MSFT"]  # Test multiple stocks
    
    try:
        historical_data = fetcher.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            fields=['date', 'open', 'high', 'low', 'close', 'volume']
        )
        
        latest_data = fetcher.get_latest_data(
            symbols=symbols,
            fields=['date', 'open', 'high', 'low', 'close', 'volume']
        )

        # Add source node information for lineage tracking
        fetcher.set_lineage("yfinance", "Yahoo Finance API")

        # Process historical data
        processed_data = processor.process(historical_data['data'])

        # Print processing result summary
        print("\nHistorical Data Summary:")
        for symbol in processed_data:
            print(f"{symbol}: {len(processed_data[symbol])} records")

        # Print latest data summary
        print("\nLatest Data:")
        for symbol in latest_data['data']:
            print(f"{symbol}: {latest_data['data'][symbol].iloc[-1].to_dict()}")

        # Print monitoring metrics
        print("\nMetrics:")
        for metric in metrics_collector.get_metrics():
            print(f"{metric.name}: {metric.value}")

        # Print data lineage
        print("\nData Lineage:")
        graph = lineage_tracker.export_graph()
        for edge in graph['edges']:
            print(f"{edge[0]} -> {edge[1]}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    demo_data_layer() 