from datetime import datetime
from metrics.collector import DefaultMetricsCollector
from alerts.alert_manager import AlertManager, AlertSeverity, ConsoleAlertNotifier
from lineage.tracker import LineageTracker, DataNode, Operation, OperationType

def demo_monitoring():
    # Initialize monitoring components
    metrics_collector = DefaultMetricsCollector()
    alert_manager = AlertManager()
    alert_manager.add_notifier(ConsoleAlertNotifier())
    lineage_tracker = LineageTracker()

    # Simulate data processing flow
    try:
        # 1. Record data retrieval performance
        metrics_collector.record_latency("fetch_yfinance_data", 150.5)  # 150.5ms
        metrics_collector.record_data_volume("yfinance", 1000)

        # 2. Create data lineage nodes
        source_node = DataNode(
            id="yfinance_raw",
            name="YFinance Raw Data",
            type="api",
            metadata={"provider": "yfinance", "market": "US"}
        )
        processed_node = DataNode(
            id="stock_daily",
            name="Processed Stock Daily Data",
            type="table",
            metadata={"schema": "finance", "update_frequency": "daily"}
        )
        
        lineage_tracker.add_node(source_node)
        lineage_tracker.add_node(processed_node)

        # 3. Record data processing operation
        operation = Operation(
            type=OperationType.TRANSFORM,
            timestamp=datetime.now(),
            operator="StockDataProcessor",
            details={"transformation": "clean_and_normalize"}
        )
        
        lineage_tracker.add_edge("yfinance_raw", "stock_daily", operation)

        # 4. Trigger quality check alert
        metrics_collector.record_data_quality(
            "price_range_check",
            success=True,
            details={"min": 10.0, "max": 100.0}
        )

    except Exception as e:
        # 5. Trigger error alert
        alert_manager.trigger_alert(
            title="Data Processing Failed",
            message=str(e),
            severity=AlertSeverity.ERROR,
            source="StockDataProcessor",
            metadata={"step": "transform", "input_records": 1000}
        )

    # Print collected metrics
    print("\nCollected Metrics:")
    for metric in metrics_collector.get_metrics():
        print(f"{metric.name}: {metric.value} ({metric.labels})")

    # Print data lineage graph
    print("\nData Lineage:")
    graph = lineage_tracker.export_graph()
    for edge in graph["edges"]:
        print(f"{edge.source.name} -> {edge.target.name} [{edge.operation.type.value}]")

if __name__ == "__main__":
    demo_monitoring() 