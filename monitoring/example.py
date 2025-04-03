from datetime import datetime
from metrics.collector import DefaultMetricsCollector
from alerts.alert_manager import AlertManager, AlertSeverity, ConsoleAlertNotifier
from lineage.tracker import LineageTracker, DataNode, Operation, OperationType

def demo_monitoring():
    # 初始化监控组件
    metrics_collector = DefaultMetricsCollector()
    alert_manager = AlertManager()
    alert_manager.add_notifier(ConsoleAlertNotifier())
    lineage_tracker = LineageTracker()

    # 模拟数据处理流程
    try:
        # 1. 记录数据获取性能
        metrics_collector.record_latency("fetch_yfinance_data", 150.5)  # 150.5ms
        metrics_collector.record_data_volume("yfinance", 1000)

        # 2. 创建数据血缘节点
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

        # 3. 记录数据处理操作
        operation = Operation(
            type=OperationType.TRANSFORM,
            timestamp=datetime.now(),
            operator="StockDataProcessor",
            details={"transformation": "clean_and_normalize"}
        )
        
        lineage_tracker.add_edge("yfinance_raw", "stock_daily", operation)

        # 4. 触发质量检查告警
        metrics_collector.record_data_quality(
            "price_range_check",
            success=True,
            details={"min": 10.0, "max": 100.0}
        )

    except Exception as e:
        # 5. 触发错误告警
        alert_manager.trigger_alert(
            title="Data Processing Failed",
            message=str(e),
            severity=AlertSeverity.ERROR,
            source="StockDataProcessor",
            metadata={"step": "transform", "input_records": 1000}
        )

    # 打印收集的指标
    print("\nCollected Metrics:")
    for metric in metrics_collector.get_metrics():
        print(f"{metric.name}: {metric.value} ({metric.labels})")

    # 打印数据血缘图
    print("\nData Lineage:")
    graph = lineage_tracker.export_graph()
    for edge in graph["edges"]:
        print(f"{edge.source.name} -> {edge.target.name} [{edge.operation.type.value}]")

if __name__ == "__main__":
    demo_monitoring() 