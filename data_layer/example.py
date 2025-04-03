from datetime import datetime, timedelta
from monitoring.metrics.collector import DefaultMetricsCollector
from monitoring.alerts.alert_manager import AlertManager, ConsoleAlertNotifier
from monitoring.lineage.tracker import LineageTracker

from fetcher.yfinance_provider import YFinanceProvider
from processor.market_data_processor import MarketDataProcessor

def main():
    # 初始化监控组件
    metrics_collector = DefaultMetricsCollector()
    alert_manager = AlertManager()
    alert_manager.add_notifier(ConsoleAlertNotifier())
    lineage_tracker = LineageTracker()
    
    # 初始化数据提供者和处理器
    data_provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    
    data_processor = MarketDataProcessor(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    
    try:
        # 获取历史数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        symbols = ["AAPL", "GOOGL", "MSFT"]  # 测试多个股票
        
        print("\nFetching historical data...")
        historical_data = data_provider.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            fields=['open', 'high', 'low', 'close', 'volume']
        )
        
        print("\nFetching latest data...")
        latest_data = data_provider.get_latest_data(
            symbols=symbols
        )
        
        # 添加源节点信息用于血缘追踪
        historical_data['source_node'] = data_provider.source_node
        
        # 处理历史数据
        processed_data = data_processor.process(historical_data)
        
        # 打印处理结果摘要
        print("\nProcessed Data Summary:")
        print(f"Symbols: {processed_data['symbols']}")
        print(f"Time Range: {processed_data['metadata']['start_date']} to {processed_data['metadata']['end_date']}")
        print(f"Added Features: {processed_data['metadata']['added_features']}")
        print(f"Total Records: {len(processed_data['data'])}")
        
        # 打印最新数据摘要
        print("\nLatest Data Summary:")
        for record in latest_data['data']:
            print(f"Symbol: {record['symbol']}, Close: {record['close']}, Date: {record['date']}")
        
        # 打印监控指标
        print("\nMetrics Summary:")
        for metric in metrics_collector.get_metrics():
            print(f"{metric.name}: {metric.value} ({metric.labels})")
        
        # 打印数据血缘
        print("\nData Lineage:")
        graph = lineage_tracker.export_graph()
        for edge in graph["edges"]:
            print(f"{edge.source.name} -> {edge.target.name} [{edge.operation.type.value}]")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 