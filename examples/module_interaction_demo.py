from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any

from data_layer.fetcher.yfinance_provider import YFinanceProvider
from data_layer.processor.market_data_processor import MarketDataProcessor
from monitoring.metrics.collector import DefaultMetricsCollector
from monitoring.alerts.alert_manager import AlertManager, ConsoleAlertNotifier
from monitoring.lineage.tracker import LineageTracker

def print_section(title: str):
    """打印分隔标题"""
    print("\n" + "="*50)
    print(f"{title}")
    print("="*50)

def print_data_info(data: Dict[str, Any], title: str = "Data Info"):
    """打印数据信息"""
    print_section(title)
    print(f"Symbols: {data.get('symbols', [])}")
    if isinstance(data.get('data'), pd.DataFrame):
        print("\nDataFrame Info:")
        print(data['data'].info())
        print("\nFirst few rows:")
        print(data['data'].head())
    else:
        print("No data available")

def main():
    """主函数"""
    # 初始化监控组件
    metrics_collector = DefaultMetricsCollector()
    alert_manager = AlertManager(notifiers=[ConsoleAlertNotifier()])
    lineage_tracker = LineageTracker()

    print_section("1. 初始化组件")
    print("Metrics Collector:", metrics_collector.__class__.__name__)
    print("Alert Manager:", alert_manager.__class__.__name__)
    print("Lineage Tracker:", lineage_tracker.__class__.__name__)

    # 初始化数据提供者
    print_section("2. 初始化数据提供者")
    provider = YFinanceProvider(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    print("Provider:", provider.__class__.__name__)
    print("Source Node ID:", provider.source_node.id if provider.source_node else "None")

    # 获取历史数据
    print_section("3. 获取历史数据")
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()
    
    try:
        historical_data = provider.get_historical_data(
            symbols=['AAPL', 'MSFT'],
            start_date=start_date,
            end_date=end_date
        )
        print_data_info(historical_data, "Historical Data")
    except Exception as e:
        print(f"Error fetching historical data: {str(e)}")

    # 获取最新数据
    print_section("4. 获取最新数据")
    try:
        latest_data = provider.get_latest_data(symbols=['AAPL', 'MSFT'])
        print_data_info(latest_data, "Latest Data")
    except Exception as e:
        print(f"Error fetching latest data: {str(e)}")

    # 初始化数据处理器
    print_section("5. 初始化数据处理器")
    processor = MarketDataProcessor(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        lineage_tracker=lineage_tracker
    )
    print("Processor:", processor.__class__.__name__)

    # 处理数据
    print_section("6. 处理数据")
    if 'historical_data' in locals() and not historical_data['data'].empty:
        try:
            processed_data = processor.process_data(historical_data['data'])
            print_data_info({'data': processed_data}, "Processed Data")
        except Exception as e:
            print(f"Error processing data: {str(e)}")

    # 查看血缘关系
    print_section("7. 数据血缘关系")
    if lineage_tracker.nodes:
        print("\nNodes:")
        for node_id, node in lineage_tracker.nodes.items():
            print(f"- {node_id}: {node.name} ({node.type})")
        
        print("\nEdges:")
        for edge in lineage_tracker.edges:
            print(f"- {edge[0]} -> {edge[1]}")
    else:
        print("No lineage information available")

    # 查看指标
    print_section("8. 性能指标")
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