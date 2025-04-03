# 数据层监控系统

本模块提供了完整的数据层监控功能，包括性能指标收集、告警管理和数据血缘追踪。

## 主要功能

### 1. 指标收集 (Metrics Collection)

- 操作延迟监控
- 数据质量指标
- 数据量统计
- 自定义指标支持

使用示例：
```python
from metrics.collector import DefaultMetricsCollector

collector = DefaultMetricsCollector()
collector.record_latency("operation_name", duration_ms)
collector.record_data_quality("check_name", success, details)
collector.record_data_volume("data_source", record_count)
```

### 2. 告警管理 (Alert Management)

- 多级别告警（INFO、WARNING、ERROR、CRITICAL）
- 可扩展的通知方式
- 告警历史记录
- 告警过滤和查询

使用示例：
```python
from alerts.alert_manager import AlertManager, AlertSeverity, ConsoleAlertNotifier

manager = AlertManager()
manager.add_notifier(ConsoleAlertNotifier())
manager.trigger_alert(
    title="Error Title",
    message="Error Message",
    severity=AlertSeverity.ERROR,
    source="Component Name"
)
```

### 3. 数据血缘追踪 (Data Lineage)

- 数据流向可视化
- 上下游依赖分析
- 数据处理过程追踪
- 影响范围分析

使用示例：
```python
from lineage.tracker import LineageTracker, DataNode, Operation, OperationType

tracker = LineageTracker()
tracker.add_node(DataNode(...))
tracker.add_edge(source_id, target_id, Operation(...))
upstream_nodes = tracker.get_upstream_nodes(node_id)
```

## 集成方式

1. 在数据获取层 (Data Fetching) 中：
   - 使用 MetricsCollector 记录API调用延迟
   - 记录数据源节点信息到 LineageTracker
   - 设置数据量监控告警阈值

2. 在数据处理层 (Data Processing) 中：
   - 记录处理时间和资源使用情况
   - 追踪数据转换和清洗操作
   - 设置数据质量检查点

3. 在数据存储层 (Data Storage) 中：
   - 监控存储操作性能
   - 记录数据版本变更
   - 追踪数据持久化流程

## 最佳实践

1. 监控覆盖
   - 在关键节点设置性能监控
   - 对重要数据质量指标进行告警
   - 完整记录数据处理链路

2. 告警配置
   - 根据业务重要性设置合适的告警级别
   - 避免过多的低级别告警
   - 确保关键告警能及时送达

3. 血缘追踪
   - 及时更新数据节点信息
   - 准确记录数据转换操作
   - 定期分析数据依赖关系

## 示例代码

查看 `example.py` 了解完整的使用示例。 