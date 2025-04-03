# Data Layer Monitoring System

This module provides comprehensive data layer monitoring functionality, including performance metrics collection, alert management, and data lineage tracking.

## Main Features

### 1. Metrics Collection (Metrics Collection)

- Operation latency monitoring
- Data quality metrics
- Data volume statistics
- Custom metrics support

Usage example:
```python
from monitoring.metrics import MetricsCollector

collector = MetricsCollector()
collector.record_latency("data_fetch", 0.5)  # Record 0.5s latency
collector.record_quality("data_completeness", 0.95)  # Record 95% completeness
```

### 2. Alert Management (Alert Management)

- Multi-level alerts (INFO, WARNING, ERROR, CRITICAL)
- Extensible notification methods
- Alert history records
- Alert filtering and querying

Usage example:
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

### 3. Data Lineage Tracking (Data Lineage)

- Data flow visualization
- Upstream/downstream dependency analysis
- Data processing process tracking
- Impact scope analysis

Usage example:
```python
from monitoring.lineage import LineageTracker

tracker = LineageTracker()
tracker.record_node("data_source", "yfinance", {"type": "api"})
tracker.record_edge("data_source", "processor", {"operation": "clean"})
```

## Integration Methods

1. In the Data Fetching layer:
- Use MetricsCollector to record API call latency
- Record data source node information in LineageTracker
- Set data volume monitoring alert thresholds

2. In the Data Processing layer:
- Record processing time and resource usage
- Track data transformation and cleaning operations

3. In the Data Storage layer:
- Monitor storage operation performance
- Record data version changes
- Track data persistence process

## Best Practices

1. Monitoring Coverage
- Set up performance monitoring at key nodes
- Alert for important data quality indicators
- Record complete data processing chain

2. Alert Configuration
- Set up appropriate alert levels based on business importance
- Avoid too many low-level alerts
- Ensure critical alerts can be delivered in time

3. Data Lineage
- Update data node information in time
- Accurately record data transformation operations
- Analyze data dependency relations periodically

## Example Code

See `example.py` for complete usage examples. 