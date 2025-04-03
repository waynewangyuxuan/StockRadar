from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List

@dataclass
class MetricPoint:
    """单个指标数据点"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]

class MetricsCollector(ABC):
    """指标收集器基类"""
    
    def __init__(self):
        self.metrics: List[MetricPoint] = []

    @abstractmethod
    def record_latency(self, operation: str, duration_ms: float) -> None:
        """记录操作延迟"""
        pass

    @abstractmethod
    def record_data_quality(self, check_name: str, success: bool, details: Dict[str, Any]) -> None:
        """记录数据质量检查结果"""
        pass

    @abstractmethod
    def record_data_volume(self, source: str, record_count: int) -> None:
        """记录数据量"""
        pass

    def get_metrics(self) -> List[MetricPoint]:
        """获取收集的所有指标"""
        return self.metrics

class DefaultMetricsCollector(MetricsCollector):
    """默认指标收集器实现"""
    
    def record_latency(self, operation: str, duration_ms: float) -> None:
        self.metrics.append(MetricPoint(
            name="operation_latency",
            value=duration_ms,
            timestamp=datetime.now(),
            labels={"operation": operation}
        ))

    def record_data_quality(self, check_name: str, success: bool, details: Dict[str, Any]) -> None:
        self.metrics.append(MetricPoint(
            name="data_quality_check",
            value=1.0 if success else 0.0,
            timestamp=datetime.now(),
            labels={
                "check_name": check_name,
                "details": str(details)
            }
        ))

    def record_data_volume(self, source: str, record_count: int) -> None:
        self.metrics.append(MetricPoint(
            name="data_volume",
            value=record_count,
            timestamp=datetime.now(),
            labels={"source": source}
        )) 