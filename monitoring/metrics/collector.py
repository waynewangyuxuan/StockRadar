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
        """初始化指标收集器"""
        self.metrics = {}

    @abstractmethod
    def record_latency(self, metric_name: str, duration_ms: float):
        """记录延迟指标"""
        pass

    @abstractmethod
    def record_data_quality(self, metric_name: str, success: bool, details: Dict[str, Any] = None):
        """记录数据质量指标"""
        pass

    @abstractmethod
    def record_data_volume(self, metric_name: str, volume: int):
        """记录数据量指标"""
        pass

    def get_metrics(self) -> List[MetricPoint]:
        """获取收集的所有指标"""
        return self.metrics

class DefaultMetricsCollector(MetricsCollector):
    """默认指标收集器"""
    
    def record_latency(self, metric_name: str, duration_ms: float):
        """记录延迟指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {
                'count': 0,
                'total': 0,
                'average': 0
            }
        
        metric = self.metrics[metric_name]
        metric['count'] += 1
        metric['total'] += duration_ms
        metric['average'] = metric['total'] / metric['count']
    
    def record_data_volume(self, metric_name: str, volume: int):
        """记录数据量指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {
                'count': 0,
                'total': 0,
                'average': 0
            }
        
        metric = self.metrics[metric_name]
        metric['count'] += 1
        metric['total'] += volume
        metric['average'] = metric['total'] / metric['count']
    
    def record_data_quality(self, metric_name: str, success: bool, details: Dict[str, Any] = None):
        """记录数据质量指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {
                'count': 0,
                'success_count': 0,
                'success_rate': 0,
                'details': []
            }
        
        metric = self.metrics[metric_name]
        metric['count'] += 1
        if success:
            metric['success_count'] += 1
        metric['success_rate'] = metric['success_count'] / metric['count']
        
        if details:
            metric['details'].append({
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'details': details
            }) 