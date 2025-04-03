from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
import time

@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float

class MetricsCollector(ABC):
    """Base metrics collector class"""
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = []
    
    @abstractmethod
    def record_latency(self, operation: str, duration_ms: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record latency metric"""
        pass
    
    @abstractmethod
    def record_data_quality(self, check_name: str, success: bool, details: Optional[Dict[str, Any]] = None) -> None:
        """Record data quality metric"""
        pass
    
    @abstractmethod
    def record_data_volume(self, source: str, count: int, labels: Optional[Dict[str, str]] = None) -> None:
        """Record data volume metric"""
        pass
    
    def get_metrics(self) -> List[MetricPoint]:
        """Get all collected metrics"""
        return self.metrics

class DefaultMetricsCollector(MetricsCollector):
    """Default metrics collector implementation"""
    
    def record_latency(self, operation: str, duration_ms: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record latency metric"""
        self.metrics.append(MetricPoint(
            name=f"{operation}_latency_ms",
            value=duration_ms,
            labels=labels or {},
            timestamp=time.time()
        ))
    
    def record_data_volume(self, source: str, count: int, labels: Optional[Dict[str, str]] = None) -> None:
        """Record data volume metric"""
        self.metrics.append(MetricPoint(
            name=f"{source}_data_volume",
            value=count,
            labels=labels or {},
            timestamp=time.time()
        ))
    
    def record_data_quality(self, check_name: str, success: bool, details: Optional[Dict[str, Any]] = None) -> None:
        """Record data quality metric"""
        self.metrics.append(MetricPoint(
            name=f"{check_name}_quality",
            value=1.0 if success else 0.0,
            labels=details or {},
            timestamp=time.time()
        )) 