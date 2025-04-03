from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

from monitoring.metrics.collector import MetricsCollector
from monitoring.alerts.alert_manager import AlertManager
from monitoring.lineage.tracker import LineageTracker

class DataProcessorBase(ABC):
    """数据处理器基类"""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector = None,
        alert_manager: AlertManager = None,
        lineage_tracker: LineageTracker = None
    ):
        """初始化数据处理器基类"""
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.lineage_tracker = lineage_tracker
        self.source_node = None

    def _setup_lineage(self, source_id: str, source_name: str, metadata: Dict[str, Any] = None):
        """设置数据血缘"""
        if self.lineage_tracker:
            self.source_node = self.lineage_tracker.create_source_node(
                source_id=source_id,
                source_name=source_name,
                metadata=metadata
            )

    def _record_lineage(self, operation: str, input_data: Dict[str, Any], output_data: pd.DataFrame):
        """记录数据血缘"""
        if self.lineage_tracker and self.source_node:
            self.lineage_tracker.record_operation(
                source_node=self.source_node,
                operation=operation,
                input_data=input_data,
                output_data=output_data
            )

    def _log_warning(self, message: str):
        """记录警告日志"""
        if self.alert_manager:
            self.alert_manager.warning(
                source=self.__class__.__name__,
                message=message
            )

    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理数据"""
        pass 