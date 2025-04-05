from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

from data_layer.monitoring.metrics import MetricsCollector
from data_layer.monitoring.alerts import AlertManager
from data_layer.monitoring.lineage import LineageTracker

class DataProcessorBase(ABC):
    """Base data processor class"""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector = None,
        alert_manager: AlertManager = None,
        lineage_tracker: LineageTracker = None
    ):
        """Initialize base data processor"""
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.lineage_tracker = lineage_tracker
        self.source_node = None

    def set_lineage(self, source_id: str, source_name: str) -> None:
        """Set data lineage"""
        if self.lineage_tracker:
            self.source_node = self.lineage_tracker.create_source_node(
                source_id=source_id,
                source_name=source_name,
                metadata={
                    "processor": self.__class__.__name__,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )

    def record_lineage(self, operation: str, input_data: Dict[str, Any], output_data: pd.DataFrame) -> None:
        """Record data lineage"""
        if self.lineage_tracker and self.source_node:
            self.lineage_tracker.record_operation(
                source_node=self.source_node,
                operation=operation,
                input_data=input_data,
                output_data=output_data
            )

    def log_warning(self, message: str, metadata: Dict[str, Any] = None) -> None:
        """Record warning log"""
        if self.alert_manager:
            self.alert_manager.warning(
                title="Data Processing Warning",
                message=message,
                source=self.__class__.__name__,
                metadata=metadata
            )

    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data"""
        pass 