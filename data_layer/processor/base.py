from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from monitoring.metrics.collector import MetricsCollector, DefaultMetricsCollector
from monitoring.alerts.alert_manager import AlertManager, AlertSeverity
from monitoring.lineage.tracker import LineageTracker, DataNode, Operation, OperationType

class DataProcessorBase(ABC):
    """数据处理基类"""
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None,
        lineage_tracker: Optional[LineageTracker] = None
    ):
        self.metrics_collector = metrics_collector or DefaultMetricsCollector()
        self.alert_manager = alert_manager
        self.lineage_tracker = lineage_tracker
        self.processor_node: Optional[DataNode] = None
        
    def setup_processor_node(self, processor_id: str, processor_name: str, metadata: Dict[str, str]) -> None:
        """设置处理器节点"""
        if self.lineage_tracker:
            self.processor_node = DataNode(
                id=processor_id,
                name=processor_name,
                type="processor",
                metadata=metadata
            )
            self.lineage_tracker.add_node(self.processor_node)
    
    @abstractmethod
    def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """处理数据的抽象方法"""
        pass
    
    def _record_processing_metrics(self, start_time: datetime, input_data: Dict[str, Any], 
                                 output_data: Dict[str, Any], processor_name: str) -> None:
        """记录数据处理相关的指标"""
        duration = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics_collector.record_latency(f"process_{processor_name}", duration)
        
        # 记录输入输出数据量
        if isinstance(input_data.get('data'), (list, tuple)):
            self.metrics_collector.record_data_volume(f"{processor_name}_input", len(input_data['data']))
        if isinstance(output_data.get('data'), (list, tuple)):
            self.metrics_collector.record_data_volume(f"{processor_name}_output", len(output_data['data']))
    
    def _record_lineage(self, input_node: DataNode, output_data: Dict[str, Any], 
                       operation_type: OperationType, details: Dict[str, str]) -> None:
        """记录数据处理的血缘关系"""
        if self.lineage_tracker and self.processor_node:
            operation = Operation(
                type=operation_type,
                timestamp=datetime.now(),
                operator=self.__class__.__name__,
                details=details
            )
            self.lineage_tracker.add_edge(input_node.id, self.processor_node.id, operation)
    
    def _handle_processing_error(self, error: Exception, processor_name: str, 
                               input_data: Dict[str, Any]) -> None:
        """处理数据处理错误"""
        if self.alert_manager:
            self.alert_manager.trigger_alert(
                title=f"Data Processing Failed: {processor_name}",
                message=str(error),
                severity=AlertSeverity.ERROR,
                source=self.__class__.__name__,
                metadata={
                    "error_type": error.__class__.__name__,
                    "input_metadata": input_data.get('metadata', {})
                }
            )
        raise error 