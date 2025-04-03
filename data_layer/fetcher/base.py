from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from monitoring.metrics.collector import MetricsCollector, DefaultMetricsCollector
from monitoring.alerts.alert_manager import AlertManager, AlertSeverity
from monitoring.lineage.tracker import LineageTracker, DataNode, Operation, OperationType

class DataFetcherBase(ABC):
    """数据获取基类"""
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None,
        lineage_tracker: Optional[LineageTracker] = None,
        config: Optional[Dict] = None
    ):
        self.metrics_collector = metrics_collector or DefaultMetricsCollector()
        self.alert_manager = alert_manager
        self.lineage_tracker = lineage_tracker
        self.source_node: Optional[DataNode] = None
        self.config = config or {}
        
    @abstractmethod
    def get_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        获取历史数据

        Args:
            symbols: 股票代码或代码列表
            start_date: 起始日期
            end_date: 结束日期
            fields: 需要获取的字段列表，默认获取所有字段

        Returns:
            Dict包含:
            - symbol: 股票代码
            - data: 包含历史数据的记录列表
            - metadata: 数据相关的元信息
        """
        pass

    @abstractmethod
    def get_latest_data(
        self,
        symbols: Union[str, List[str]],
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        获取最新数据

        Args:
            symbols: 股票代码或代码列表
            fields: 需要获取的字段列表，默认获取所有字段

        Returns:
            Dict包含最新的市场数据
        """
        pass

    @abstractmethod
    def validate_symbols(self, symbols: Union[str, List[str]]) -> List[str]:
        """
        验证股票代码是否有效

        Args:
            symbols: 待验证的股票代码或代码列表

        Returns:
            List[str]: 有效的股票代码列表
        """
        pass

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化数据格式

        Args:
            df: 原始数据DataFrame

        Returns:
            pd.DataFrame: 标准化后的数据
        """
        # 确保必要字段存在
        required_fields = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(f"数据缺少必要字段: {missing_fields}")
        
        # 标准化日期格式
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # 确保数值类型正确
        numeric_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        
        return df

    def validate_dates(self, start_date: Union[str, datetime], 
                      end_date: Union[str, datetime]) -> tuple:
        """
        验证并标准化日期格式

        Args:
            start_date: 起始日期
            end_date: 结束日期

        Returns:
            tuple: (datetime, datetime) 标准化后的起止日期
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        if start_date > end_date:
            raise ValueError("起始日期不能晚于结束日期")
            
        return start_date, end_date

    def _record_fetch_metrics(self, start_time: datetime, data: Dict[str, Any], source: str) -> None:
        """记录数据获取相关的指标"""
        duration = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics_collector.record_latency(f"fetch_{source}", duration)
        
        # 记录数据量
        if isinstance(data.get('data'), (list, tuple)):
            self.metrics_collector.record_data_volume(source, len(data['data']))
            
    def _setup_lineage(self, source_id: str, source_name: str, metadata: Dict[str, str]) -> None:
        """设置数据血缘节点"""
        if self.lineage_tracker:
            self.source_node = DataNode(
                id=source_id,
                name=source_name,
                type="api",
                metadata=metadata
            )
            self.lineage_tracker.add_node(self.source_node)
            
    def _handle_fetch_error(self, error: Exception, source: str) -> None:
        """处理数据获取错误"""
        if self.alert_manager:
            self.alert_manager.trigger_alert(
                title=f"Data Fetch Failed: {source}",
                message=str(error),
                severity=AlertSeverity.ERROR,
                source=self.__class__.__name__,
                metadata={"error_type": error.__class__.__name__}
            )
        raise error

    def validate_response(self, data: Dict[str, Any]) -> bool:
        """验证响应数据的基本结构"""
        return True  # 子类可以重写此方法实现具体的验证逻辑 