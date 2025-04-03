from typing import Dict, Any
import pandas as pd
import numpy as np

from .base import DataProcessorBase
from monitoring.metrics.collector import MetricsCollector
from monitoring.alerts.alert_manager import AlertManager
from monitoring.lineage.tracker import LineageTracker

class MarketDataProcessor(DataProcessorBase):
    """市场数据处理器"""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector = None,
        alert_manager: AlertManager = None,
        lineage_tracker: LineageTracker = None
    ):
        """初始化市场数据处理器"""
        super().__init__(metrics_collector, alert_manager, lineage_tracker)
        self._setup_lineage(
            source_id="market_data_processor",
            source_name="Market Data Processor",
            metadata={"processor": "market_data"}
        )
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理市场数据"""
        if data.empty:
            return data
            
        # 记录处理开始时间
        start_time = pd.Timestamp.now()
        
        try:
            # 计算技术指标
            processed_data = data.copy()
            
            # 计算日收益率
            processed_data['daily_return'] = processed_data.groupby('symbol')['close'].pct_change()
            
            # 计算移动平均线
            processed_data['ma5'] = processed_data.groupby('symbol')['close'].rolling(window=5).mean().reset_index(0, drop=True)
            processed_data['ma20'] = processed_data.groupby('symbol')['close'].rolling(window=20).mean().reset_index(0, drop=True)
            
            # 计算波动率
            processed_data['volatility'] = processed_data.groupby('symbol')['daily_return'].rolling(window=20).std().reset_index(0, drop=True)
            
            # 计算交易量变化
            processed_data['volume_change'] = processed_data.groupby('symbol')['volume'].pct_change()
            
            # 记录处理完成时间
            end_time = pd.Timestamp.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            # 记录性能指标
            if self.metrics_collector:
                self.metrics_collector.record_latency("market_data_processing", processing_time)
                self.metrics_collector.record_data_volume("processed_market_data", len(processed_data))
            
            # 记录数据血缘
            if self.lineage_tracker:
                self._record_lineage(
                    operation="process_market_data",
                    input_data={
                        "shape": data.shape,
                        "columns": list(data.columns)
                    },
                    output_data=processed_data
                )
            
            return processed_data
            
        except Exception as e:
            if self.alert_manager:
                self.alert_manager.error(
                    source=self.__class__.__name__,
                    message=f"处理市场数据时出错: {str(e)}",
                    metadata={"error_type": e.__class__.__name__}
                )
            raise 