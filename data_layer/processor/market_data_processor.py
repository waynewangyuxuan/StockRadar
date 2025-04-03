from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from .base import DataProcessorBase
from monitoring.lineage.tracker import OperationType

class MarketDataProcessor(DataProcessorBase):
    """市场数据处理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_processor_node(
            processor_id="market_data_processor",
            processor_name="Market Data Processor",
            metadata={"type": "market_data", "version": "1.0"}
        )
    
    def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """处理市场数据"""
        try:
            start_time = datetime.now()
            
            # 转换为DataFrame
            df = pd.DataFrame(data['data'])
            
            # 数据清洗和转换
            df = self._clean_data(df)
            
            # 计算技术指标
            df = self._calculate_indicators(df)
            
            # 准备输出数据
            output_data = {
                'symbol': data['symbol'],
                'data': df.to_dict('records'),
                'metadata': {
                    **data['metadata'],
                    'processed_at': datetime.now().isoformat(),
                    'added_features': ['returns', 'volatility', 'ma_20', 'ma_50']
                }
            }
            
            # 记录处理指标
            self._record_processing_metrics(start_time, data, output_data, "market_data")
            
            # 记录数据血缘
            if data.get('source_node'):
                self._record_lineage(
                    data['source_node'],
                    output_data,
                    OperationType.TRANSFORM,
                    {"transformation": "market_data_processing"}
                )
            
            return output_data
            
        except Exception as e:
            self._handle_processing_error(e, "market_data", data)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗市场数据"""
        # 删除重复行
        df = df.drop_duplicates()
        
        # 处理缺失值
        df['Volume'] = df['Volume'].fillna(0)
        df = df.fillna(method='ffill')  # 用前一个值填充其他缺失值
        
        # 确保时间索引
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 计算日收益率
        df['returns'] = df['Close'].pct_change()
        
        # 计算波动率 (20日滚动标准差)
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # 计算移动平均线
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()
        
        # 记录指标计算的质量
        self._validate_indicators(df)
        
        return df
    
    def _validate_indicators(self, df: pd.DataFrame) -> None:
        """验证计算的指标"""
        # 检查是否有无限值
        inf_check = np.isinf(df[['returns', 'volatility', 'ma_20', 'ma_50']]).any().any()
        self.metrics_collector.record_data_quality(
            "infinite_values_check",
            success=not inf_check,
            details={"has_infinite_values": inf_check}
        )
        
        # 检查缺失值比例
        na_ratio = df[['returns', 'volatility', 'ma_20', 'ma_50']].isna().mean()
        self.metrics_collector.record_data_quality(
            "missing_values_ratio",
            success=na_ratio.max() < 0.1,  # 允许最多10%的缺失值
            details={"missing_ratios": na_ratio.to_dict()}
        ) 