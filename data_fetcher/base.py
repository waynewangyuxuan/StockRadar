from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd

class DataProviderBase(ABC):
    """数据提供者基类，定义统一的数据获取接口"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据提供者
        
        Args:
            config: 数据源配置参数
        """
        self.config = config or {}
    
    @abstractmethod
    def get_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            symbols: 股票代码或代码列表
            start_date: 起始日期
            end_date: 结束日期
            fields: 需要获取的字段列表，默认获取所有字段
            
        Returns:
            pd.DataFrame: 包含历史数据的DataFrame，必须包含以下列：
                - symbol: 股票代码
                - date: 交易日期
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - volume: 成交量
                其他可选列可由具体数据源提供
        """
        pass
    
    @abstractmethod
    def get_latest_data(
        self,
        symbols: Union[str, List[str]],
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取最新数据
        
        Args:
            symbols: 股票代码或代码列表
            fields: 需要获取的字段列表，默认获取所有字段
            
        Returns:
            pd.DataFrame: 包含最新数据的DataFrame
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
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化数据格式
        
        Args:
            data: 原始数据DataFrame
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        # 确保必要字段存在
        required_fields = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_fields = [field for field in required_fields if field not in data.columns]
        if missing_fields:
            raise ValueError(f"数据缺少必要字段: {missing_fields}")
        
        # 标准化日期格式
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        # 确保数值类型正确
        numeric_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in numeric_fields:
            data[field] = pd.to_numeric(data[field], errors='coerce')
        
        return data
    
    def validate_dates(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> tuple:
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