from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class FactorBase(ABC):
    """因子基类，定义因子计算的基本接口"""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        初始化因子
        
        Args:
            name: 因子名称
            params: 因子参数
        """
        self.name = name
        self.params = params or {}
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值
        
        Args:
            data: 输入数据，包含必要的行情数据
            
        Returns:
            pd.Series: 因子值序列
        """
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据是否满足因子计算要求
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 数据是否有效
        """
        pass
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        return data
    
    def postprocess(self, factor: pd.Series) -> pd.Series:
        """
        因子后处理（如标准化、去极值等）
        
        Args:
            factor: 原始因子值
            
        Returns:
            pd.Series: 处理后的因子值
        """
        return factor
    
    def get_required_fields(self) -> list:
        """
        获取因子计算所需的字段列表
        
        Returns:
            list: 所需字段列表
        """
        return []
    
    def get_factor_info(self) -> Dict[str, Any]:
        """
        获取因子信息
        
        Returns:
            Dict[str, Any]: 因子信息字典
        """
        return {
            'name': self.name,
            'params': self.params,
            'required_fields': self.get_required_fields()
        } 