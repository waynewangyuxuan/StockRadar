from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

class StrategyBase(ABC):
    """策略基类，定义策略执行的基本接口"""
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        初始化策略
        
        Args:
            name: 策略名称
            params: 策略参数
        """
        self.name = name
        self.params = params or {}
        self.factors: List[Any] = []  # 策略使用的因子列表
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 输入数据，包含因子值
            
        Returns:
            pd.DataFrame: 信号DataFrame，包含信号值和相关元数据
        """
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据是否满足策略要求
        
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
    
    def postprocess(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        信号后处理（如信号平滑、风险控制等）
        
        Args:
            signals: 原始信号
            
        Returns:
            pd.DataFrame: 处理后的信号
        """
        return signals
    
    def add_factor(self, factor: Any) -> None:
        """
        添加因子到策略
        
        Args:
            factor: 因子实例
        """
        self.factors.append(factor)
    
    def get_required_factors(self) -> List[str]:
        """
        获取策略所需的因子列表
        
        Returns:
            List[str]: 所需因子名称列表
        """
        return [factor.name for factor in self.factors]
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            Dict[str, Any]: 策略信息字典
        """
        return {
            'name': self.name,
            'params': self.params,
            'required_factors': self.get_required_factors()
        }
    
    def save_signals(self, signals: pd.DataFrame, output_path: str) -> None:
        """
        保存信号到文件
        
        Args:
            signals: 信号DataFrame
            output_path: 输出文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.name}_{timestamp}.csv"
        signals.to_csv(f"{output_path}/{filename}") 