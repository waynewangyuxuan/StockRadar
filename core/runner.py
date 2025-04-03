import yaml
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import importlib
import logging
from datetime import datetime

from .strategy_base import StrategyBase
from .factor_base import FactorBase

class StrategyRunner:
    """策略执行器，负责加载配置、执行策略和保存结果"""
    
    def __init__(self, config_path: str):
        """
        初始化执行器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logger()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger('StrategyRunner')
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建文件处理器
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f'logs/strategy_{timestamp}.log')
        file_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_factor(self, factor_config: Dict[str, Any]) -> FactorBase:
        """加载因子"""
        module_path = factor_config['module']
        class_name = factor_config['class']
        
        # 动态导入因子模块
        module = importlib.import_module(module_path)
        factor_class = getattr(module, class_name)
        
        # 创建因子实例
        return factor_class(
            name=factor_config['name'],
            params=factor_config.get('params', {})
        )
    
    def _load_strategy(self, strategy_config: Dict[str, Any]) -> StrategyBase:
        """加载策略"""
        module_path = strategy_config['module']
        class_name = strategy_config['class']
        
        # 动态导入策略模块
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        
        # 创建策略实例
        strategy = strategy_class(
            name=strategy_config['name'],
            params=strategy_config.get('params', {})
        )
        
        # 添加因子
        for factor_config in strategy_config.get('factors', []):
            factor = self._load_factor(factor_config)
            strategy.add_factor(factor)
        
        return strategy
    
    def _load_data(self, data_config: Dict[str, Any]) -> pd.DataFrame:
        """加载数据"""
        # TODO: 实现数据加载逻辑，支持多种数据源
        data_path = data_config['path']
        return pd.read_csv(data_path)
    
    def run(self) -> None:
        """执行策略"""
        try:
            self.logger.info("开始执行策略")
            
            # 加载数据
            data = self._load_data(self.config['data'])
            self.logger.info(f"加载数据完成，数据形状: {data.shape}")
            
            # 加载并执行策略
            for strategy_config in self.config['strategies']:
                strategy = self._load_strategy(strategy_config)
                self.logger.info(f"开始执行策略: {strategy.name}")
                
                # 数据预处理
                data = strategy.preprocess(data)
                
                # 验证数据
                if not strategy.validate(data):
                    self.logger.error(f"策略 {strategy.name} 数据验证失败")
                    continue
                
                # 生成信号
                signals = strategy.generate_signals(data)
                
                # 信号后处理
                signals = strategy.postprocess(signals)
                
                # 保存信号
                strategy.save_signals(signals, self.config['output']['path'])
                self.logger.info(f"策略 {strategy.name} 执行完成")
            
            self.logger.info("所有策略执行完成")
            
        except Exception as e:
            self.logger.error(f"策略执行出错: {str(e)}", exc_info=True)
            raise 