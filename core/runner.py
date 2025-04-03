"""Strategy executor responsible for loading configurations, executing strategies, and saving results"""

import os
import logging
import importlib
from typing import Dict, Any, List
import yaml
import pandas as pd
from datetime import datetime

from .strategy_base import StrategyBase
from .factor_base import FactorBase

class StrategyRunner:
    def __init__(self, config_path: str):
        """Initialize executor
        
        Args:
            config_path: Configuration file path
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logger()
        self.factors = self._load_factors()
        self.strategies = self._load_strategies()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> None:
        """Set up logger"""
        logger = logging.getLogger('strategy_runner')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        log_dir = os.path.join('logs', 'strategy')
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'strategy.log')
        )
        
        # Set log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    def _load_factors(self) -> List[Any]:
        """Load factors"""
        factors = []
        
        # Dynamically import factor modules
        for factor_config in self.config.get('factors', []):
            module = importlib.import_module(factor_config['module'])
            
            # Create factor instance
            factor_class = getattr(module, factor_config['class'])
            factor = factor_class(
                name=factor_config['name'],
                params=factor_config.get('params', {})
            )
            factors.append(factor)
        
        return factors
    
    def _load_strategies(self) -> List[Any]:
        """Load strategies"""
        strategies = []
        
        # Dynamically import strategy modules
        for strategy_config in self.config.get('strategies', []):
            module = importlib.import_module(strategy_config['module'])
            strategy_class = getattr(module, strategy_config['class'])
            strategy = strategy_class(
                name=strategy_config['name'],
                factors=self.factors,
                params=strategy_config.get('params', {})
            )
            strategies.append(strategy)
        
        return strategies
    
    def _load_data(self, data_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data"""
        # TODO: Implement data loading logic, support multiple data sources
        pass
    
    def run(self) -> None:
        """Execute strategy"""
        try:
            self.logger.info("Starting strategy execution")
            
            # Load data
            data = self._load_data(self.config['data'])
            self.logger.info(f"Data loading completed, shape: {data.shape}")
            
            # Load and execute strategies
            for strategy in self.strategies:
                self.logger.info(f"Starting strategy execution: {strategy.name}")
                
                # Data preprocessing
                processed_data = strategy.preprocess(data)
                
                # Validate data
                if not strategy.validate(processed_data):
                    self.logger.error(f"Strategy {strategy.name} data validation failed")
                    continue
                
                # Generate signals
                signals = strategy.generate_signals(processed_data)
                
                # Signal post-processing
                signals = strategy.postprocess(signals)
                
                # Save signals
                self._save_signals(signals, strategy.name)
                
                self.logger.info(f"Strategy {strategy.name} execution completed")
            
            self.logger.info("All strategies execution completed")
            
        except Exception as e:
            self.logger.error(f"Strategy execution error: {str(e)}")
            raise 