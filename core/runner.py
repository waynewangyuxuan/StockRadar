"""Strategy executor responsible for loading configurations, executing strategies, and saving results"""

import os
import logging
import importlib
from typing import Dict, Any, List, Type
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path

from .strategy_base import StrategyBase
from .factor_base import FactorBase

class StrategyRunner:
    """Main orchestrator class that runs the entire strategy pipeline.
    
    This class is responsible for:
    1. Loading configuration
    2. Fetching market data
    3. Computing factors
    4. Generating signals
    5. Saving results
    """
    
    def __init__(self, config_path: str):
        """Initialize the runner with a configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Will be initialized during run
        self.market_data = None
        self.factor_data = None
        self.signals = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        required_keys = ['data_source', 'factors', 'strategies', 'output']
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
            
        return config
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('StockRadar')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(f'logs/stockradar_{datetime.now():%Y%m%d_%H%M%S}.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def run(self) -> pd.DataFrame:
        """Execute the complete strategy pipeline.
        
        Returns:
            DataFrame containing the generated signals
        """
        try:
            self.logger.info("Starting strategy pipeline")
            
            # 1. Load market data
            self.logger.info("Fetching market data")
            self.market_data = self._fetch_market_data()
            
            # 2. Compute factors
            self.logger.info("Computing factors")
            self.factor_data = self._compute_factors()
            
            # 3. Generate signals
            self.logger.info("Generating signals")
            self.signals = self._generate_signals()
            
            # 4. Save results
            self.logger.info("Saving results")
            self._save_results()
            
            self.logger.info("Pipeline completed successfully")
            return self.signals
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
            
    def _fetch_market_data(self) -> pd.DataFrame:
        """Fetch market data using configured data provider."""
        # This will be implemented by the data_fetcher module
        raise NotImplementedError
        
    def _compute_factors(self) -> pd.DataFrame:
        """Compute all configured factors."""
        if not self.market_data is not None:
            raise ValueError("Market data must be loaded before computing factors")
            
        factor_data = self.market_data.copy()
        
        for factor_config in self.config['factors']:
            factor_class = self._import_class(factor_config['class'])
            factor = factor_class(factor_config.get('params', {}))
            
            self.logger.info(f"Computing factor: {factor.name}")
            factor_data = factor.calculate(factor_data)
            
        return factor_data
        
    def _generate_signals(self) -> pd.DataFrame:
        """Generate signals using all configured strategies."""
        if self.factor_data is None:
            raise ValueError("Factors must be computed before generating signals")
            
        signals_list = []
        
        for strategy_config in self.config['strategies']:
            strategy_class = self._import_class(strategy_config['class'])
            strategy = strategy_class(strategy_config.get('params', {}))
            
            self.logger.info(f"Running strategy: {strategy.name}")
            strategy_signals = strategy.generate_signals(self.market_data, self.factor_data)
            signals_list.append(strategy_signals)
            
        # Combine signals from all strategies
        return pd.concat(signals_list, axis=0, ignore_index=True)
        
    def _save_results(self):
        """Save the generated signals to configured output location."""
        if self.signals is None:
            raise ValueError("No signals to save")
            
        output_path = Path(self.config['output']['path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.signals.to_csv(output_path, index=False)
        self.logger.info(f"Saved signals to {output_path}")
        
    @staticmethod
    def _import_class(class_path: str) -> Type:
        """Dynamically import a class from its string path."""
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name) 