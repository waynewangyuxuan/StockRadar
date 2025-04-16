#!/usr/bin/env python3
"""
StockRadar - Main entry point for trading and backtesting.

This script serves as the main entry point for running both backtesting
and live trading operations using the StockRadar framework.
"""

import argparse
import logging
import os
import sys
import importlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml
import coloredlogs

from core.config import ConfigManager
from core.runner import TradingRunner
from core.strategy_registry import StrategyRegistry
from core.factor_registry import FactorRegistry
from data_fetcher.yfinance_provider import YahooFinanceProvider
from data_processor.processor import DataProcessor

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    logging_config = config.get("logging", {})
    log_level = logging_config.get("level", "INFO")
    output_dir = logging_config.get("output_dir", "output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up colored logging for console
    coloredlogs.install(
        level=log_level,
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        field_styles={
            'asctime': {'color': 'green'},
            'levelname': {'color': 'yellow', 'bold': True},
            'name': {'color': 'blue'}
        },
        level_styles={
            'debug': {'color': 'magenta'},
            'info': {'color': 'green'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red'},
            'critical': {'color': 'red', 'bold': True}
        }
    )
    
    # Set up file logging
    if logging_config.get("file_output", True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{output_dir}/stockradar_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Add file handler to root logger
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    logger.info(f"Log file: {log_file if logging_config.get('file_output', True) else 'disabled'}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="StockRadar Trading System")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration loaded from {config_path}")
        
        # Log important configuration settings
        logger.info(f"Mode: {config['general']['mode']}")
        logger.info(f"Paper Trading: {config['general']['paper_trading']}")
        logger.info(f"Trading Universe: {config['data']['universe_type']}")
        if config['data']['universe_type'] == 'custom':
            logger.info(f"Symbols: {', '.join(config['data']['custom_universe'])}")
        logger.info(f"Data Interval: {config['data']['interval']}")
        logger.info(f"Initial Capital: ${config['portfolio']['initial_capital']:,.2f}")
        
        return config
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)

def register_strategies(strategy_registry: StrategyRegistry) -> None:
    """Register strategies from plugins/strategies directory."""
    strategy_dir = "plugins/strategies"
    if not os.path.exists(strategy_dir):
        logging.warning(f"Strategy directory {strategy_dir} not found")
        return

    for filename in os.listdir(strategy_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            strategy_name = filename[:-3]  # Remove .py extension
            try:
                # Import the module
                module = importlib.import_module(f"plugins.strategies.{strategy_name}")
                
                # Find the strategy class (should end with 'Strategy')
                strategy_class = None
                for attr_name in dir(module):
                    if attr_name.endswith('Strategy'):
                        strategy_class = getattr(module, attr_name)
                        break
                
                if strategy_class is None:
                    logging.warning(f"No strategy class found in {filename}")
                    continue
                
                # Register the strategy
                strategy_registry.register(strategy_name, strategy_class)
                logging.info(f"Registered strategy: {strategy_name}")
                
            except Exception as e:
                logging.warning(f"Failed to register strategy {strategy_name}: {str(e)}")

def register_factors(factor_registry: FactorRegistry) -> None:
    """Register factors from plugins/factors directory."""
    factor_dir = "plugins/factors"
    if not os.path.exists(factor_dir):
        logging.warning(f"Factor directory {factor_dir} not found")
        return

    for filename in os.listdir(factor_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            factor_name = filename[:-3]  # Remove .py extension
            try:
                # Import the module
                module = importlib.import_module(f"plugins.factors.{factor_name}")
                
                # Find the factor class (should end with 'Factor')
                factor_class = None
                for attr_name in dir(module):
                    if attr_name.endswith('Factor'):
                        factor_class = getattr(module, attr_name)
                        break
                
                if factor_class is None:
                    logging.warning(f"No factor class found in {filename}")
                    continue
                
                # Register the factor
                factor_registry.register(factor_name, factor_class)
                logging.info(f"Registered factor: {factor_name}")
                
            except Exception as e:
                logging.warning(f"Failed to register factor {factor_name}: {str(e)}")

def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        logger.info("Initializing trading system components...")
        
        strategy_registry = StrategyRegistry()
        factor_registry = FactorRegistry()
        
        # Register strategies and factors
        register_strategies(strategy_registry)
        register_factors(factor_registry)
        
        data_provider = YahooFinanceProvider(config.get("data", {}))
        data_processor = DataProcessor(config.get("data_processor", {}))
        
        # Create runner
        runner = TradingRunner(
            config=config,
            strategy_registry=strategy_registry,
            factor_registry=factor_registry,
            data_provider=data_provider,
            data_processor=data_processor
        )
        
        # Run based on mode
        mode = config["general"]["mode"]
        if mode == "live":
            logger.info("Starting live trading session")
            logger.info("Press Ctrl+C to stop")
            runner.run_live()
        elif mode == "backtest":
            logger.info("Starting backtest")
            results = runner.run_backtest()
            logger.info("Backtest completed")
            logger.info(f"Results saved to {config['logging']['output_dir']}")
        else:
            logger.error(f"Unknown mode: {mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        if mode == "live":
            runner.stop_live()
        logger.info("Trading system shutdown complete")
        
    except Exception as e:
        logger.exception(f"Error running trading system: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 