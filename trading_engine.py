"""
Unified trading engine for both backtesting and live trading.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from backtester.evaluator import StrategyEvaluator
from backtester.visualization import BacktestVisualizer
from strategy_engine.strategy_interface import StrategyInterface
from strategy_engine.portfolio_manager import PortfolioManager
from strategy_engine.risk_manager import RiskManager
from strategy_engine.execution_engine import ExecutionEngine
from data_engine.data_loader import DataLoader
from data_engine.data_processor import DataProcessor

class TradingEngine:
    """Unified trading engine for both backtesting and live trading."""
    
    def __init__(self, config_path: str):
        """
        Initialize the trading engine.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_components()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config['logging']
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_components(self):
        """Initialize all trading components."""
        # Initialize data components
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        
        # Initialize strategy components
        self.strategies = self._load_strategies()
        
        # Initialize portfolio and risk management
        self.portfolio_manager = PortfolioManager(
            max_positions=self.config['portfolio']['max_positions'],
            position_size=self.config['portfolio']['position_size']
        )
        
        self.risk_manager = RiskManager(
            max_drawdown=self.config['risk']['max_drawdown'],
            stop_loss=self.config['risk']['stop_loss'],
            volatility_target=self.config['risk']['volatility_target']
        )
        
        # Initialize execution engine
        self.execution_engine = ExecutionEngine(
            broker=self.config['execution']['broker'],
            commission=self.config['execution']['commission'],
            slippage=self.config['execution']['slippage']
        )
        
        # Initialize backtesting components if needed
        if self.config['mode'] == 'backtest':
            self.evaluator = StrategyEvaluator()
            self.visualizer = BacktestVisualizer()
            
    def _load_strategies(self) -> List[StrategyInterface]:
        """Load and initialize trading strategies."""
        strategies = []
        for strategy_config in self.config['strategies']:
            if strategy_config['enabled']:
                strategy_class = self._get_strategy_class(strategy_config['name'])
                strategy = strategy_class(**strategy_config['params'])
                strategies.append(strategy)
        return strategies
        
    def _get_strategy_class(self, strategy_name: str):
        """Get strategy class by name."""
        # This would be implemented to return the appropriate strategy class
        # based on the strategy name from the config
        pass
        
    def run(self):
        """Run the trading engine in either backtest or live mode."""
        if self.config['mode'] == 'backtest':
            self._run_backtest()
        else:
            self._run_live()
            
    def _run_backtest(self):
        """Run backtesting simulation."""
        self.logger.info("Starting backtest...")
        
        # Load historical data
        data = self.data_loader.load_historical_data(
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date'],
            universe=self.config['data']['universe'],
            custom_universe=self.config['data']['custom_universe']
        )
        
        # Process data
        processed_data = self.data_processor.process_data(data)
        
        # Run strategy evaluation
        results = self.evaluator.evaluate(processed_data)
        
        # Generate visualizations
        self.visualizer.plot_results(results)
        
        self.logger.info("Backtest completed.")
        
    def _run_live(self):
        """Run live trading."""
        self.logger.info("Starting live trading...")
        
        while True:
            try:
                # Load latest market data
                data = self.data_loader.load_live_data(
                    universe=self.config['data']['universe'],
                    custom_universe=self.config['data']['custom_universe']
                )
                
                # Process data
                processed_data = self.data_processor.process_data(data)
                
                # Generate signals from all strategies
                signals = self._generate_signals(processed_data)
                
                # Get portfolio recommendations
                portfolio = self.portfolio_manager.get_portfolio_recommendations(
                    signals,
                    processed_data
                )
                
                # Apply risk management
                portfolio = self.risk_manager.apply_risk_management(portfolio)
                
                # Execute trades
                self.execution_engine.execute_trades(portfolio)
                
                # Wait for next update
                self._wait_for_next_update()
                
            except Exception as e:
                self.logger.error(f"Error in live trading: {str(e)}")
                # Implement error handling and recovery logic
                
    def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from all strategies."""
        signals = pd.DataFrame(index=data.index)
        
        for strategy in self.strategies:
            strategy_signals = strategy.generate_signals(data)
            signals[f"{strategy.__class__.__name__}_signal"] = strategy_signals['signal']
            
        return signals
        
    def _wait_for_next_update(self):
        """Wait for the next update based on rebalance frequency."""
        # Implement waiting logic based on rebalance_frequency
        pass 