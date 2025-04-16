"""
Trading Runner Module

This module contains the TradingRunner class which orchestrates the execution
of trading strategies in both backtesting and live trading modes.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import os
import time
from threading import Thread, Event
import importlib

from .config import ConfigManager
from .strategy_registry import StrategyRegistry
from .factor_registry import FactorRegistry
from data_fetcher.base import DataProviderBase as DataProvider
from data_processor.processor import DataProcessor
from backtester.evaluator import StrategyEvaluator
from backtester.visualization import BacktestVisualizer

class TradingRunner:
    """
    Trading Runner class that orchestrates the execution of trading strategies.
    
    This class manages:
    1. Loading market data
    2. Running strategies
    3. Managing portfolios
    4. Executing trades
    5. Generating performance reports
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        strategy_registry: StrategyRegistry,
        factor_registry: FactorRegistry,
        data_provider: DataProvider,
        data_processor: DataProcessor
    ):
        """
        Initialize the TradingRunner.
        
        Args:
            config: Configuration dictionary
            strategy_registry: Registry of available strategies
            factor_registry: Registry of available factors
            data_provider: Data provider instance
            data_processor: Data processor instance
        """
        self.config = config
        self.strategy_registry = strategy_registry
        self.factor_registry = factor_registry
        self.data_provider = data_provider
        self.data_processor = data_processor
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.factors = []
        self.strategies = []
        self._initialize_components()
        
        # Live trading attributes
        self.is_running = False
        self.stop_event = Event()
        self.trading_thread = None
        
    def _initialize_components(self) -> None:
        """Initialize trading components based on configuration."""
        # Load enabled factors
        self._load_factors()
        
        # Load enabled strategies
        self._load_strategies()
        
    def _load_factors(self) -> None:
        """Load enabled factors from configuration."""
        factors_config = self.config.get('factors', {})
        for factor_name, factor_config in factors_config.items():
            if factor_config.get('enabled', False):
                try:
                    # Import factor class
                    module_name = f"plugins.factors.{factor_name}"
                    module = importlib.import_module(module_name)
                    
                    # Get the class name from the module
                    # First try the exact class name from the file
                    class_name = f"{factor_name.title().replace('_', '')}Factor"
                    if not hasattr(module, class_name):
                        # If not found, try alternative naming patterns
                        if factor_name == 'ma_factor':
                            class_name = 'MAFactor'
                        elif factor_name == 'volume_spike_factor':
                            class_name = 'VolumeSpikeFactor'
                        else:
                            # Try to find any class that ends with 'Factor'
                            factor_classes = [c for c in dir(module) if c.endswith('Factor')]
                            if factor_classes:
                                class_name = factor_classes[0]
                            else:
                                self.logger.warning(f"No factor class found in {module_name}")
                                continue
                    
                    factor_class = getattr(module, class_name)
                    factor = factor_class(factor_config)
                    self.factors.append(factor)
                    self.logger.info(f"Loaded factor: {factor_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load factor {factor_name}: {str(e)}")
                    continue
        
        if not self.factors:
            self.logger.warning("No factors loaded")
    
    def _load_strategies(self) -> None:
        """Load enabled strategies from configuration."""
        strategies_config = self.config.get("strategies", [])
        
        for strategy_config in strategies_config:
            if strategy_config.get("enabled", False):
                strategy_name = strategy_config.get("name")
                if strategy_name:
                    try:
                        # Get strategy from registry
                        strategy_class = self.strategy_registry.get_strategy(strategy_name)
                        if strategy_class:
                            # Create strategy instance with parameters
                            strategy = strategy_class(strategy_config.get("parameters", {}))
                            self.strategies.append(strategy)
                            self.logger.info(f"Loaded strategy: {strategy_name}")
                        else:
                            self.logger.warning(f"Strategy {strategy_name} not found in registry")
                    except Exception as e:
                        self.logger.warning(f"Failed to load strategy {strategy_name}: {str(e)}")
        
        if not self.strategies:
            self.logger.warning("No strategies loaded")
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run backtesting simulation.
        
        Returns:
            Dict containing backtest results
        """
        self.logger.info("Starting backtest")
        
        # Load historical data
        data = self._load_historical_data()
        
        # Process data with factors
        processed_data = self._process_data(data)
        
        # Initialize evaluator
        evaluator = StrategyEvaluator(
            initial_capital=self.config["portfolio"]["initial_capital"],
            commission=self.config["portfolio"]["commission"],
            slippage=self.config["portfolio"]["slippage"]
        )
        
        # Run strategies
        for strategy in self.strategies:
            self.logger.info(f"Running strategy: {strategy}")
            
            # Generate signals
            signals = strategy.generate_signals(processed_data)
            
            # Evaluate strategy
            results = evaluator.evaluate(processed_data, signals)
            
            # Generate report
            self._generate_backtest_report(results, strategy)
        
        return evaluator.get_results()
    
    def run_live(self) -> None:
        """
        Run live trading.
        
        This method:
        1. Initializes the live trading environment
        2. Starts a background thread for trading
        3. Monitors the trading thread
        4. Handles graceful shutdown
        """
        if self.is_running:
            self.logger.warning("Live trading is already running")
            return
            
        self.logger.info("Starting live trading")
        
        # Initialize live trading components
        self._initialize_live_trading()
        
        # Start trading thread
        self.stop_event.clear()
        self.is_running = True
        self.trading_thread = Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        try:
            # Monitor the trading thread
            while self.is_running and not self.stop_event.is_set():
                if not self.trading_thread.is_alive():
                    self.logger.error("Trading thread died unexpectedly")
                    break
                    
                # Sleep to avoid high CPU usage
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
            self.stop_live()
            
    def stop_live(self) -> None:
        """Stop live trading gracefully."""
        if not self.is_running:
            return
            
        self.logger.info("Stopping live trading")
        self.stop_event.set()
        
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=10)
            
        self.is_running = False
        self.logger.info("Live trading stopped")
        
    def _initialize_live_trading(self) -> None:
        """Initialize components for live trading."""
        # Initialize portfolio
        self.portfolio = {
            'cash': self.config["portfolio"]["initial_capital"],
            'positions': {},
            'trades': []
        }
        
        # Initialize risk manager
        self.risk_manager = {
            'max_position_size': self.config["risk"]["max_position_size"],
            'max_drawdown': self.config["risk"]["max_drawdown"],
            'stop_loss': self.config["risk"]["stop_loss"],
            'take_profit': self.config["risk"]["take_profit"],
            'max_positions': self.config["risk"]["max_positions"]
        }
        
        # Initialize execution provider
        execution_config = self.config.get("execution", {})
        self.execution_provider = execution_config.get("provider", "paper")
        self.paper_trading = execution_config.get("paper_trading", True)
        
        self.logger.info(f"Initialized live trading with {self.execution_provider} provider")
        self.logger.info(f"Paper trading: {self.paper_trading}")
        
    def _trading_loop(self) -> None:
        """
        Main trading loop.
        
        This method:
        1. Fetches latest market data
        2. Processes data with factors
        3. Generates signals from strategies
        4. Executes trades based on signals
        5. Updates portfolio
        6. Manages risk
        """
        self.logger.info("Trading loop started")
        
        while not self.stop_event.is_set():
            try:
                # Fetch latest market data
                data = self._fetch_live_data()
                
                # Process data with factors
                processed_data = self._process_data(data)
                
                # Generate signals from strategies
                signals = self._generate_live_signals(processed_data)
                
                # Execute trades based on signals
                self._execute_trades(signals, processed_data)
                
                # Update portfolio
                self._update_portfolio(processed_data)
                
                # Manage risk
                self._manage_risk(processed_data)
                
                # Sleep until next update
                self._wait_for_next_update()
                
            except Exception as e:
                self.logger.exception(f"Error in trading loop: {str(e)}")
                # Sleep to avoid rapid retries on persistent errors
                time.sleep(60)
                
    def _fetch_live_data(self) -> pd.DataFrame:
        """
        Fetch latest market data for live trading.
        
        Returns:
            DataFrame with latest market data
        """
        # Get symbols to trade
        symbols = self._get_universe_symbols()
        
        # Get data interval
        interval = self.config["data"].get("interval", "1d")
        
        # Fetch latest data
        data = self.data_provider.fetch_live_data(
            symbols=symbols,
            interval=interval
        )
        
        return data
        
    def _generate_live_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals from strategies for live trading.
        
        Args:
            data: Processed market data
            
        Returns:
            DataFrame with signals
        """
        # Combine signals from all strategies
        all_signals = []
        
        for strategy in self.strategies:
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Add strategy name
            signals['strategy'] = strategy.__class__.__name__
            
            all_signals.append(signals)
            
        # Combine signals
        if all_signals:
            combined_signals = pd.concat(all_signals)
            return combined_signals
        else:
            return pd.DataFrame()
            
    def _execute_trades(self, signals: pd.DataFrame, data: pd.DataFrame) -> None:
        """
        Execute trades based on signals.
        
        Args:
            signals: DataFrame with signals
            data: Market data
        """
        if signals.empty:
            return
            
        # Group signals by ticker
        for ticker in signals['ticker'].unique():
            ticker_signals = signals[signals['ticker'] == ticker]
            
            # Get latest signal
            latest_signal = ticker_signals.iloc[-1]
            
            # Get current position
            current_position = self.portfolio['positions'].get(ticker, 0)
            
            # Get current price
            current_price = data[data['ticker'] == ticker]['close'].iloc[-1]
            
            # Determine action based on signal and current position
            if latest_signal['signal'] == 1 and current_position <= 0:  # BUY
                # Calculate position size
                position_size = self._calculate_position_size(ticker, current_price)
                
                # Execute buy order
                self._execute_buy_order(ticker, position_size, current_price)
                
            elif latest_signal['signal'] == -1 and current_position >= 0:  # SELL
                # Execute sell order
                self._execute_sell_order(ticker, abs(current_position), current_price)
                
    def _calculate_position_size(self, ticker: str, price: float) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            ticker: Symbol to trade
            price: Current price
            
        Returns:
            Number of shares to trade
        """
        # Get risk parameters
        max_position_size = self.risk_manager['max_position_size']
        max_positions = self.risk_manager['max_positions']
        
        # Check if we've reached max positions
        if len(self.portfolio['positions']) >= max_positions:
            return 0
            
        # Calculate position size based on available cash and risk limits
        available_cash = self.portfolio['cash']
        position_value = min(available_cash * 0.1, max_position_size)  # Use 10% of cash per position
        
        # Calculate number of shares
        shares = int(position_value / price)
        
        return shares
        
    def _execute_buy_order(self, ticker: str, shares: int, price: float) -> None:
        """
        Execute a buy order.
        
        Args:
            ticker: Symbol to buy
            shares: Number of shares to buy
            price: Price to buy at
        """
        if shares <= 0:
            return
            
        # Calculate order value
        order_value = shares * price
        
        # Check if we have enough cash
        if order_value > self.portfolio['cash']:
            self.logger.warning(f"Not enough cash to buy {shares} shares of {ticker}")
            return
            
        # Calculate commission
        commission = order_value * self.config["portfolio"]["commission"]
        
        # Update portfolio
        self.portfolio['cash'] -= (order_value + commission)
        self.portfolio['positions'][ticker] = self.portfolio['positions'].get(ticker, 0) + shares
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'value': order_value,
            'commission': commission
        }
        self.portfolio['trades'].append(trade)
        
        # Log trade
        self.logger.info(f"BUY: {shares} shares of {ticker} at ${price:.2f}")
        
        # Execute order through provider
        if not self.paper_trading:
            # This would call the actual broker API
            pass
            
    def _execute_sell_order(self, ticker: str, shares: int, price: float) -> None:
        """
        Execute a sell order.
        
        Args:
            ticker: Symbol to sell
            shares: Number of shares to sell
            price: Price to sell at
        """
        if shares <= 0:
            return
            
        # Calculate order value
        order_value = shares * price
        
        # Calculate commission
        commission = order_value * self.config["portfolio"]["commission"]
        
        # Update portfolio
        self.portfolio['cash'] += (order_value - commission)
        self.portfolio['positions'][ticker] = self.portfolio['positions'].get(ticker, 0) - shares
        
        # Remove position if zero
        if self.portfolio['positions'][ticker] == 0:
            del self.portfolio['positions'][ticker]
            
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'value': order_value,
            'commission': commission
        }
        self.portfolio['trades'].append(trade)
        
        # Log trade
        self.logger.info(f"SELL: {shares} shares of {ticker} at ${price:.2f}")
        
        # Execute order through provider
        if not self.paper_trading:
            # This would call the actual broker API
            pass
            
    def _update_portfolio(self, data: pd.DataFrame) -> None:
        """
        Update portfolio values based on latest market data.
        
        Args:
            data: Latest market data
        """
        # Calculate portfolio value
        portfolio_value = self.portfolio['cash']
        
        for ticker, position in self.portfolio['positions'].items():
            # Get latest price
            ticker_data = data[data['ticker'] == ticker]
            if not ticker_data.empty:
                latest_price = ticker_data['close'].iloc[-1]
                
                # Add position value
                position_value = position * latest_price
                portfolio_value += position_value
                
        # Log portfolio value
        self.logger.info(f"Portfolio value: ${portfolio_value:.2f}")
        
    def _manage_risk(self, data: pd.DataFrame) -> None:
        """
        Manage risk based on portfolio and market data.
        
        Args:
            data: Latest market data
        """
        # Check stop loss and take profit
        for ticker, position in list(self.portfolio['positions'].items()):
            # Get latest price
            ticker_data = data[data['ticker'] == ticker]
            if ticker_data.empty:
                continue
                
            latest_price = ticker_data['close'].iloc[-1]
            
            # Find entry price from trades
            entry_trades = [t for t in self.portfolio['trades'] 
                           if t['ticker'] == ticker and t['action'] == 'BUY']
            
            if not entry_trades:
                continue
                
            # Calculate average entry price
            entry_price = sum(t['price'] * t['shares'] for t in entry_trades) / sum(t['shares'] for t in entry_trades)
            
            # Calculate return
            returns = (latest_price - entry_price) / entry_price
            
            # Check stop loss
            if returns < -self.risk_manager['stop_loss']:
                self.logger.info(f"Stop loss triggered for {ticker}")
                self._execute_sell_order(ticker, position, latest_price)
                
            # Check take profit
            elif returns > self.risk_manager['take_profit']:
                self.logger.info(f"Take profit triggered for {ticker}")
                self._execute_sell_order(ticker, position, latest_price)
                
    def _wait_for_next_update(self) -> None:
        """Wait until the next update based on data interval."""
        # Get update interval
        interval = self.config["data"].get("interval", "1d")
        
        # Convert interval to seconds
        if interval == "1d":
            sleep_time = 24 * 60 * 60  # 24 hours
        elif interval == "1h":
            sleep_time = 60 * 60  # 1 hour
        elif interval == "15m":
            sleep_time = 15 * 60  # 15 minutes
        elif interval == "5m":
            sleep_time = 5 * 60  # 5 minutes
        elif interval == "1m":
            sleep_time = 60  # 1 minute
        else:
            sleep_time = 60  # Default to 1 minute
            
        # Sleep until next update or until stop event
        self.stop_event.wait(sleep_time)
    
    def _load_historical_data(self) -> pd.DataFrame:
        """
        Load historical market data.
        
        Returns:
            DataFrame containing market data
        """
        data_config = self.config.get("data", {})
        
        # Get date range
        start_date = data_config.get("start_date")
        end_date = data_config.get("end_date")
        
        # Get symbols
        symbols = self._get_universe_symbols()
        
        self.logger.info(f"Loading historical data for {len(symbols)} symbols")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        
        # Load data
        data = self.data_provider.fetch_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        return data
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process market data with factors.
        
        Args:
            data: Raw market data
            
        Returns:
            Processed data with factor values
        """
        # First process with data processor
        processed_data = self.data_processor.process(data)
        
        # Then apply each factor
        for factor in self.factors:
            try:
                processed_data = factor.calculate(processed_data)
                self.logger.info(f"Applied factor: {factor.metadata.name}")
            except Exception as e:
                self.logger.warning(f"Failed to apply factor {factor.metadata.name}: {str(e)}")
                continue
                
        return processed_data
    
    def _get_universe_symbols(self) -> List[str]:
        """
        Get list of symbols to trade based on configuration.
        
        Returns:
            List of symbols
        """
        data_config = self.config.get("data", {})
        
        # Check for custom universe
        custom_universe = data_config.get("custom_universe", [])
        if custom_universe:
            return custom_universe
        
        # Check for universe type
        universe_type = data_config.get("universe_type", "sp500")
        
        # Get symbols based on universe type
        if universe_type == "sp500":
            return self.data_provider.get_sp500_symbols()
        elif universe_type == "nasdaq100":
            return self.data_provider.get_nasdaq100_symbols()
        else:
            self.logger.warning(f"Unknown universe type: {universe_type}")
            return []
    
    def _generate_backtest_report(
        self,
        results: Dict[str, Any],
        strategy: Any
    ) -> None:
        """
        Generate backtest report.
        
        Args:
            results: Backtest results
            strategy: Strategy that generated the results
        """
        # Create output directory
        output_dir = self.config["logging"]["output_dir"]
        strategy_dir = f"{output_dir}/{strategy.__class__.__name__}"
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame(results["metrics"], index=[0])
        metrics_df.to_csv(f"{strategy_dir}/metrics.csv")
        
        # Save trade summary
        trades_df = pd.DataFrame(results["trades"])
        trades_df.to_csv(f"{strategy_dir}/trades.csv")
        
        # Save equity curve
        equity_df = pd.DataFrame(results["equity_curve"])
        equity_df.to_csv(f"{strategy_dir}/equity_curve.csv")
        
        self.logger.info(f"Backtest report saved to {strategy_dir}") 