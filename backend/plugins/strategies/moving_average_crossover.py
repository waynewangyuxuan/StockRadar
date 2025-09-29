"""
Moving Average Crossover Strategy.

This strategy generates buy signals when the short-term moving average
crosses above the long-term moving average, and sell signals when it
crosses below.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.strategy_base import StrategyBase, SignalType

class MovingAverageCrossoverStrategy(StrategyBase):
    """
    Moving Average Crossover Strategy.
    
    This strategy generates:
    - Buy signals when the short-term moving average crosses above the long-term moving average
    - Sell signals when the short-term moving average crosses below the long-term moving average
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy.
        
        Args:
            config: Strategy configuration with the following keys:
                - short_window: Short-term moving average window (default: 20)
                - long_window: Long-term moving average window (default: 50)
        """
        super().__init__(config)
        
        # Set default parameters
        self.short_window = self.config.get("short_window", 20)
        self.long_window = self.config.get("long_window", 50)
        
        # Validate parameters
        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window")
    
    def generate_signals(self, market_data: pd.DataFrame, factor_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            market_data: DataFrame with OHLCV data
            factor_data: Optional DataFrame with pre-computed factors
            
        Returns:
            DataFrame with signal column
        """
        # Validate input
        self.validate_input(market_data, factor_data)
        
        # Create a copy of the data to avoid modifying the original
        data = market_data.copy()
        
        # Calculate moving averages
        data['short_ma'] = data.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window=self.short_window).mean()
        )
        data['long_ma'] = data.groupby('ticker')['close'].transform(
            lambda x: x.rolling(window=self.long_window).mean()
        )
        
        # Initialize signal column
        data['signal'] = SignalType.HOLD.value
        
        # Generate signals
        # Buy when short MA crosses above long MA
        data.loc[data['short_ma'] > data['long_ma'], 'signal'] = SignalType.BUY.value
        
        # Sell when short MA crosses below long MA
        data.loc[data['short_ma'] < data['long_ma'], 'signal'] = SignalType.SELL.value
        
        # Calculate signal strength (distance between MAs as percentage of price)
        data['signal_strength'] = abs(data['short_ma'] - data['long_ma']) / data['close']
        
        # Add strategy name and timestamp
        data['strategy_name'] = self.name
        data['timestamp'] = datetime.now()
        
        # Select only the signal columns
        signal_columns = self.get_signal_columns()
        signals = data[signal_columns].copy()
        
        return signals
    
    def get_required_factors(self) -> List[str]:
        """
        Get list of factor names required by this strategy.
        
        Returns:
            Empty list as this strategy doesn't require any pre-computed factors
        """
        return []
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name}(short_window={self.short_window}, long_window={self.long_window})" 