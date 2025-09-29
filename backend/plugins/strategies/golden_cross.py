from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from core.strategy_base import StrategyBase, SignalType

class GoldenCrossStrategy(StrategyBase):
    """Golden Cross Strategy implementation.
    
    This strategy generates signals based on moving average crossovers:
    - BUY when short-term MA crosses above long-term MA
    - SELL when short-term MA crosses below long-term MA
    - HOLD otherwise
    
    Required factors:
    - ma_short: Short-term moving average
    - ma_long: Long-term moving average
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Golden Cross strategy.
        
        Args:
            config: Dictionary containing:
                - short_window: Window size for short-term MA (default: 20)
                - long_window: Window size for long-term MA (default: 50)
                - min_cross_threshold: Minimum difference between MAs to trigger signal (default: 0.0)
        """
        super().__init__(config)
        self.short_window = self.config.get('short_window', 20)
        self.long_window = self.config.get('long_window', 50)
        self.min_cross_threshold = self.config.get('min_cross_threshold', 0.0)
        
        # Validate configuration
        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window")
        if self.min_cross_threshold < 0:
            raise ValueError("Minimum cross threshold must be non-negative")
    
    def get_required_factors(self) -> List[str]:
        """Get list of required factor names."""
        return ['ma_short', 'ma_long']
    
    def generate_signals(self, 
                        market_data: pd.DataFrame,
                        factor_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate trading signals based on MA crossovers.
        
        Args:
            market_data: DataFrame with market data
            factor_data: DataFrame with pre-computed MA values
            
        Returns:
            DataFrame with signal columns added
        """
        # Validate inputs
        self.validate_input(market_data, factor_data)
        
        # Create result DataFrame
        result = market_data.copy()
        
        # Initialize signal columns
        result['signal'] = SignalType.HOLD
        result['signal_strength'] = 0.0
        result['strategy_name'] = self.name
        result['timestamp'] = datetime.now()
        
        # Get MA values from factor data
        ma_short = factor_data['ma_short']
        ma_long = factor_data['ma_long']
        
        # Calculate MA difference
        ma_diff = ma_short - ma_long
        
        # Generate signals based on crossovers
        # Previous difference to detect crossovers
        prev_diff = ma_diff.shift(1)
        
        # Buy signals: current diff > threshold and previous diff <= threshold
        buy_mask = (ma_diff > self.min_cross_threshold) & (prev_diff <= self.min_cross_threshold)
        
        # Sell signals: current diff < -threshold and previous diff >= -threshold
        sell_mask = (ma_diff < -self.min_cross_threshold) & (prev_diff >= -self.min_cross_threshold)
        
        # Calculate signal strength based on MA difference
        # Normalize to [0, 1] range
        max_diff = ma_diff.abs().max()
        if max_diff > 0:
            signal_strength = (ma_diff.abs() / max_diff).clip(0, 1)
        else:
            signal_strength = pd.Series(0, index=ma_diff.index)
        
        # Apply signals
        result.loc[buy_mask, 'signal'] = SignalType.BUY
        result.loc[sell_mask, 'signal'] = SignalType.SELL
        result['signal_strength'] = signal_strength
        
        return result
    
    def calculate_signal_strength(self, ma_diff: pd.Series) -> pd.Series:
        """Calculate normalized signal strength based on MA difference.
        
        Args:
            ma_diff: Series of differences between short and long MAs
            
        Returns:
            Series of normalized signal strengths in [0, 1] range
        """
        max_diff = ma_diff.abs().max()
        if max_diff > 0:
            return (ma_diff.abs() / max_diff).clip(0, 1)
        return pd.Series(0, index=ma_diff.index)
