from typing import Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy

class MACrossStrategy(BaseStrategy):
    """Moving Average Crossover Strategy."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        """
        Initialize the MA Crossover strategy.
        
        Args:
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
        """
        parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period
        }
        super().__init__("MA_Crossover", parameters)
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        return (
            self.parameters["fast_period"] > 0 and
            self.parameters["slow_period"] > self.parameters["fast_period"]
        )
    
    def get_required_indicators(self) -> list:
        """Get required indicators for the strategy."""
        return [
            f"MA_{self.parameters['fast_period']}",
            f"MA_{self.parameters['slow_period']}"
        ]
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals based on MA crossover.
        
        Args:
            data: DataFrame containing price and MA indicators
            
        Returns:
            DataFrame with calculated signals
        """
        fast_ma = f"MA_{self.parameters['fast_period']}"
        slow_ma = f"MA_{self.parameters['slow_period']}"
        
        if fast_ma not in data.columns or slow_ma not in data.columns:
            raise ValueError("Required MA indicators not found in data")
        
        # Calculate crossover signals
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        
        # Calculate the difference between fast and slow MA
        ma_diff = data[fast_ma] - data[slow_ma]
        
        # Bullish crossover (fast MA crosses above slow MA)
        bullish_cross = (ma_diff > 0) & (ma_diff.shift(1) <= 0)
        signals.loc[bullish_cross, "signal"] = 1
        
        # Bearish crossover (fast MA crosses below slow MA)
        bearish_cross = (ma_diff < 0) & (ma_diff.shift(1) >= 0)
        signals.loc[bearish_cross, "signal"] = -1
        
        return signals
    
    def generate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate position recommendations based on signals.
        
        Args:
            signals: DataFrame containing trading signals
            
        Returns:
            DataFrame with position recommendations
        """
        positions = pd.DataFrame(index=signals.index)
        positions["position"] = 0
        
        # Convert signals to positions
        positions.loc[signals["signal"] == 1, "position"] = 1  # Long position
        positions.loc[signals["signal"] == -1, "position"] = -1  # Short position
        
        return positions 