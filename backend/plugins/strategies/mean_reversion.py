from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from core.strategy_base import StrategyBase, SignalType

class MeanReversionStrategy(StrategyBase):
    """Mean Reversion Strategy implementation.
    
    This strategy generates signals based on price deviations from moving averages:
    - BUY when price is significantly below moving average
    - SELL when price is significantly above moving average
    - HOLD otherwise
    
    Required factors:
    - ma: Moving average
    - std: Standard deviation of price
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Mean Reversion strategy.
        
        Args:
            config: Dictionary containing:
                - window: Window size for MA calculation (default: 20)
                - std_threshold: Number of standard deviations for signal threshold (default: 2.0)
                - min_deviation: Minimum price deviation to trigger signal (default: 0.01)
        """
        super().__init__(config)
        self.window = self.config.get('window', 20)
        self.std_threshold = self.config.get('std_threshold', 2.0)
        self.min_deviation = self.config.get('min_deviation', 0.01)
        
        # Validate configuration
        if self.window <= 0:
            raise ValueError("Window size must be positive")
        if self.std_threshold <= 0:
            raise ValueError("Standard deviation threshold must be positive")
        if self.min_deviation <= 0:
            raise ValueError("Minimum deviation must be positive")
    
    def get_required_factors(self) -> List[str]:
        """Get list of required factor names."""
        return ['ma', 'std']
    
    def generate_signals(self, 
                        market_data: pd.DataFrame,
                        factor_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate trading signals based on price deviations.
        
        Args:
            market_data: DataFrame with market data
            factor_data: DataFrame with pre-computed MA and std values
            
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
        
        # Get price and factor data
        price = market_data['close']
        ma = factor_data['ma']
        std = factor_data['std']
        
        # Calculate price deviation from MA
        deviation = (price - ma) / ma
        
        # Calculate z-score
        z_score = (price - ma) / std
        
        # Generate signals based on deviations
        # Buy signals: price significantly below MA
        buy_mask = (deviation < -self.min_deviation) & (z_score < -self.std_threshold)
        
        # Sell signals: price significantly above MA
        sell_mask = (deviation > self.min_deviation) & (z_score > self.std_threshold)
        
        # Calculate signal strength based on deviation
        # Normalize to [0, 1] range using z-score
        signal_strength = (z_score.abs() / self.std_threshold).clip(0, 1)
        
        # Apply signals using numpy arrays for masks
        result.loc[buy_mask.values, 'signal'] = SignalType.BUY
        result.loc[sell_mask.values, 'signal'] = SignalType.SELL
        result['signal_strength'] = signal_strength
        
        return result
    
    def calculate_signal_strength(self, z_score: pd.Series) -> pd.Series:
        """Calculate normalized signal strength based on z-score.
        
        Args:
            z_score: Series of z-scores
            
        Returns:
            Series of normalized signal strengths in [0, 1] range
        """
        return (z_score.abs() / self.std_threshold).clip(0, 1)
