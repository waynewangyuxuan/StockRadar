from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from core.strategy_base import StrategyBase, SignalType

class MomentumBreakoutStrategy(StrategyBase):
    """Momentum Breakout Strategy implementation.
    
    This strategy generates signals based on price breakouts and momentum:
    - BUY when price breaks above resistance with high momentum
    - SELL when price breaks below support with high momentum
    - HOLD otherwise
    
    Required factors:
    - resistance: Price resistance level
    - support: Price support level
    - momentum: Price momentum indicator
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Momentum Breakout strategy.
        
        Args:
            config: Dictionary containing:
                - momentum_threshold: Minimum momentum value for signal (default: 0.02)
                - breakout_threshold: Minimum breakout distance (default: 0.01)
                - confirmation_periods: Number of periods to confirm breakout (default: 3)
        """
        super().__init__(config)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.02)
        self.breakout_threshold = self.config.get('breakout_threshold', 0.01)
        self.confirmation_periods = self.config.get('confirmation_periods', 3)
        
        # Validate configuration
        if self.momentum_threshold <= 0:
            raise ValueError("Momentum threshold must be positive")
        if self.breakout_threshold <= 0:
            raise ValueError("Breakout threshold must be positive")
        if self.confirmation_periods < 1:
            raise ValueError("Confirmation periods must be at least 1")
    
    def get_required_factors(self) -> List[str]:
        """Get list of required factor names."""
        return ['resistance', 'support', 'momentum']
    
    def generate_signals(self, 
                        market_data: pd.DataFrame,
                        factor_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate trading signals based on breakouts and momentum.
        
        Args:
            market_data: DataFrame with market data
            factor_data: DataFrame with pre-computed factor values
            
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
        resistance = factor_data['resistance']
        support = factor_data['support']
        momentum = factor_data['momentum']
        
        # Ensure indices match
        price = price.reindex(factor_data.index)
        resistance = resistance.reindex(factor_data.index)
        support = support.reindex(factor_data.index)
        momentum = momentum.reindex(factor_data.index)
        
        # Calculate breakout distances
        resistance_breakout = (price - resistance) / resistance
        support_breakout = (price - support) / support
        
        # Calculate confirmation masks
        resistance_confirmed = self._check_confirmation(resistance_breakout > 0)
        support_confirmed = self._check_confirmation(support_breakout < 0)
        
        # Generate signals based on breakouts and momentum
        # Buy signals: breakout above resistance with high momentum
        buy_mask = (resistance_breakout > self.breakout_threshold) & \
                  (resistance_confirmed) & \
                  (momentum > self.momentum_threshold)
        
        # Sell signals: breakout below support with high momentum
        sell_mask = (support_breakout < -self.breakout_threshold) & \
                   (support_confirmed) & \
                   (momentum < -self.momentum_threshold)
        
        # Calculate signal strength based on breakout and momentum
        # Combine breakout distance and momentum for strength
        breakout_strength = pd.concat([
            resistance_breakout.clip(0, 1),
            -support_breakout.clip(-1, 0)
        ], axis=1).max(axis=1)
        
        momentum_strength = momentum.abs().clip(0, 1)
        
        # Combine strengths with equal weights
        signal_strength = (breakout_strength + momentum_strength) / 2
        
        # Apply signals using numpy arrays for masks
        result.loc[buy_mask.values, 'signal'] = SignalType.BUY
        result.loc[sell_mask.values, 'signal'] = SignalType.SELL
        result['signal_strength'] = signal_strength
        
        return result
    
    def _check_confirmation(self, condition: pd.Series) -> pd.Series:
        """Check if a condition is confirmed over multiple periods.
        
        Args:
            condition: Boolean Series to check for confirmation
            
        Returns:
            Boolean Series indicating confirmed conditions
        """
        # Use rolling sum to count consecutive True values
        confirmed = condition.rolling(
            window=self.confirmation_periods,
            min_periods=1
        ).sum() >= self.confirmation_periods
        
        # Handle NaN values
        confirmed = confirmed.fillna(False)
        
        return confirmed
    
    def calculate_signal_strength(self, 
                                breakout: pd.Series,
                                momentum: pd.Series) -> pd.Series:
        """Calculate normalized signal strength based on breakout and momentum.
        
        Args:
            breakout: Series of breakout distances
            momentum: Series of momentum values
            
        Returns:
            Series of normalized signal strengths in [0, 1] range
        """
        # Normalize breakout distance
        max_breakout = breakout.abs().max()
        if max_breakout > 0:
            breakout_strength = (breakout.abs() / max_breakout).clip(0, 1)
        else:
            breakout_strength = pd.Series(0, index=breakout.index)
        
        # Normalize momentum
        max_momentum = momentum.abs().max()
        if max_momentum > 0:
            momentum_strength = (momentum.abs() / max_momentum).clip(0, 1)
        else:
            momentum_strength = pd.Series(0, index=momentum.index)
        
        # Combine strengths with equal weights
        return (breakout_strength + momentum_strength) / 2
