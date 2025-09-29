import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.strategy_base import SignalType
from plugins.strategies.golden_cross import GoldenCrossStrategy
from plugins.strategies.mean_reversion import MeanReversionStrategy
from plugins.strategies.momentum_breakout import MomentumBreakoutStrategy

@pytest.fixture
def market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = {
        'ticker': ['AAPL'] * 100,
        'date': dates,
        'open': np.linspace(100, 200, 100),
        'high': np.linspace(105, 210, 100),
        'low': np.linspace(95, 190, 100),
        'close': np.linspace(100, 200, 100),
        'volume': np.random.randint(1000000, 2000000, 100)
    }
    df = pd.DataFrame(data)
    df.set_index(['date', 'ticker'], inplace=True)
    return df

@pytest.fixture
def factor_data():
    """Create sample factor data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    tickers = ['AAPL'] * 100
    
    # Create multi-index
    index = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])
    
    # Generate factor data
    data = {
        'ma_short': np.linspace(100, 200, 100),
        'ma_long': np.linspace(90, 180, 100),  # Ensure crossover points
        'ma': np.linspace(95, 190, 100),
        'std': np.ones(100) * 2.0,  # Constant std for predictable signals
        'resistance': np.linspace(110, 220, 100),
        'support': np.linspace(90, 170, 100),
        'momentum': np.sin(np.linspace(0, 4*np.pi, 100)) * 0.05  # Oscillating momentum
    }
    return pd.DataFrame(data, index=index)

class TestGoldenCrossStrategy:
    """Test suite for Golden Cross Strategy."""
    
    def test_initialization(self):
        """Test strategy initialization with valid and invalid configs."""
        # Test default initialization
        strategy = GoldenCrossStrategy()
        assert strategy.short_window == 20
        assert strategy.long_window == 50
        assert strategy.min_cross_threshold == 0.0
        
        # Test custom initialization
        config = {
            'short_window': 10,
            'long_window': 30,
            'min_cross_threshold': 0.01
        }
        strategy = GoldenCrossStrategy(config)
        assert strategy.short_window == 10
        assert strategy.long_window == 30
        assert strategy.min_cross_threshold == 0.01
        
        # Test invalid initialization
        with pytest.raises(ValueError):
            GoldenCrossStrategy({'short_window': 50, 'long_window': 20})
        with pytest.raises(ValueError):
            GoldenCrossStrategy({'min_cross_threshold': -0.01})
    
    def test_required_factors(self):
        """Test required factors list."""
        strategy = GoldenCrossStrategy()
        required = strategy.get_required_factors()
        assert required == ['ma_short', 'ma_long']
    
    def test_signal_generation(self, market_data, factor_data):
        """Test signal generation logic."""
        strategy = GoldenCrossStrategy()
        result = strategy.generate_signals(market_data, factor_data)
        
        # Check required columns
        assert 'signal' in result.columns
        assert 'signal_strength' in result.columns
        assert 'strategy_name' in result.columns
        assert 'timestamp' in result.columns
        
        # Check signal types
        assert result['signal'].dtype == SignalType
        assert all(s in [SignalType.BUY, SignalType.SELL, SignalType.HOLD] 
                  for s in result['signal'].unique())
        
        # Check signal strength range
        assert result['signal_strength'].min() >= 0
        assert result['signal_strength'].max() <= 1
        assert not result['signal_strength'].isna().any()
        
        # Check strategy name
        assert all(result['strategy_name'] == 'GoldenCrossStrategy')
        
        # Check timestamp
        assert all(isinstance(t, datetime) for t in result['timestamp'])

class TestMeanReversionStrategy:
    """Test suite for Mean Reversion Strategy."""
    
    def test_initialization(self):
        """Test strategy initialization with valid and invalid configs."""
        # Test default initialization
        strategy = MeanReversionStrategy()
        assert strategy.window == 20
        assert strategy.std_threshold == 2.0
        assert strategy.min_deviation == 0.01
        
        # Test custom initialization
        config = {
            'window': 30,
            'std_threshold': 3.0,
            'min_deviation': 0.02
        }
        strategy = MeanReversionStrategy(config)
        assert strategy.window == 30
        assert strategy.std_threshold == 3.0
        assert strategy.min_deviation == 0.02
        
        # Test invalid initialization
        with pytest.raises(ValueError):
            MeanReversionStrategy({'window': 0})
        with pytest.raises(ValueError):
            MeanReversionStrategy({'std_threshold': -1})
        with pytest.raises(ValueError):
            MeanReversionStrategy({'min_deviation': -0.01})
    
    def test_required_factors(self):
        """Test required factors list."""
        strategy = MeanReversionStrategy()
        required = strategy.get_required_factors()
        assert required == ['ma', 'std']
    
    def test_signal_generation(self, market_data, factor_data):
        """Test signal generation logic."""
        strategy = MeanReversionStrategy()
        result = strategy.generate_signals(market_data, factor_data)
        
        # Check required columns
        assert 'signal' in result.columns
        assert 'signal_strength' in result.columns
        assert 'strategy_name' in result.columns
        assert 'timestamp' in result.columns
        
        # Check signal types
        assert result['signal'].dtype == SignalType
        assert all(s in [SignalType.BUY, SignalType.SELL, SignalType.HOLD] 
                  for s in result['signal'].unique())
        
        # Check signal strength range
        assert result['signal_strength'].min() >= 0
        assert result['signal_strength'].max() <= 1
        assert not result['signal_strength'].isna().any()
        
        # Check strategy name
        assert all(result['strategy_name'] == 'MeanReversionStrategy')
        
        # Check timestamp
        assert all(isinstance(t, datetime) for t in result['timestamp'])

class TestMomentumBreakoutStrategy:
    """Test suite for Momentum Breakout Strategy."""
    
    def test_initialization(self):
        """Test strategy initialization with valid and invalid configs."""
        # Test default initialization
        strategy = MomentumBreakoutStrategy()
        assert strategy.momentum_threshold == 0.02
        assert strategy.breakout_threshold == 0.01
        assert strategy.confirmation_periods == 3
        
        # Test custom initialization
        config = {
            'momentum_threshold': 0.03,
            'breakout_threshold': 0.02,
            'confirmation_periods': 5
        }
        strategy = MomentumBreakoutStrategy(config)
        assert strategy.momentum_threshold == 0.03
        assert strategy.breakout_threshold == 0.02
        assert strategy.confirmation_periods == 5
        
        # Test invalid initialization
        with pytest.raises(ValueError):
            MomentumBreakoutStrategy({'momentum_threshold': -0.01})
        with pytest.raises(ValueError):
            MomentumBreakoutStrategy({'breakout_threshold': -0.01})
        with pytest.raises(ValueError):
            MomentumBreakoutStrategy({'confirmation_periods': 0})
    
    def test_required_factors(self):
        """Test required factors list."""
        strategy = MomentumBreakoutStrategy()
        required = strategy.get_required_factors()
        assert required == ['resistance', 'support', 'momentum']
    
    def test_signal_generation(self, market_data, factor_data):
        """Test signal generation logic."""
        strategy = MomentumBreakoutStrategy()
        result = strategy.generate_signals(market_data, factor_data)
        
        # Check required columns
        assert 'signal' in result.columns
        assert 'signal_strength' in result.columns
        assert 'strategy_name' in result.columns
        assert 'timestamp' in result.columns
        
        # Check signal types
        assert result['signal'].dtype == SignalType
        assert all(s in [SignalType.BUY, SignalType.SELL, SignalType.HOLD] 
                  for s in result['signal'].unique())
        
        # Check signal strength range
        assert result['signal_strength'].min() >= 0
        assert result['signal_strength'].max() <= 1
        assert not result['signal_strength'].isna().any()
        
        # Check strategy name
        assert all(result['strategy_name'] == 'MomentumBreakoutStrategy')
        
        # Check timestamp
        assert all(isinstance(t, datetime) for t in result['timestamp'])
    
    def test_confirmation_check(self):
        """Test the confirmation check logic."""
        strategy = MomentumBreakoutStrategy({'confirmation_periods': 3})
        
        # Test with all True values
        condition = pd.Series([True] * 5)
        confirmed = strategy._check_confirmation(condition)
        assert all(confirmed[2:])  # First two periods can't be confirmed
        assert not any(confirmed[:2])  # First two periods should be False
        
        # Test with alternating values
        condition = pd.Series([True, False, True, False, True])
        confirmed = strategy._check_confirmation(condition)
        assert not any(confirmed)  # No 3 consecutive True values
        
        # Test with partial True values
        condition = pd.Series([True, True, True, False, True])
        confirmed = strategy._check_confirmation(condition)
        assert confirmed[2]  # Only position 2 should be True
        assert not any(confirmed[[0,1,3,4]])  # Others should be False
        
        # Test with NaN values
        condition = pd.Series([True, True, np.nan, True, True])
        confirmed = strategy._check_confirmation(condition)
        assert not any(confirmed)  # NaN breaks confirmation 