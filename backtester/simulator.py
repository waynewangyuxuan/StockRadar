"""
Backtest simulator for evaluating trading strategies.

This module provides the core simulation engine for backtesting trading strategies.
It handles order execution, position tracking, and portfolio management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    """Types of orders that can be executed in the simulator."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

@dataclass
class Order:
    """Represents a trading order in the simulator."""
    ticker: str
    order_type: OrderType
    side: int  # 1 for buy, -1 for sell
    quantity: int
    price: Optional[float] = None  # For limit and stop orders
    timestamp: datetime = None
    filled: bool = False
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None

@dataclass
class Position:
    """Represents a position in the simulator."""
    ticker: str
    quantity: int
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float

class BacktestSimulator:
    """
    Simulates market conditions and executes trades based on strategy signals.
    
    This class handles the core backtesting functionality, including:
    - Order execution with slippage and transaction costs
    - Position tracking and portfolio management
    - Basic market constraints
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0001
    ):
        """
        Initialize the backtest simulator.
        
        Args:
            initial_capital: Starting capital for the backtest
            transaction_cost: Transaction cost as a fraction of trade value
            slippage: Slippage as a fraction of trade value
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Tracking containers
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_value: List[float] = []
        self.timestamps: List[datetime] = []
        
    def run(
        self,
        strategy: Any,
        market_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Run a backtest simulation.
        
        Args:
            strategy: Strategy object that generates signals
            market_data: DataFrame with market data (multi-index: date, ticker)
            start_date: Start date for the backtest
            end_date: End date for the backtest
            
        Returns:
            Dictionary containing backtest results
        """
        # Reset simulator state
        self._reset()
        
        # Filter data for date range
        mask = (market_data.index.get_level_values('date') >= start_date) & \
               (market_data.index.get_level_values('date') <= end_date)
        market_data = market_data[mask].copy()
        
        # Sort by date to ensure chronological processing
        market_data = market_data.sort_index()
        
        # Process each timestamp
        for date in market_data.index.get_level_values('date').unique():
            # Get data for current timestamp
            current_data = market_data.xs(date, level='date')
            
            # Generate signals from strategy
            signals = strategy.generate_signals(current_data)
            
            # Process signals and execute trades
            self._process_signals(signals, current_data)
            
            # Update positions and portfolio value
            self._update_positions(current_data)
            self._record_portfolio_state(date)
        
        # Calculate final results
        return self._calculate_results()
    
    def _reset(self) -> None:
        """Reset the simulator state."""
        self.current_capital = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.portfolio_value.clear()
        self.timestamps.clear()
    
    def _process_signals(self, signals: pd.DataFrame, market_data: pd.DataFrame) -> None:
        """
        Process trading signals and execute trades.
        
        Args:
            signals: DataFrame with strategy signals
            market_data: Current market data
        """
        for ticker in signals.index:
            signal = signals.loc[ticker, 'signal']
            signal_strength = signals.loc[ticker, 'signal_strength']
            
            if signal == 0:  # HOLD
                continue
                
            current_price = market_data.loc[ticker, 'close']
            position = self.positions.get(ticker)
            
            # Determine trade size based on signal strength and capital
            max_position_value = self.current_capital * 0.1  # Max 10% per position
            position_value = max_position_value * signal_strength
            quantity = int(position_value / current_price)
            
            if quantity == 0:
                continue
                
            # Create and execute order
            order = Order(
                ticker=ticker,
                order_type=OrderType.MARKET,
                side=signal,  # 1 for buy, -1 for sell
                quantity=quantity,
                timestamp=market_data.index[0][0]  # Current timestamp
            )
            
            self._execute_order(order, current_price)
    
    def _execute_order(self, order: Order, current_price: float) -> None:
        """
        Execute a trading order.
        
        Args:
            order: Order to execute
            current_price: Current market price
        """
        # Apply slippage
        execution_price = current_price * (1 + self.slippage * order.side)
        
        # Calculate transaction cost
        trade_value = execution_price * order.quantity
        transaction_cost = trade_value * self.transaction_cost
        
        # Update capital
        self.current_capital -= (trade_value + transaction_cost) * order.side
        
        # Update position
        if order.ticker not in self.positions:
            self.positions[order.ticker] = Position(
                ticker=order.ticker,
                quantity=order.quantity * order.side,
                average_price=execution_price,
                current_price=execution_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
        else:
            position = self.positions[order.ticker]
            # Update average price
            total_quantity = position.quantity + (order.quantity * order.side)
            if total_quantity != 0:
                position.average_price = (
                    (position.quantity * position.average_price) +
                    (order.quantity * order.side * execution_price)
                ) / total_quantity
            position.quantity = total_quantity
        
        # Record trade
        self.trades.append({
            'timestamp': order.timestamp,
            'ticker': order.ticker,
            'side': order.side,
            'quantity': order.quantity,
            'price': execution_price,
            'value': trade_value,
            'cost': transaction_cost
        })
        
        # Mark order as filled
        order.filled = True
        order.fill_price = execution_price
        order.fill_time = order.timestamp
        self.orders.append(order)
    
    def _update_positions(self, market_data: pd.DataFrame) -> None:
        """
        Update position values and P&L.
        
        Args:
            market_data: Current market data
        """
        for ticker, position in self.positions.items():
            if ticker in market_data.index:
                current_price = market_data.loc[ticker, 'close']
                position.current_price = current_price
                position.unrealized_pnl = position.quantity * (current_price - position.average_price)
    
    def _record_portfolio_state(self, timestamp: datetime) -> None:
        """
        Record the current portfolio state.
        
        Args:
            timestamp: Current timestamp
        """
        # Calculate total portfolio value
        position_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        total_value = self.current_capital + position_value
        
        self.portfolio_value.append(total_value)
        self.timestamps.append(timestamp)
    
    def _calculate_results(self) -> Dict[str, Any]:
        """
        Calculate final backtest results.
        
        Returns:
            Dictionary containing backtest results
        """
        # Convert to numpy arrays for calculations
        portfolio_values = np.array(self.portfolio_value)
        timestamps = np.array(self.timestamps)
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / self.initial_capital) - 1
        annual_return = total_return * (252 / len(returns))  # Assuming daily data
        
        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': portfolio_values[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': drawdown.min(),
            'num_trades': len(self.trades),
            'portfolio_values': portfolio_values,
            'timestamps': timestamps,
            'trades': self.trades,
            'positions': self.positions
        }
