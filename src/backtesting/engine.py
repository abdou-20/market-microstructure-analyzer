"""
Event-Driven Backtesting Engine

This module implements a realistic backtesting engine for market microstructure models
with proper handling of transaction costs, market impact, and slippage.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class MarketData:
    """Market data snapshot for backtesting."""
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    features: Optional[np.ndarray] = None
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        return (self.spread / self.mid_price) * 10000


@dataclass
class Order:
    """Order representation for backtesting."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    status: str = "pending"
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return abs(self.filled_quantity - self.quantity) < 1e-8
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity


@dataclass
class Trade:
    """Executed trade representation."""
    trade_id: str
    timestamp: datetime
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    market_impact: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Portfolio position representation."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Get current market value."""
        return self.quantity * self.avg_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return abs(self.quantity) < 1e-8


@dataclass
class Portfolio:
    """Portfolio state representation."""
    cash: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_market_impact: float = 0.0
    
    def get_position(self, symbol: str) -> Position:
        """Get position for symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_position(self, trade: Trade):
        """Update position based on executed trade."""
        position = self.get_position(trade.symbol)
        
        if trade.side == OrderSide.BUY:
            if position.quantity >= 0:
                # Adding to long position or going long from flat
                total_quantity = position.quantity + trade.quantity
                if total_quantity > 0:
                    position.avg_price = (
                        (position.quantity * position.avg_price + trade.quantity * trade.price) / 
                        total_quantity
                    )
                position.quantity = total_quantity
            else:
                # Covering short position
                if trade.quantity >= abs(position.quantity):
                    # Closing short and potentially going long
                    pnl = abs(position.quantity) * (position.avg_price - trade.price)
                    position.realized_pnl += pnl
                    
                    remaining = trade.quantity - abs(position.quantity)
                    if remaining > 0:
                        position.quantity = remaining
                        position.avg_price = trade.price
                    else:
                        position.quantity = 0
                        position.avg_price = 0
                else:
                    # Partial cover
                    position.quantity += trade.quantity
        
        else:  # SELL
            if position.quantity <= 0:
                # Adding to short position or going short from flat
                total_quantity = abs(position.quantity) + trade.quantity
                if position.quantity < 0:
                    position.avg_price = (
                        (abs(position.quantity) * position.avg_price + trade.quantity * trade.price) / 
                        total_quantity
                    )
                else:
                    position.avg_price = trade.price
                position.quantity = -total_quantity
            else:
                # Selling long position
                if trade.quantity >= position.quantity:
                    # Closing long and potentially going short
                    pnl = position.quantity * (trade.price - position.avg_price)
                    position.realized_pnl += pnl
                    
                    remaining = trade.quantity - position.quantity
                    if remaining > 0:
                        position.quantity = -remaining
                        position.avg_price = trade.price
                    else:
                        position.quantity = 0
                        position.avg_price = 0
                else:
                    # Partial sale
                    position.quantity -= trade.quantity
        
        # Update cash
        if trade.side == OrderSide.BUY:
            self.cash -= trade.quantity * trade.price + trade.commission
        else:
            self.cash += trade.quantity * trade.price - trade.commission
        
        # Update totals
        self.total_commission += trade.commission
        self.total_slippage += trade.slippage
        self.total_market_impact += trade.market_impact
    
    def calculate_unrealized_pnl(self, market_prices: Dict[str, float]):
        """Calculate unrealized PnL for all positions."""
        total_unrealized = 0.0
        
        for symbol, position in self.positions.items():
            if not position.is_flat and symbol in market_prices:
                current_price = market_prices[symbol]
                if position.quantity > 0:
                    position.unrealized_pnl = position.quantity * (current_price - position.avg_price)
                else:
                    position.unrealized_pnl = abs(position.quantity) * (position.avg_price - current_price)
                total_unrealized += position.unrealized_pnl
        
        return total_unrealized
    
    @property
    def total_realized_pnl(self) -> float:
        """Get total realized PnL."""
        return sum(pos.realized_pnl for pos in self.positions.values())
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Get total unrealized PnL."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Get total PnL."""
        return self.total_realized_pnl + self.total_unrealized_pnl
    
    @property
    def net_liquidation_value(self) -> float:
        """Get net liquidation value."""
        return self.cash + self.total_unrealized_pnl


class TransactionCostModel:
    """Model for transaction costs including commission and slippage."""
    
    def __init__(self,
                 commission_rate: float = 0.001,  # 10 bps
                 min_commission: float = 1.0,
                 slippage_model: str = "linear",
                 slippage_factor: float = 0.0001):
        """
        Initialize transaction cost model.
        
        Args:
            commission_rate: Commission rate as fraction of trade value
            min_commission: Minimum commission per trade
            slippage_model: Type of slippage model ('linear', 'sqrt', 'fixed')
            slippage_factor: Slippage factor
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_model = slippage_model
        self.slippage_factor = slippage_factor
    
    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade."""
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)
    
    def calculate_slippage(self, 
                          order: Order, 
                          market_data: MarketData, 
                          volume_participation: float = 0.1) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            order: Order to execute
            market_data: Current market data
            volume_participation: Fraction of market volume we represent
            
        Returns:
            Slippage cost in currency units
        """
        if self.slippage_model == "fixed":
            return order.quantity * self.slippage_factor
        
        elif self.slippage_model == "linear":
            # Linear slippage based on order size relative to market volume
            if market_data.volume > 0:
                impact_factor = (order.quantity / market_data.volume) * volume_participation
            else:
                impact_factor = volume_participation
            
            slippage_bps = self.slippage_factor * impact_factor * 10000
            return order.quantity * market_data.mid_price * (slippage_bps / 10000)
        
        elif self.slippage_model == "sqrt":
            # Square root slippage model (common in academic literature)
            if market_data.volume > 0:
                impact_factor = np.sqrt(order.quantity / market_data.volume) * volume_participation
            else:
                impact_factor = volume_participation
            
            slippage_bps = self.slippage_factor * impact_factor * 10000
            return order.quantity * market_data.mid_price * (slippage_bps / 10000)
        
        else:
            raise ValueError(f"Unknown slippage model: {self.slippage_model}")


class MarketImpactModel:
    """Model for temporary and permanent market impact."""
    
    def __init__(self,
                 temporary_impact_factor: float = 0.0001,
                 permanent_impact_factor: float = 0.00005,
                 impact_decay_rate: float = 0.95):
        """
        Initialize market impact model.
        
        Args:
            temporary_impact_factor: Temporary impact factor
            permanent_impact_factor: Permanent impact factor
            impact_decay_rate: Rate at which temporary impact decays
        """
        self.temporary_impact_factor = temporary_impact_factor
        self.permanent_impact_factor = permanent_impact_factor
        self.impact_decay_rate = impact_decay_rate
        self.impact_history = []
    
    def calculate_impact(self, 
                        order: Order, 
                        market_data: MarketData,
                        adv: float = 1000000) -> Tuple[float, float]:
        """
        Calculate temporary and permanent market impact.
        
        Args:
            order: Order being executed
            market_data: Current market data
            adv: Average daily volume
            
        Returns:
            Tuple of (temporary_impact, permanent_impact)
        """
        # Participation rate
        participation_rate = order.quantity / adv if adv > 0 else 0.01
        
        # Temporary impact (mean-reverting)
        temp_impact = (
            self.temporary_impact_factor * 
            np.sqrt(participation_rate) * 
            market_data.mid_price
        )
        
        # Permanent impact (persistent)
        perm_impact = (
            self.permanent_impact_factor * 
            participation_rate * 
            market_data.mid_price
        )
        
        # Adjust for order side
        if order.side == OrderSide.SELL:
            temp_impact = -temp_impact
            perm_impact = -perm_impact
        
        return temp_impact, perm_impact
    
    def update_impact_decay(self):
        """Update impact decay for temporary impacts."""
        for impact in self.impact_history:
            impact['temp_impact'] *= self.impact_decay_rate


class BacktestingEngine:
    """
    Event-driven backtesting engine for market microstructure models.
    """
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 transaction_cost_model: Optional[TransactionCostModel] = None,
                 market_impact_model: Optional[MarketImpactModel] = None,
                 max_position_size: float = 0.2,
                 risk_free_rate: float = 0.02):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost_model: Model for transaction costs
            market_impact_model: Model for market impact
            max_position_size: Maximum position size as fraction of portfolio
            risk_free_rate: Risk-free rate for performance calculations
        """
        self.initial_capital = initial_capital
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        self.market_impact_model = market_impact_model or MarketImpactModel()
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        
        # State
        self.portfolio = Portfolio(cash=initial_capital)
        self.orders = []
        self.trades = []
        self.market_data_history = []
        self.portfolio_history = []
        self.current_time = None
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info(f"Initialized backtesting engine with ${initial_capital:,.2f}")
    
    def add_market_data(self, market_data: MarketData):
        """Add market data point to the engine."""
        self.market_data_history.append(market_data)
        self.current_time = market_data.timestamp
        
        # Update unrealized PnL
        market_prices = {market_data.symbol: market_data.mid_price}
        self.portfolio.calculate_unrealized_pnl(market_prices)
        
        # Record portfolio state
        self.portfolio_history.append({
            'timestamp': self.current_time,
            'cash': self.portfolio.cash,
            'total_pnl': self.portfolio.total_pnl,
            'net_liquidation_value': self.portfolio.net_liquidation_value,
            'total_commission': self.portfolio.total_commission,
            'total_slippage': self.portfolio.total_slippage,
            'total_market_impact': self.portfolio.total_market_impact
        })
    
    def place_order(self, 
                   symbol: str,
                   side: OrderSide,
                   quantity: float,
                   order_type: OrderType = OrderType.MARKET,
                   price: Optional[float] = None) -> str:
        """
        Place an order.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type
            price: Limit price (for limit orders)
            
        Returns:
            Order ID
        """
        # Risk check
        if not self._risk_check(symbol, side, quantity):
            logger.warning(f"Risk check failed for {side.value} {quantity} {symbol}")
            return None
        
        order_id = f"order_{len(self.orders) + 1}_{int(self.current_time.timestamp())}"
        
        order = Order(
            order_id=order_id,
            timestamp=self.current_time,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )
        
        self.orders.append(order)
        
        # Execute immediately for market orders
        if order_type == OrderType.MARKET and self.market_data_history:
            self._execute_order(order, self.market_data_history[-1])
        
        return order_id
    
    def _risk_check(self, symbol: str, side: OrderSide, quantity: float) -> bool:
        """Perform risk checks before placing order."""
        # Check maximum position size
        current_position = self.portfolio.get_position(symbol)
        current_market_data = self.market_data_history[-1] if self.market_data_history else None
        
        if current_market_data:
            estimated_value = quantity * current_market_data.mid_price
            portfolio_value = self.portfolio.net_liquidation_value
            
            # More relaxed position size check - allow up to max_position_size of portfolio value per trade
            # but consider current position to avoid excessive concentration
            new_position_size = abs(current_position.quantity + (quantity if side == OrderSide.BUY else -quantity))
            total_position_value = new_position_size * current_market_data.mid_price
            
            # Allow position up to max_position_size * portfolio_value
            max_allowed_value = self.max_position_size * portfolio_value
            if total_position_value > max_allowed_value:
                logger.debug(f"Position size check failed: {total_position_value:.2f} > {max_allowed_value:.2f}")
                return False
        
        # Check sufficient cash for buy orders (with smaller buffer)
        if side == OrderSide.BUY and current_market_data:
            required_cash = quantity * current_market_data.ask_price * 1.02  # 2% buffer instead of 10%
            if required_cash > self.portfolio.cash:
                logger.debug(f"Insufficient cash: need {required_cash:.2f}, have {self.portfolio.cash:.2f}")
                return False
        
        return True
    
    def _execute_order(self, order: Order, market_data: MarketData):
        """Execute an order against market data."""
        if order.symbol != market_data.symbol:
            return
        
        # Determine execution price
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                execution_price = market_data.ask_price
            else:
                execution_price = market_data.bid_price
        else:
            execution_price = order.price
        
        # Calculate costs
        trade_value = order.quantity * execution_price
        commission = self.transaction_cost_model.calculate_commission(trade_value)
        slippage = self.transaction_cost_model.calculate_slippage(order, market_data)
        temp_impact, perm_impact = self.market_impact_model.calculate_impact(order, market_data)
        
        # Adjust execution price for market impact
        if order.side == OrderSide.BUY:
            execution_price += (temp_impact + perm_impact) / order.quantity
        else:
            execution_price -= (temp_impact + perm_impact) / order.quantity
        
        # Create trade
        trade_id = f"trade_{len(self.trades) + 1}"
        trade = Trade(
            trade_id=trade_id,
            timestamp=self.current_time,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            market_impact=abs(temp_impact + perm_impact),
            slippage=slippage
        )
        
        self.trades.append(trade)
        
        # Update order
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.commission = commission
        order.status = "filled"
        
        # Update portfolio
        self.portfolio.update_position(trade)
        
        # Update statistics
        self.total_trades += 1
        
        logger.debug(f"Executed {trade.side.value} {trade.quantity} {trade.symbol} @ {trade.price:.4f}")
    
    def get_portfolio_value_series(self) -> pd.DataFrame:
        """Get portfolio value time series."""
        return pd.DataFrame(self.portfolio_history)
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'market_impact': trade.market_impact,
                'slippage': trade.slippage,
                'trade_value': trade.quantity * trade.price
            })
        
        return pd.DataFrame(trades_data)
    
    def get_positions_df(self) -> pd.DataFrame:
        """Get current positions as DataFrame."""
        positions_data = []
        for symbol, position in self.portfolio.positions.items():
            if not position.is_flat:
                positions_data.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl
                })
        
        return pd.DataFrame(positions_data)
    
    def reset(self):
        """Reset the backtesting engine."""
        self.portfolio = Portfolio(cash=self.initial_capital)
        self.orders = []
        self.trades = []
        self.market_data_history = []
        self.portfolio_history = []
        self.current_time = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info("Backtesting engine reset")


if __name__ == "__main__":
    # Test the backtesting engine
    print("Testing Backtesting Engine...")
    
    # Create engine
    engine = BacktestingEngine(initial_capital=100000.0)
    
    # Create some market data
    base_time = datetime.now()
    for i in range(10):
        market_data = MarketData(
            timestamp=base_time + timedelta(minutes=i),
            symbol="TEST",
            bid_price=100.0 + np.random.normal(0, 0.1),
            ask_price=100.1 + np.random.normal(0, 0.1),
            bid_size=1000,
            ask_size=1000,
            last_price=100.05 + np.random.normal(0, 0.1),
            volume=10000
        )
        
        engine.add_market_data(market_data)
        
        # Place some orders
        if i == 2:
            engine.place_order("TEST", OrderSide.BUY, 100)
        elif i == 5:
            engine.place_order("TEST", OrderSide.SELL, 50)
    
    # Get results
    portfolio_df = engine.get_portfolio_value_series()
    trades_df = engine.get_trades_df()
    positions_df = engine.get_positions_df()
    
    print(f"Final portfolio value: ${engine.portfolio.net_liquidation_value:.2f}")
    print(f"Total PnL: ${engine.portfolio.total_pnl:.2f}")
    print(f"Total trades: {len(trades_df)}")
    print(f"Total commission: ${engine.portfolio.total_commission:.2f}")
    
    print("✅ Backtesting engine test passed!")