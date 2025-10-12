#!/usr/bin/env python3
"""
Test script to verify the backtesting engine fix works properly.
"""

import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from backtesting.engine import BacktestingEngine, MarketData, OrderSide

def test_backtesting_fix():
    """Test that trades are now executing properly."""
    print("Testing Backtesting Engine Fix...")
    
    # Create engine with realistic parameters
    engine = BacktestingEngine(
        initial_capital=50000.0,
        max_position_size=0.2  # Allow up to 20% position size
    )
    
    # Create market data
    base_time = datetime.now()
    base_price = 100.0
    
    trades_executed = 0
    
    for i in range(20):
        # Create realistic market data
        price_change = np.random.normal(0, 0.1)
        current_price = base_price + price_change
        
        market_data = MarketData(
            timestamp=base_time + timedelta(minutes=i),
            symbol="BACKTEST",
            bid_price=current_price - 0.05,
            ask_price=current_price + 0.05,
            bid_size=1000,
            ask_size=1000,
            last_price=current_price,
            volume=10000,
            features=np.random.randn(46)
        )
        
        engine.add_market_data(market_data)
        
        # Try to place orders at different intervals
        if i % 3 == 0:  # Every 3rd interval
            # Buy order
            order_id = engine.place_order("BACKTEST", OrderSide.BUY, 100)
            if order_id:
                trades_executed += 1
                print(f"✅ Buy order executed: {order_id}")
            else:
                print(f"❌ Buy order rejected at step {i}")
                
        elif i % 5 == 0:  # Every 5th interval
            # Sell order (if we have positions)
            current_position = engine.portfolio.get_position("BACKTEST")
            if current_position.quantity > 0:
                order_id = engine.place_order("BACKTEST", OrderSide.SELL, 50)
                if order_id:
                    trades_executed += 1
                    print(f"✅ Sell order executed: {order_id}")
                else:
                    print(f"❌ Sell order rejected at step {i}")
    
    # Get final results
    portfolio_df = engine.get_portfolio_value_series()
    trades_df = engine.get_trades_df()
    
    print(f"\n📊 RESULTS:")
    print(f"Initial Capital: ${engine.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${engine.portfolio.net_liquidation_value:,.2f}")
    print(f"Total Return: {((engine.portfolio.net_liquidation_value / engine.initial_capital) - 1) * 100:.2f}%")
    print(f"Total Trades Attempted: {trades_executed}")
    print(f"Total Trades Executed: {len(trades_df)}")
    print(f"Total Commission: ${engine.portfolio.total_commission:.2f}")
    print(f"Market Data Points: {len(portfolio_df)}")
    
    # Check if fix worked
    if len(trades_df) > 0:
        print(f"\n✅ SUCCESS: Risk management fix worked! {len(trades_df)} trades executed.")
        
        # Show some trades
        print("\nSample Trades:")
        for _, trade in trades_df.head(3).iterrows():
            print(f"  {trade['side'].upper()} {trade['quantity']} @ ${trade['price']:.4f} "
                  f"(commission: ${trade['commission']:.2f})")
        
        return True
    else:
        print(f"\n❌ FAILED: No trades executed. Risk management still too restrictive.")
        return False

if __name__ == "__main__":
    success = test_backtesting_fix()
    
    if success:
        print("\n🎉 Backtesting engine fix verification PASSED!")
    else:
        print("\n💥 Backtesting engine fix verification FAILED!")
        sys.exit(1)