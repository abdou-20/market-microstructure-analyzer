"""
Order Book Parser Module

This module provides functionality to parse Level 2 order book data from various formats
including CSV, binary, and JSON. It handles real-time and historical order book data
with proper validation and error handling.
"""

from typing import Dict, List, Optional, Union, Any, Iterator
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import struct

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Represents a single order book snapshot at a specific timestamp."""
    timestamp: datetime
    symbol: str
    bids: List[tuple]  # List of (price, quantity) tuples
    asks: List[tuple]  # List of (price, quantity) tuples
    best_bid: float
    best_ask: float
    spread: float
    mid_price: float
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.bids and self.asks:
            self.best_bid = max(self.bids, key=lambda x: x[0])[0] if self.bids else 0.0
            self.best_ask = min(self.asks, key=lambda x: x[0])[0] if self.asks else float('inf')
            self.spread = self.best_ask - self.best_bid
            self.mid_price = (self.best_bid + self.best_ask) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'bids': self.bids,
            'asks': self.asks,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'spread': self.spread,
            'mid_price': self.mid_price
        }


class OrderBookParser:
    """
    Main parser class for handling various order book data formats.
    
    Supports:
    - CSV format with standard columns
    - Binary format for high-frequency data
    - JSON format for API responses
    - Real-time data streams
    """
    
    def __init__(self, 
                 max_levels: int = 10,
                 validate_data: bool = True,
                 sort_levels: bool = True):
        """
        Initialize the order book parser.
        
        Args:
            max_levels: Maximum number of bid/ask levels to retain
            validate_data: Whether to validate order book data integrity
            sort_levels: Whether to sort bid/ask levels by price
        """
        self.max_levels = max_levels
        self.validate_data = validate_data
        self.sort_levels = sort_levels
        self.processed_count = 0
        self.error_count = 0
    
    def parse_csv(self, file_path: Union[str, Path]) -> Iterator[OrderBookSnapshot]:
        """
        Parse order book data from CSV format.
        
        Expected CSV format:
        timestamp,symbol,side,price,quantity,level
        
        Args:
            file_path: Path to the CSV file
            
        Yields:
            OrderBookSnapshot objects
        """
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Group by timestamp and symbol
            grouped = df.groupby(['timestamp', 'symbol'])
            
            for (timestamp, symbol), group in grouped:
                bids = []
                asks = []
                
                for _, row in group.iterrows():
                    price_qty = (float(row['price']), float(row['quantity']))
                    
                    if row['side'].lower() == 'bid':
                        bids.append(price_qty)
                    elif row['side'].lower() == 'ask':
                        asks.append(price_qty)
                
                if self.sort_levels:
                    bids.sort(key=lambda x: x[0], reverse=True)  # Highest bid first
                    asks.sort(key=lambda x: x[0])  # Lowest ask first
                
                # Limit to max levels
                bids = bids[:self.max_levels]
                asks = asks[:self.max_levels]
                
                snapshot = OrderBookSnapshot(
                    timestamp=timestamp,
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    best_bid=0.0,  # Will be calculated in __post_init__
                    best_ask=0.0,
                    spread=0.0,
                    mid_price=0.0
                )
                
                if self.validate_data and self._validate_snapshot(snapshot):
                    yield snapshot
                    self.processed_count += 1
                elif not self.validate_data:
                    yield snapshot
                    self.processed_count += 1
                else:
                    self.error_count += 1
                    logger.warning(f"Invalid snapshot at {timestamp} for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {e}")
            raise
    
    def parse_json(self, file_path: Union[str, Path]) -> Iterator[OrderBookSnapshot]:
        """
        Parse order book data from JSON format.
        
        Expected JSON format:
        {
            "timestamp": "2023-01-01T12:00:00Z",
            "symbol": "BTC-USD",
            "bids": [[50000.0, 1.5], [49999.0, 2.0]],
            "asks": [[50001.0, 1.2], [50002.0, 0.8]]
        }
        
        Args:
            file_path: Path to the JSON file
            
        Yields:
            OrderBookSnapshot objects
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Multiple snapshots in array
                for item in data:
                    snapshot = self._parse_json_item(item)
                    if snapshot:
                        yield snapshot
            else:
                # Single snapshot
                snapshot = self._parse_json_item(data)
                if snapshot:
                    yield snapshot
                    
        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            raise
    
    def _parse_json_item(self, item: Dict) -> Optional[OrderBookSnapshot]:
        """Parse a single JSON item into OrderBookSnapshot."""
        try:
            timestamp = pd.to_datetime(item['timestamp'])
            symbol = item['symbol']
            bids = [(float(bid[0]), float(bid[1])) for bid in item['bids']]
            asks = [(float(ask[0]), float(ask[1])) for ask in item['asks']]
            
            if self.sort_levels:
                bids.sort(key=lambda x: x[0], reverse=True)
                asks.sort(key=lambda x: x[0])
            
            bids = bids[:self.max_levels]
            asks = asks[:self.max_levels]
            
            snapshot = OrderBookSnapshot(
                timestamp=timestamp,
                symbol=symbol,
                bids=bids,
                asks=asks,
                best_bid=0.0,
                best_ask=0.0,
                spread=0.0,
                mid_price=0.0
            )
            
            if self.validate_data and self._validate_snapshot(snapshot):
                self.processed_count += 1
                return snapshot
            elif not self.validate_data:
                self.processed_count += 1
                return snapshot
            else:
                self.error_count += 1
                return None
                
        except Exception as e:
            logger.error(f"Error parsing JSON item: {e}")
            self.error_count += 1
            return None
    
    def parse_binary(self, file_path: Union[str, Path]) -> Iterator[OrderBookSnapshot]:
        """
        Parse order book data from binary format.
        
        Binary format (little-endian):
        - 8 bytes: timestamp (Unix timestamp as double)
        - 4 bytes: symbol length
        - N bytes: symbol string
        - 4 bytes: number of bid levels
        - For each bid: 8 bytes price (double), 8 bytes quantity (double)
        - 4 bytes: number of ask levels
        - For each ask: 8 bytes price (double), 8 bytes quantity (double)
        
        Args:
            file_path: Path to the binary file
            
        Yields:
            OrderBookSnapshot objects
        """
        try:
            with open(file_path, 'rb') as f:
                while True:
                    # Read timestamp
                    timestamp_bytes = f.read(8)
                    if len(timestamp_bytes) < 8:
                        break
                    
                    timestamp = struct.unpack('<d', timestamp_bytes)[0]
                    timestamp = datetime.fromtimestamp(timestamp)
                    
                    # Read symbol
                    symbol_len = struct.unpack('<I', f.read(4))[0]
                    symbol = f.read(symbol_len).decode('utf-8')
                    
                    # Read bids
                    num_bids = struct.unpack('<I', f.read(4))[0]
                    bids = []
                    for _ in range(num_bids):
                        price = struct.unpack('<d', f.read(8))[0]
                        quantity = struct.unpack('<d', f.read(8))[0]
                        bids.append((price, quantity))
                    
                    # Read asks
                    num_asks = struct.unpack('<I', f.read(4))[0]
                    asks = []
                    for _ in range(num_asks):
                        price = struct.unpack('<d', f.read(8))[0]
                        quantity = struct.unpack('<d', f.read(8))[0]
                        asks.append((price, quantity))
                    
                    if self.sort_levels:
                        bids.sort(key=lambda x: x[0], reverse=True)
                        asks.sort(key=lambda x: x[0])
                    
                    bids = bids[:self.max_levels]
                    asks = asks[:self.max_levels]
                    
                    snapshot = OrderBookSnapshot(
                        timestamp=timestamp,
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        best_bid=0.0,
                        best_ask=0.0,
                        spread=0.0,
                        mid_price=0.0
                    )
                    
                    if self.validate_data and self._validate_snapshot(snapshot):
                        yield snapshot
                        self.processed_count += 1
                    elif not self.validate_data:
                        yield snapshot
                        self.processed_count += 1
                    else:
                        self.error_count += 1
                        
        except Exception as e:
            logger.error(f"Error parsing binary file {file_path}: {e}")
            raise
    
    def _validate_snapshot(self, snapshot: OrderBookSnapshot) -> bool:
        """
        Validate order book snapshot for consistency.
        
        Args:
            snapshot: OrderBookSnapshot to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check basic requirements
            if not snapshot.bids or not snapshot.asks:
                return False
            
            # Check that bids are sorted in descending order
            bid_prices = [bid[0] for bid in snapshot.bids]
            if bid_prices != sorted(bid_prices, reverse=True):
                return False
            
            # Check that asks are sorted in ascending order
            ask_prices = [ask[0] for ask in snapshot.asks]
            if ask_prices != sorted(ask_prices):
                return False
            
            # Check that best bid < best ask (no crossed book)
            if snapshot.best_bid >= snapshot.best_ask:
                return False
            
            # Check that all prices and quantities are positive
            for price, qty in snapshot.bids + snapshot.asks:
                if price <= 0 or qty <= 0:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, int]:
        """Get parsing statistics."""
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'success_rate': self.processed_count / (self.processed_count + self.error_count) * 100 
                          if (self.processed_count + self.error_count) > 0 else 0
        }


def create_synthetic_order_book_data(
    symbol: str = "BTC-USD",
    num_snapshots: int = 1000,
    start_price: float = 50000.0,
    volatility: float = 0.001,
    num_levels: int = 10
) -> List[OrderBookSnapshot]:
    """
    Create synthetic order book data for testing purposes.
    
    Args:
        symbol: Trading symbol
        num_snapshots: Number of snapshots to generate
        start_price: Starting mid price
        volatility: Price volatility per snapshot
        num_levels: Number of bid/ask levels
        
    Returns:
        List of OrderBookSnapshot objects
    """
    snapshots = []
    current_time = datetime.now()
    current_price = start_price
    
    for i in range(num_snapshots):
        # Random walk for price
        price_change = np.random.normal(0, volatility * current_price)
        current_price = max(current_price + price_change, 1.0)  # Ensure positive price
        
        # Generate bids (below mid price)
        bids = []
        for level in range(num_levels):
            price = current_price - (level + 0.5) * current_price * 0.0001
            quantity = np.random.exponential(10.0)
            bids.append((price, quantity))
        
        # Generate asks (above mid price)
        asks = []
        for level in range(num_levels):
            price = current_price + (level + 0.5) * current_price * 0.0001
            quantity = np.random.exponential(10.0)
            asks.append((price, quantity))
        
        # Sort levels
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        
        snapshot = OrderBookSnapshot(
            timestamp=current_time,
            symbol=symbol,
            bids=bids,
            asks=asks,
            best_bid=0.0,
            best_ask=0.0,
            spread=0.0,
            mid_price=0.0
        )
        
        snapshots.append(snapshot)
        current_time = current_time + pd.Timedelta(seconds=1)
    
    return snapshots


if __name__ == "__main__":
    # Example usage
    parser = OrderBookParser(max_levels=10, validate_data=True)
    
    # Generate synthetic data for testing
    synthetic_data = create_synthetic_order_book_data(num_snapshots=100)
    print(f"Generated {len(synthetic_data)} synthetic order book snapshots")
    
    # Example snapshot
    if synthetic_data:
        sample = synthetic_data[0]
        print(f"Sample snapshot: {sample.symbol} at {sample.timestamp}")
        print(f"Best bid: {sample.best_bid:.2f}, Best ask: {sample.best_ask:.2f}")
        print(f"Spread: {sample.spread:.4f}, Mid price: {sample.mid_price:.2f}")