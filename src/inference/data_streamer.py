"""
Real-time Data Streamer

Handles real-time market data streaming and preprocessing for inference.
Supports multiple data sources and maintains data quality for predictions.
"""

import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import threading
import queue
import time
from collections import deque
import aiohttp
from pathlib import Path
import sys

# Import data processing components
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.order_book_parser import create_synthetic_order_book_data, OrderBookSnapshot

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Standardized market data structure."""
    timestamp: datetime
    symbol: str
    mid_price: float
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    volume: float
    spread: float
    last_trade_price: Optional[float] = None
    last_trade_volume: Optional[float] = None
    vwap: Optional[float] = None


@dataclass
class StreamConfig:
    """Configuration for data streaming."""
    source_type: str  # 'websocket', 'rest', 'file', 'synthetic'
    symbol: str
    update_frequency: float  # Updates per second
    buffer_size: int = 1000
    quality_checks: bool = True
    retry_attempts: int = 3
    timeout_seconds: int = 30


class DataQualityChecker:
    """Ensures data quality for real-time inference."""
    
    def __init__(self):
        self.last_prices = deque(maxlen=100)
        self.anomaly_threshold = 0.05  # 5% price change threshold
        
    def check_data_quality(self, data: MarketData) -> bool:
        """
        Check if incoming data meets quality standards.
        
        Args:
            data: Market data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Basic validation
            if data.mid_price <= 0 or data.bid_price <= 0 or data.ask_price <= 0:
                logger.warning("Invalid price data: prices must be positive")
                return False
            
            if data.bid_price >= data.ask_price:
                logger.warning("Invalid spread: bid price >= ask price")
                return False
            
            if data.volume < 0:
                logger.warning("Invalid volume: volume must be non-negative")
                return False
            
            # Check for extreme price movements
            if len(self.last_prices) > 0:
                last_price = self.last_prices[-1]
                price_change = abs(data.mid_price - last_price) / last_price
                
                if price_change > self.anomaly_threshold:
                    logger.warning(f"Extreme price movement detected: {price_change:.2%}")
                    # Still accept but flag
                    
            # Update price history
            self.last_prices.append(data.mid_price)
            
            return True
            
        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            return False


class SyntheticDataStream:
    """Generates realistic synthetic market data for testing."""
    
    def __init__(self, symbol: str = "BTCUSD", base_price: float = 50000.0):
        self.symbol = symbol
        self.base_price = base_price
        self.current_price = base_price
        self.trend = 0.0
        self.volatility = 0.02
        
    def generate_next_data(self) -> MarketData:
        """Generate next synthetic market data point."""
        
        # Random walk with trend
        price_change = np.random.normal(self.trend, self.volatility)
        self.current_price *= (1 + price_change / 100)
        
        # Evolving trend
        self.trend += np.random.normal(0, 0.001)
        self.trend = np.clip(self.trend, -0.01, 0.01)  # Limit trend
        
        # Generate bid/ask with realistic spread
        spread_bps = np.random.uniform(1, 10)  # 1-10 basis points
        spread = self.current_price * spread_bps / 10000
        
        bid_price = self.current_price - spread / 2
        ask_price = self.current_price + spread / 2
        mid_price = (bid_price + ask_price) / 2
        
        # Generate volumes
        base_volume = 1000
        bid_volume = np.random.exponential(base_volume)
        ask_volume = np.random.exponential(base_volume)
        total_volume = bid_volume + ask_volume
        
        return MarketData(
            timestamp=datetime.now(),
            symbol=self.symbol,
            mid_price=mid_price,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            volume=total_volume,
            spread=spread,
            last_trade_price=mid_price + np.random.normal(0, spread/4),
            last_trade_volume=np.random.exponential(100),
            vwap=mid_price * (1 + np.random.normal(0, 0.001))
        )


class DataStreamer:
    """
    Real-time market data streaming system.
    
    Handles multiple data sources and provides clean, validated market data
    for real-time inference.
    """
    
    def __init__(self, config: StreamConfig):
        """
        Initialize the data streamer.
        
        Args:
            config: Streaming configuration
        """
        self.config = config
        self.quality_checker = DataQualityChecker()
        self.data_buffer = deque(maxlen=config.buffer_size)
        self.subscribers: List[Callable[[MarketData], None]] = []
        
        # Streaming control
        self.is_streaming = False
        self.stream_task: Optional[asyncio.Task] = None
        self.synthetic_generator = SyntheticDataStream(config.symbol)
        
        # Performance tracking
        self.data_count = 0
        self.error_count = 0
        self.last_update = None
        self.start_time = None
        
        logger.info(f"Initialized DataStreamer for {config.symbol} ({config.source_type})")
    
    def subscribe(self, callback: Callable[[MarketData], None]):
        """
        Subscribe to real-time data updates.
        
        Args:
            callback: Function to call with new market data
        """
        self.subscribers.append(callback)
        logger.info(f"Added subscriber, total: {len(self.subscribers)}")
    
    def unsubscribe(self, callback: Callable[[MarketData], None]):
        """Unsubscribe from data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Removed subscriber, total: {len(self.subscribers)}")
    
    def _notify_subscribers(self, data: MarketData):
        """Notify all subscribers of new data."""
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Subscriber callback failed: {e}")
    
    async def _stream_synthetic_data(self):
        """Stream synthetic market data."""
        interval = 1.0 / self.config.update_frequency
        
        while self.is_streaming:
            try:
                # Generate synthetic data
                data = self.synthetic_generator.generate_next_data()
                
                # Validate data quality
                if self.config.quality_checks and not self.quality_checker.check_data_quality(data):
                    self.error_count += 1
                    await asyncio.sleep(interval / 10)  # Brief pause on bad data
                    continue
                
                # Add to buffer
                self.data_buffer.append(data)
                
                # Notify subscribers
                self._notify_subscribers(data)
                
                # Update statistics
                self.data_count += 1
                self.last_update = datetime.now()
                
                # Control update frequency
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Synthetic streaming error: {e}")
                self.error_count += 1
                await asyncio.sleep(interval)
    
    async def _stream_websocket_data(self, url: str):
        """Stream data from WebSocket source."""
        retry_count = 0
        
        while self.is_streaming and retry_count < self.config.retry_attempts:
            try:
                async with websockets.connect(url) as websocket:
                    logger.info(f"Connected to WebSocket: {url}")
                    retry_count = 0  # Reset on successful connection
                    
                    while self.is_streaming:
                        try:
                            # Receive data with timeout
                            message = await asyncio.wait_for(
                                websocket.recv(), 
                                timeout=self.config.timeout_seconds
                            )
                            
                            # Parse message (format depends on source)
                            data = self._parse_websocket_message(message)
                            if data:
                                # Validate and process
                                if not self.config.quality_checks or self.quality_checker.check_data_quality(data):
                                    self.data_buffer.append(data)
                                    self._notify_subscribers(data)
                                    self.data_count += 1
                                    self.last_update = datetime.now()
                                else:
                                    self.error_count += 1
                            
                        except asyncio.TimeoutError:
                            logger.warning("WebSocket timeout, checking connection...")
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed")
                            break
                            
            except Exception as e:
                retry_count += 1
                logger.error(f"WebSocket error (attempt {retry_count}): {e}")
                if retry_count < self.config.retry_attempts:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    logger.error("Max retry attempts reached for WebSocket")
                    break
    
    def _parse_websocket_message(self, message: str) -> Optional[MarketData]:
        """Parse WebSocket message to MarketData format."""
        try:
            data = json.loads(message)
            
            # Example parsing for common WebSocket formats
            # This would need to be customized for specific data providers
            return MarketData(
                timestamp=datetime.fromtimestamp(data.get('timestamp', time.time())),
                symbol=data.get('symbol', self.config.symbol),
                mid_price=float(data.get('price', 0)),
                bid_price=float(data.get('bid', 0)),
                ask_price=float(data.get('ask', 0)),
                bid_volume=float(data.get('bid_size', 0)),
                ask_volume=float(data.get('ask_size', 0)),
                volume=float(data.get('volume', 0)),
                spread=float(data.get('spread', 0))
            )
            
        except Exception as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
            return None
    
    async def _stream_rest_data(self, url: str):
        """Stream data from REST API with polling."""
        interval = 1.0 / self.config.update_frequency
        
        async with aiohttp.ClientSession() as session:
            while self.is_streaming:
                try:
                    async with session.get(url, timeout=self.config.timeout_seconds) as response:
                        if response.status == 200:
                            data_json = await response.json()
                            data = self._parse_rest_response(data_json)
                            
                            if data and (not self.config.quality_checks or 
                                       self.quality_checker.check_data_quality(data)):
                                self.data_buffer.append(data)
                                self._notify_subscribers(data)
                                self.data_count += 1
                                self.last_update = datetime.now()
                            else:
                                self.error_count += 1
                        else:
                            logger.warning(f"REST API error: {response.status}")
                            self.error_count += 1
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"REST streaming error: {e}")
                    self.error_count += 1
                    await asyncio.sleep(interval)
    
    def _parse_rest_response(self, data: Dict) -> Optional[MarketData]:
        """Parse REST API response to MarketData format."""
        try:
            # Example parsing - customize for specific APIs
            return MarketData(
                timestamp=datetime.now(),
                symbol=data.get('symbol', self.config.symbol),
                mid_price=float(data.get('price', 0)),
                bid_price=float(data.get('bid', 0)),
                ask_price=float(data.get('ask', 0)),
                bid_volume=float(data.get('bid_volume', 0)),
                ask_volume=float(data.get('ask_volume', 0)),
                volume=float(data.get('volume', 0)),
                spread=float(data.get('spread', 0))
            )
        except Exception as e:
            logger.error(f"Failed to parse REST response: {e}")
            return None
    
    async def start_streaming(self, source_url: Optional[str] = None):
        """
        Start the data streaming.
        
        Args:
            source_url: URL for external data sources (WebSocket/REST)
        """
        if self.is_streaming:
            logger.warning("Streaming already started")
            return
        
        self.is_streaming = True
        self.start_time = datetime.now()
        
        logger.info(f"Starting {self.config.source_type} streaming for {self.config.symbol}")
        
        # Select streaming method based on source type
        if self.config.source_type == 'synthetic':
            self.stream_task = asyncio.create_task(self._stream_synthetic_data())
        elif self.config.source_type == 'websocket' and source_url:
            self.stream_task = asyncio.create_task(self._stream_websocket_data(source_url))
        elif self.config.source_type == 'rest' and source_url:
            self.stream_task = asyncio.create_task(self._stream_rest_data(source_url))
        else:
            # Default to synthetic for demo
            logger.info("No valid source specified, using synthetic data")
            self.stream_task = asyncio.create_task(self._stream_synthetic_data())
    
    async def stop_streaming(self):
        """Stop the data streaming."""
        if not self.is_streaming:
            logger.warning("Streaming not active")
            return
        
        logger.info("Stopping data streaming...")
        self.is_streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Data streaming stopped")
    
    def get_recent_data(self, count: int = 10) -> List[MarketData]:
        """Get recent market data from buffer."""
        return list(self.data_buffer)[-count:] if self.data_buffer else []
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        data_rate = self.data_count / uptime if uptime > 0 else 0
        error_rate = self.error_count / max(self.data_count, 1)
        
        return {
            'is_streaming': self.is_streaming,
            'symbol': self.config.symbol,
            'source_type': self.config.source_type,
            'uptime_seconds': uptime,
            'total_data_points': self.data_count,
            'total_errors': self.error_count,
            'data_rate_per_second': data_rate,
            'error_rate': error_rate,
            'buffer_size': len(self.data_buffer),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'subscribers': len(self.subscribers)
        }


class MultiSymbolStreamer:
    """Manages streaming for multiple symbols simultaneously."""
    
    def __init__(self):
        self.streamers: Dict[str, DataStreamer] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def add_symbol(self, symbol: str, config: StreamConfig):
        """Add a new symbol for streaming."""
        if symbol in self.streamers:
            logger.warning(f"Symbol {symbol} already being streamed")
            return
        
        streamer = DataStreamer(config)
        self.streamers[symbol] = streamer
        self.subscribers[symbol] = []
        
        logger.info(f"Added symbol {symbol} for streaming")
    
    def subscribe_to_symbol(self, symbol: str, callback: Callable[[MarketData], None]):
        """Subscribe to updates for a specific symbol."""
        if symbol not in self.streamers:
            logger.error(f"Symbol {symbol} not found")
            return
        
        self.streamers[symbol].subscribe(callback)
        self.subscribers[symbol].append(callback)
    
    async def start_all_streaming(self):
        """Start streaming for all symbols."""
        tasks = []
        for symbol, streamer in self.streamers.items():
            task = streamer.start_streaming()
            if task:
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def stop_all_streaming(self):
        """Stop streaming for all symbols."""
        tasks = []
        for streamer in self.streamers.values():
            tasks.append(streamer.stop_streaming())
        
        if tasks:
            await asyncio.gather(*tasks)
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all streamers."""
        return {
            symbol: streamer.get_streaming_stats()
            for symbol, streamer in self.streamers.items()
        }


# Example usage and testing
async def test_data_streamer():
    """Test the data streaming functionality."""
    
    # Configure synthetic data streaming
    config = StreamConfig(
        source_type='synthetic',
        symbol='BTCUSD',
        update_frequency=10.0,  # 10 updates per second
        buffer_size=100,
        quality_checks=True
    )
    
    # Create streamer
    streamer = DataStreamer(config)
    
    # Data collector for testing
    received_data = []
    
    def data_handler(data: MarketData):
        received_data.append(data)
        print(f"Received: {data.symbol} @ {data.mid_price:.2f} "
              f"(spread: {data.spread:.4f}) at {data.timestamp}")
    
    # Subscribe to data
    streamer.subscribe(data_handler)
    
    # Start streaming
    await streamer.start_streaming()
    
    # Let it run for a few seconds
    await asyncio.sleep(5)
    
    # Get stats
    stats = streamer.get_streaming_stats()
    print(f"Streaming stats: {stats}")
    
    # Stop streaming
    await streamer.stop_streaming()
    
    print(f"Collected {len(received_data)} data points")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_data_streamer())