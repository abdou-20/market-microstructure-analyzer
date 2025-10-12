"""
Prediction Engine for Real-time Inference

Orchestrates model server and data streaming for real-time predictions
with buffering, caching, and performance monitoring.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Deque
from collections import deque
from datetime import datetime, timedelta
import logging
import threading
import time
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# Import components
from .model_server import ModelServer, PredictionResult
from .data_streamer import DataStreamer, MarketData, StreamConfig

logger = logging.getLogger(__name__)


@dataclass
class CachedPrediction:
    """Cached prediction with metadata."""
    prediction: PredictionResult
    market_data_hash: str
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class PredictionBuffer:
    """Buffer for managing prediction history."""
    predictions: Deque[PredictionResult]
    max_size: int
    created_at: datetime


class PredictionCache:
    """Advanced caching system for predictions."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize prediction cache.
        
        Args:
            max_size: Maximum number of cached predictions
            ttl_seconds: Time-to-live for cached predictions
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CachedPrediction] = {}
        self.lock = threading.RLock()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def _generate_key(self, market_data: List[Dict]) -> str:
        """Generate cache key from market data."""
        # Create hash from key market data features
        key_data = []
        for data in market_data[-5:]:  # Use last 5 data points for key
            key_data.extend([
                data.get('mid_price', 0),
                data.get('spread', 0),
                data.get('volume', 0)
            ])
        
        return str(hash(tuple(key_data)))
    
    def get(self, market_data: List[Dict]) -> Optional[PredictionResult]:
        """Get cached prediction if available."""
        key = self._generate_key(market_data)
        
        with self.lock:
            if key in self.cache:
                cached = self.cache[key]
                
                # Check TTL
                age = (datetime.now() - cached.created_at).total_seconds()
                if age <= self.ttl_seconds:
                    # Update access statistics
                    cached.access_count += 1
                    cached.last_accessed = datetime.now()
                    self.hits += 1
                    
                    logger.debug(f"Cache hit for key {key[:8]}...")
                    return cached.prediction
                else:
                    # Expired entry
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, market_data: List[Dict], prediction: PredictionResult):
        """Cache a prediction."""
        key = self._generate_key(market_data)
        data_hash = self._generate_key(market_data)
        
        with self.lock:
            # Check if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Store prediction
            self.cache[key] = CachedPrediction(
                prediction=prediction,
                market_data_hash=data_hash,
                created_at=datetime.now()
            )
            
            logger.debug(f"Cached prediction for key {key[:8]}...")
    
    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if not self.cache:
            return
        
        # Find oldest entry
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].created_at
        )
        
        del self.cache[oldest_key]
        self.evictions += 1
        
        logger.debug(f"Evicted cache entry {oldest_key[:8]}...")
    
    def clear(self):
        """Clear all cached predictions."""
        with self.lock:
            self.cache.clear()
            logger.info("Prediction cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'ttl_seconds': self.ttl_seconds
            }


class PredictionEngine:
    """
    Real-time prediction engine.
    
    Orchestrates model server and data streaming to provide real-time
    market predictions with caching, buffering, and monitoring.
    """
    
    def __init__(self,
                 model_server: ModelServer,
                 stream_config: StreamConfig,
                 buffer_size: int = 1000,
                 cache_size: int = 500,
                 cache_ttl: int = 300,
                 prediction_interval: float = 1.0):
        """
        Initialize prediction engine.
        
        Args:
            model_server: Initialized model server
            stream_config: Data streaming configuration
            buffer_size: Size of prediction buffer
            cache_size: Size of prediction cache
            cache_ttl: Cache time-to-live in seconds
            prediction_interval: Prediction frequency in seconds
        """
        self.model_server = model_server
        self.stream_config = stream_config
        self.prediction_interval = prediction_interval
        
        # Initialize data streamer
        self.data_streamer = DataStreamer(stream_config)
        
        # Initialize caching and buffering
        self.cache = PredictionCache(cache_size, cache_ttl)
        self.prediction_buffer = PredictionBuffer(
            predictions=deque(maxlen=buffer_size),
            max_size=buffer_size,
            created_at=datetime.now()
        )
        
        # Market data buffer
        self.market_data_buffer = deque(maxlen=50)  # Keep last 50 data points
        
        # Prediction management
        self.is_running = False
        self.prediction_task: Optional[asyncio.Task] = None
        self.subscribers: List[Callable[[PredictionResult], None]] = []
        
        # Performance tracking
        self.predictions_made = 0
        self.cache_hits = 0
        self.prediction_times = deque(maxlen=100)
        self.start_time: Optional[datetime] = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized PredictionEngine for {stream_config.symbol}")
    
    def subscribe(self, callback: Callable[[PredictionResult], None]):
        """Subscribe to prediction updates."""
        with self.lock:
            self.subscribers.append(callback)
            logger.info(f"Added prediction subscriber, total: {len(self.subscribers)}")
    
    def unsubscribe(self, callback: Callable[[PredictionResult], None]):
        """Unsubscribe from prediction updates."""
        with self.lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)
                logger.info(f"Removed prediction subscriber, total: {len(self.subscribers)}")
    
    def _notify_subscribers(self, prediction: PredictionResult):
        """Notify all subscribers of new prediction."""
        with self.lock:
            for callback in self.subscribers:
                try:
                    callback(prediction)
                except Exception as e:
                    logger.error(f"Subscriber notification failed: {e}")
    
    def _on_market_data(self, data: MarketData):
        """Handle new market data from streamer."""
        # Convert MarketData to dict format for processing
        data_dict = {
            'timestamp': data.timestamp,
            'mid_price': data.mid_price,
            'bid_price': data.bid_price,
            'ask_price': data.ask_price,
            'bid_volume': data.bid_volume,
            'ask_volume': data.ask_volume,
            'volume': data.volume,
            'spread': data.spread
        }
        
        # Add to market data buffer
        self.market_data_buffer.append(data_dict)
        
        logger.debug(f"Received market data: {data.symbol} @ {data.mid_price:.2f}")
    
    async def _prediction_loop(self):
        """Main prediction loop."""
        while self.is_running:
            try:
                # Wait for sufficient market data
                if len(self.market_data_buffer) < 25:  # Need sequence length
                    await asyncio.sleep(0.1)
                    continue
                
                # Prepare market data for prediction
                market_data_list = list(self.market_data_buffer)
                
                # Check cache first
                cached_prediction = self.cache.get(market_data_list)
                if cached_prediction:
                    self.cache_hits += 1
                    self._notify_subscribers(cached_prediction)
                    self.prediction_buffer.predictions.append(cached_prediction)
                    
                    logger.debug("Used cached prediction")
                else:
                    # Make new prediction
                    start_time = time.time()
                    
                    try:
                        prediction = await self.model_server.predict_async(market_data_list)
                        
                        prediction_time = (time.time() - start_time) * 1000
                        self.prediction_times.append(prediction_time)
                        
                        # Cache the prediction
                        self.cache.put(market_data_list, prediction)
                        
                        # Add to buffer
                        self.prediction_buffer.predictions.append(prediction)
                        
                        # Update statistics
                        with self.lock:
                            self.predictions_made += 1
                        
                        # Notify subscribers
                        self._notify_subscribers(prediction)
                        
                        logger.debug(f"New prediction: direction={prediction.predicted_direction}, "
                                   f"confidence={prediction.confidence:.3f}, "
                                   f"time={prediction_time:.1f}ms")
                        
                    except Exception as e:
                        logger.error(f"Prediction failed: {e}")
                
                # Wait for next prediction cycle
                await asyncio.sleep(self.prediction_interval)
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def start(self, source_url: Optional[str] = None):
        """Start the prediction engine."""
        if self.is_running:
            logger.warning("Prediction engine already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info(f"Starting prediction engine for {self.stream_config.symbol}")
        
        # Subscribe to data stream
        self.data_streamer.subscribe(self._on_market_data)
        
        # Start data streaming
        await self.data_streamer.start_streaming(source_url)
        
        # Start prediction loop
        self.prediction_task = asyncio.create_task(self._prediction_loop())
        
        logger.info("Prediction engine started successfully")
    
    async def stop(self):
        """Stop the prediction engine."""
        if not self.is_running:
            logger.warning("Prediction engine not running")
            return
        
        logger.info("Stopping prediction engine...")
        self.is_running = False
        
        # Stop data streaming
        await self.data_streamer.stop_streaming()
        
        # Cancel prediction task
        if self.prediction_task:
            self.prediction_task.cancel()
            try:
                await self.prediction_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Prediction engine stopped")
    
    def get_recent_predictions(self, count: int = 10) -> List[PredictionResult]:
        """Get recent predictions from buffer."""
        with self.lock:
            return list(self.prediction_buffer.predictions)[-count:]
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction engine statistics."""
        with self.lock:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            prediction_rate = self.predictions_made / uptime if uptime > 0 else 0
            
            avg_prediction_time = (
                np.mean(self.prediction_times) if self.prediction_times else 0
            )
            
            return {
                'is_running': self.is_running,
                'symbol': self.stream_config.symbol,
                'uptime_seconds': uptime,
                'total_predictions': self.predictions_made,
                'cache_hits': self.cache_hits,
                'cache_hit_rate': self.cache_hits / max(self.predictions_made, 1),
                'prediction_rate_per_second': prediction_rate,
                'average_prediction_time_ms': avg_prediction_time,
                'buffer_size': len(self.prediction_buffer.predictions),
                'market_data_buffer_size': len(self.market_data_buffer),
                'subscribers': len(self.subscribers),
                'cache_stats': self.cache.get_stats(),
                'streaming_stats': self.data_streamer.get_streaming_stats()
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        stats = self.get_prediction_stats()
        recent_predictions = self.get_recent_predictions(20)
        
        # Analyze recent prediction patterns
        if recent_predictions:
            directions = [p.predicted_direction for p in recent_predictions]
            confidences = [p.confidence for p in recent_predictions]
            
            direction_counts = {
                'up': sum(1 for d in directions if d > 0),
                'down': sum(1 for d in directions if d < 0),
                'neutral': sum(1 for d in directions if d == 0)
            }
            
            avg_confidence = np.mean(confidences)
            
            prediction_analysis = {
                'recent_prediction_count': len(recent_predictions),
                'direction_distribution': direction_counts,
                'average_confidence': avg_confidence,
                'confidence_std': np.std(confidences),
                'last_prediction_time': recent_predictions[-1].timestamp.isoformat()
            }
        else:
            prediction_analysis = {
                'recent_prediction_count': 0,
                'direction_distribution': {'up': 0, 'down': 0, 'neutral': 0},
                'average_confidence': 0.0,
                'confidence_std': 0.0,
                'last_prediction_time': None
            }
        
        return {
            'engine_stats': stats,
            'prediction_analysis': prediction_analysis,
            'model_info': self.model_server.get_model_info(),
            'model_performance': self.model_server.get_performance_stats()
        }
    
    def clear_cache(self):
        """Clear prediction cache."""
        self.cache.clear()
        logger.info("Prediction cache cleared")
    
    def export_predictions(self, filepath: str, limit: Optional[int] = None):
        """Export predictions to file."""
        predictions = self.get_recent_predictions(limit) if limit else list(self.prediction_buffer.predictions)
        
        # Convert to serializable format
        export_data = []
        for pred in predictions:
            export_data.append({
                'timestamp': pred.timestamp.isoformat(),
                'predicted_direction': pred.predicted_direction,
                'confidence': pred.confidence,
                'predicted_change': pred.predicted_change,
                'directional_probability': pred.directional_probability,
                'model_name': pred.model_name,
                'inference_time_ms': pred.inference_time_ms
            })
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} predictions to {filepath}")


# Testing and example usage
async def test_prediction_engine():
    """Test the prediction engine."""
    # Initialize model server
    model_server = ModelServer()
    
    # Configure streaming
    stream_config = StreamConfig(
        source_type='synthetic',
        symbol='BTCUSD',
        update_frequency=2.0,  # 2 predictions per second
        buffer_size=100
    )
    
    # Create prediction engine
    engine = PredictionEngine(
        model_server=model_server,
        stream_config=stream_config,
        buffer_size=50,
        cache_size=100,
        prediction_interval=0.5  # Predict every 0.5 seconds
    )
    
    # Prediction collector for testing
    received_predictions = []
    
    def prediction_handler(prediction: PredictionResult):
        received_predictions.append(prediction)
        print(f"Prediction: {prediction.predicted_direction} "
              f"(confidence: {prediction.confidence:.3f}) "
              f"at {prediction.timestamp}")
    
    # Subscribe to predictions
    engine.subscribe(prediction_handler)
    
    # Start engine
    await engine.start()
    
    # Let it run for a few seconds
    await asyncio.sleep(10)
    
    # Get performance stats
    stats = engine.get_performance_summary()
    print(f"Engine stats: {json.dumps(stats, indent=2, default=str)}")
    
    # Stop engine
    await engine.stop()
    
    print(f"Collected {len(received_predictions)} predictions")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_prediction_engine())