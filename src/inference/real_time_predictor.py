"""
Real-time Predictor - Complete End-to-End Inference System

Provides a high-level interface for real-time market predictions
combining all inference components into a unified system.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

# Import inference components
from .model_server import ModelServer, PredictionResult
from .data_streamer import DataStreamer, MarketData, StreamConfig
from .prediction_engine import PredictionEngine

logger = logging.getLogger(__name__)


@dataclass
class PredictorConfig:
    """Configuration for real-time predictor."""
    symbol: str
    model_path: Optional[str] = None
    data_source_type: str = 'synthetic'  # 'synthetic', 'websocket', 'rest'
    data_source_url: Optional[str] = None
    update_frequency: float = 1.0  # Updates per second
    prediction_interval: float = 1.0  # Predictions per second
    buffer_size: int = 1000
    cache_size: int = 500
    cache_ttl: int = 300
    enable_monitoring: bool = True
    log_level: str = 'INFO'


@dataclass
class SystemHealth:
    """System health status."""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    components: Dict[str, bool]
    performance_metrics: Dict[str, float]
    last_check: datetime
    issues: List[str]


class RealTimePredictor:
    """
    Complete real-time prediction system.
    
    Provides a unified interface for real-time market predictions,
    combining model serving, data streaming, and prediction management.
    """
    
    def __init__(self, config: PredictorConfig):
        """
        Initialize real-time predictor.
        
        Args:
            config: Predictor configuration
        """
        self.config = config
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        
        # Initialize components
        self.model_server: Optional[ModelServer] = None
        self.prediction_engine: Optional[PredictionEngine] = None
        
        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        
        # Monitoring and callbacks
        self.prediction_callbacks: List[Callable[[PredictionResult], None]] = []
        self.health_callbacks: List[Callable[[SystemHealth], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        
        # Performance tracking
        self.total_predictions = 0
        self.total_errors = 0
        self.performance_history = []
        
        logger.info(f"Initialized RealTimePredictor for {config.symbol}")
    
    async def initialize(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing real-time predictor components...")
            
            # Initialize model server
            self.model_server = ModelServer(
                model_path=self.config.model_path,
                device=None,  # Auto-detect best device
                max_workers=4
            )
            
            # Create stream configuration
            stream_config = StreamConfig(
                source_type=self.config.data_source_type,
                symbol=self.config.symbol,
                update_frequency=self.config.update_frequency,
                buffer_size=self.config.buffer_size,
                quality_checks=True
            )
            
            # Initialize prediction engine
            self.prediction_engine = PredictionEngine(
                model_server=self.model_server,
                stream_config=stream_config,
                buffer_size=self.config.buffer_size,
                cache_size=self.config.cache_size,
                cache_ttl=self.config.cache_ttl,
                prediction_interval=self.config.prediction_interval
            )
            
            # Subscribe to predictions
            self.prediction_engine.subscribe(self._on_prediction)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def _on_prediction(self, prediction: PredictionResult):
        """Handle new predictions from engine."""
        try:
            self.total_predictions += 1
            
            # Notify callbacks
            for callback in self.prediction_callbacks:
                try:
                    callback(prediction)
                except Exception as e:
                    logger.error(f"Prediction callback failed: {e}")
                    self._on_error("prediction_callback", e)
            
            logger.debug(f"Processed prediction: {prediction.predicted_direction}")
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"Prediction processing failed: {e}")
            self._on_error("prediction_processing", e)
    
    def _on_error(self, component: str, error: Exception):
        """Handle system errors."""
        self.total_errors += 1
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(component, error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    async def start(self):
        """Start the real-time prediction system."""
        if self.is_running:
            logger.warning("Predictor already running")
            return
        
        try:
            # Initialize if not already done
            if self.model_server is None or self.prediction_engine is None:
                await self.initialize()
            
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info(f"Starting real-time predictor for {self.config.symbol}")
            
            # Start prediction engine
            await self.prediction_engine.start(self.config.data_source_url)
            
            # Start monitoring if enabled
            if self.config.enable_monitoring:
                asyncio.create_task(self._monitoring_loop())
            
            logger.info("Real-time predictor started successfully")
            
        except Exception as e:
            self.is_running = False
            logger.error(f"Failed to start predictor: {e}")
            raise
    
    async def stop(self):
        """Stop the real-time prediction system."""
        if not self.is_running:
            logger.warning("Predictor not running")
            return
        
        logger.info("Stopping real-time predictor...")
        self.is_running = False
        
        try:
            # Stop prediction engine
            if self.prediction_engine:
                await self.prediction_engine.stop()
            
            # Shutdown model server
            if self.model_server:
                self.model_server.shutdown()
            
            logger.info("Real-time predictor stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _monitoring_loop(self):
        """Continuous system monitoring."""
        while self.is_running:
            try:
                # Perform health check
                health = await self.health_check()
                
                # Notify health callbacks
                for callback in self.health_callbacks:
                    try:
                        callback(health)
                    except Exception as e:
                        logger.error(f"Health callback failed: {e}")
                
                # Log health status
                if health.status != 'healthy':
                    logger.warning(f"System health: {health.status} - Issues: {health.issues}")
                
                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def health_check(self) -> SystemHealth:
        """Perform comprehensive system health check."""
        try:
            self.last_health_check = datetime.now()
            issues = []
            components = {}
            performance_metrics = {}
            
            # Check model server
            if self.model_server:
                try:
                    model_health = self.model_server.health_check()
                    components['model_server'] = model_health['status'] == 'healthy'
                    if model_health['status'] != 'healthy':
                        issues.append(f"Model server: {model_health.get('error', 'unhealthy')}")
                    performance_metrics['model_inference_time'] = model_health.get('health_check_time_ms', 0)
                except Exception as e:
                    components['model_server'] = False
                    issues.append(f"Model server check failed: {e}")
            else:
                components['model_server'] = False
                issues.append("Model server not initialized")
            
            # Check prediction engine
            if self.prediction_engine:
                try:
                    engine_stats = self.prediction_engine.get_prediction_stats()
                    components['prediction_engine'] = engine_stats['is_running']
                    if not engine_stats['is_running']:
                        issues.append("Prediction engine not running")
                    
                    performance_metrics.update({
                        'prediction_rate': engine_stats['prediction_rate_per_second'],
                        'cache_hit_rate': engine_stats['cache_hit_rate'],
                        'avg_prediction_time': engine_stats['average_prediction_time_ms']
                    })
                except Exception as e:
                    components['prediction_engine'] = False
                    issues.append(f"Prediction engine check failed: {e}")
            else:
                components['prediction_engine'] = False
                issues.append("Prediction engine not initialized")
            
            # Check data streaming
            if self.prediction_engine and self.prediction_engine.data_streamer:
                try:
                    stream_stats = self.prediction_engine.data_streamer.get_streaming_stats()
                    components['data_streamer'] = stream_stats['is_streaming']
                    if not stream_stats['is_streaming']:
                        issues.append("Data streamer not active")
                    
                    performance_metrics.update({
                        'data_rate': stream_stats['data_rate_per_second'],
                        'data_error_rate': stream_stats['error_rate']
                    })
                except Exception as e:
                    components['data_streamer'] = False
                    issues.append(f"Data streamer check failed: {e}")
            else:
                components['data_streamer'] = False
                issues.append("Data streamer not available")
            
            # Determine overall health status
            if all(components.values()):
                status = 'healthy'
            elif any(components.values()):
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            # Add performance-based issues
            if performance_metrics.get('prediction_rate', 0) < 0.1:
                issues.append("Low prediction rate")
            
            if performance_metrics.get('data_rate', 0) < 0.1:
                issues.append("Low data rate")
            
            return SystemHealth(
                status=status,
                components=components,
                performance_metrics=performance_metrics,
                last_check=self.last_health_check,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return SystemHealth(
                status='unhealthy',
                components={},
                performance_metrics={},
                last_check=datetime.now(),
                issues=[f"Health check error: {e}"]
            )
    
    def subscribe_to_predictions(self, callback: Callable[[PredictionResult], None]):
        """Subscribe to prediction updates."""
        self.prediction_callbacks.append(callback)
        logger.info(f"Added prediction subscriber, total: {len(self.prediction_callbacks)}")
    
    def subscribe_to_health(self, callback: Callable[[SystemHealth], None]):
        """Subscribe to health status updates."""
        self.health_callbacks.append(callback)
        logger.info(f"Added health subscriber, total: {len(self.health_callbacks)}")
    
    def subscribe_to_errors(self, callback: Callable[[str, Exception], None]):
        """Subscribe to error notifications."""
        self.error_callbacks.append(callback)
        logger.info(f"Added error subscriber, total: {len(self.error_callbacks)}")
    
    def get_recent_predictions(self, count: int = 10) -> List[PredictionResult]:
        """Get recent predictions."""
        if self.prediction_engine:
            return self.prediction_engine.get_recent_predictions(count)
        return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'config': asdict(self.config),
            'system_status': {
                'is_running': self.is_running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'total_predictions': self.total_predictions,
                'total_errors': self.total_errors,
                'error_rate': self.total_errors / max(self.total_predictions, 1)
            }
        }
        
        # Add component stats
        if self.model_server:
            stats['model_server'] = self.model_server.get_performance_stats()
        
        if self.prediction_engine:
            stats['prediction_engine'] = self.prediction_engine.get_performance_summary()
        
        # Add recent health check
        if self.last_health_check:
            asyncio.create_task(self._add_health_to_stats(stats))
        
        return stats
    
    async def _add_health_to_stats(self, stats: Dict[str, Any]):
        """Add health check to stats (async helper)."""
        try:
            health = await self.health_check()
            stats['health'] = asdict(health)
        except Exception as e:
            stats['health'] = {'error': str(e)}
    
    def export_performance_data(self, filepath: str):
        """Export performance data to file."""
        stats = self.get_system_stats()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Exported performance data to {filepath}")


# Example usage and testing
async def demo_real_time_predictor():
    """Demonstrate the real-time predictor system."""
    
    # Configure predictor
    config = PredictorConfig(
        symbol='BTCUSD',
        data_source_type='synthetic',
        update_frequency=5.0,  # 5 data updates per second
        prediction_interval=2.0,  # 1 prediction every 2 seconds
        enable_monitoring=True,
        log_level='INFO'
    )
    
    # Create predictor
    predictor = RealTimePredictor(config)
    
    # Set up callbacks
    def on_prediction(prediction: PredictionResult):
        print(f"🎯 Prediction: {prediction.predicted_direction} "
              f"(confidence: {prediction.confidence:.3f}) "
              f"at {prediction.timestamp.strftime('%H:%M:%S')}")
    
    def on_health(health: SystemHealth):
        if health.status != 'healthy':
            print(f"⚠️  Health: {health.status} - Issues: {health.issues}")
    
    def on_error(component: str, error: Exception):
        print(f"❌ Error in {component}: {error}")
    
    # Subscribe to events
    predictor.subscribe_to_predictions(on_prediction)
    predictor.subscribe_to_health(on_health)
    predictor.subscribe_to_errors(on_error)
    
    try:
        # Start the system
        await predictor.start()
        
        print(f"🚀 Real-time predictor started for {config.symbol}")
        print("   Press Ctrl+C to stop...")
        
        # Let it run for a while
        await asyncio.sleep(30)
        
        # Get final stats
        stats = predictor.get_system_stats()
        print(f"\n📊 Final Statistics:")
        print(f"   Total Predictions: {stats['system_status']['total_predictions']}")
        print(f"   Error Rate: {stats['system_status']['error_rate']:.2%}")
        print(f"   Uptime: {stats['system_status']['uptime_seconds']:.1f} seconds")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping predictor...")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        await predictor.stop()
        print("✅ Predictor stopped")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demo_real_time_predictor())