"""
API Server for Real-time Predictions

Provides REST API endpoints for accessing real-time predictions
and system monitoring via HTTP interface.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Define dummy classes for when FastAPI is not available
    class BaseModel:
        pass
    class Field:
        def __init__(self, *args, **kwargs):
            pass

# Import inference components
from .real_time_predictor import RealTimePredictor, PredictorConfig
from .model_server import PredictionResult

logger = logging.getLogger(__name__)


# Pydantic models for API
class PredictionResponse(BaseModel):
    """API response model for predictions."""
    timestamp: str
    predicted_direction: int = Field(..., description="Direction: -1 (down), 0 (neutral), 1 (up)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    predicted_change: float = Field(..., description="Predicted price change")
    directional_probability: Dict[str, float] = Field(..., description="Probability distribution")
    model_name: str
    inference_time_ms: float


class SystemStatusResponse(BaseModel):
    """API response model for system status."""
    status: str = Field(..., description="System status: healthy, degraded, unhealthy")
    is_running: bool
    uptime_seconds: float
    total_predictions: int
    error_rate: float
    components: Dict[str, bool]
    performance_metrics: Dict[str, float]


class ConfigRequest(BaseModel):
    """API request model for configuration updates."""
    symbol: Optional[str] = None
    prediction_interval: Optional[float] = Field(None, gt=0.1, le=10.0)
    cache_size: Optional[int] = Field(None, gt=0, le=10000)
    enable_monitoring: Optional[bool] = None


class APIServer:
    """
    REST API server for real-time predictions.
    
    Provides HTTP endpoints for accessing predictions, system status,
    and configuration management.
    """
    
    def __init__(self, predictor: RealTimePredictor, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize API server.
        
        Args:
            predictor: Real-time predictor instance
            host: Server host address
            port: Server port
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for API server. Install with: pip install fastapi uvicorn")
        
        self.predictor = predictor
        self.host = host
        self.port = port
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Market Microstructure Real-time Predictions API",
            description="REST API for real-time market predictions and system monitoring",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes()
        
        # Server state
        self.server_start_time = datetime.now()
        self.request_count = 0
        
        logger.info(f"Initialized API server for {predictor.config.symbol}")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint with API information."""
            return {
                "service": "Market Microstructure Real-time Predictions API",
                "version": "1.0.0",
                "symbol": self.predictor.config.symbol,
                "status": "running" if self.predictor.is_running else "stopped",
                "docs": "/docs"
            }
        
        @self.app.get("/health", response_model=SystemStatusResponse)
        async def health_check():
            """Get system health status."""
            try:
                health = await self.predictor.health_check()
                stats = self.predictor.get_system_stats()
                
                return SystemStatusResponse(
                    status=health.status,
                    is_running=self.predictor.is_running,
                    uptime_seconds=stats['system_status']['uptime_seconds'],
                    total_predictions=stats['system_status']['total_predictions'],
                    error_rate=stats['system_status']['error_rate'],
                    components=health.components,
                    performance_metrics=health.performance_metrics
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail=f"Health check failed: {e}")
        
        @self.app.get("/predictions/latest", response_model=PredictionResponse)
        async def get_latest_prediction():
            """Get the most recent prediction."""
            try:
                predictions = self.predictor.get_recent_predictions(1)
                if not predictions:
                    raise HTTPException(status_code=404, detail="No predictions available")
                
                prediction = predictions[0]
                return PredictionResponse(
                    timestamp=prediction.timestamp.isoformat(),
                    predicted_direction=prediction.predicted_direction,
                    confidence=prediction.confidence,
                    predicted_change=prediction.predicted_change,
                    directional_probability=prediction.directional_probability,
                    model_name=prediction.model_name,
                    inference_time_ms=prediction.inference_time_ms
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get latest prediction: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get prediction: {e}")
        
        @self.app.get("/predictions/recent", response_model=List[PredictionResponse])
        async def get_recent_predictions(count: int = 10):
            """Get recent predictions."""
            try:
                if count < 1 or count > 100:
                    raise HTTPException(status_code=400, detail="Count must be between 1 and 100")
                
                predictions = self.predictor.get_recent_predictions(count)
                
                return [
                    PredictionResponse(
                        timestamp=pred.timestamp.isoformat(),
                        predicted_direction=pred.predicted_direction,
                        confidence=pred.confidence,
                        predicted_change=pred.predicted_change,
                        directional_probability=pred.directional_probability,
                        model_name=pred.model_name,
                        inference_time_ms=pred.inference_time_ms
                    )
                    for pred in predictions
                ]
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get recent predictions: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get predictions: {e}")
        
        @self.app.get("/stats", response_model=Dict[str, Any])
        async def get_system_stats():
            """Get comprehensive system statistics."""
            try:
                self.request_count += 1
                stats = self.predictor.get_system_stats()
                
                # Add API server stats
                stats['api_server'] = {
                    'start_time': self.server_start_time.isoformat(),
                    'request_count': self.request_count,
                    'host': self.host,
                    'port': self.port
                }
                
                return stats
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")
        
        @self.app.post("/start")
        async def start_predictor():
            """Start the prediction system."""
            try:
                if self.predictor.is_running:
                    return {"message": "Predictor already running", "status": "running"}
                
                await self.predictor.start()
                return {"message": "Predictor started successfully", "status": "running"}
            except Exception as e:
                logger.error(f"Failed to start predictor: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start predictor: {e}")
        
        @self.app.post("/stop")
        async def stop_predictor():
            """Stop the prediction system."""
            try:
                if not self.predictor.is_running:
                    return {"message": "Predictor already stopped", "status": "stopped"}
                
                await self.predictor.stop()
                return {"message": "Predictor stopped successfully", "status": "stopped"}
            except Exception as e:
                logger.error(f"Failed to stop predictor: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to stop predictor: {e}")
        
        @self.app.post("/config/update")
        async def update_config(config: ConfigRequest):
            """Update system configuration."""
            try:
                # Note: This is a simplified implementation
                # In a full system, you'd need to restart components with new config
                updates = {}
                if config.symbol:
                    updates['symbol'] = config.symbol
                if config.prediction_interval:
                    updates['prediction_interval'] = config.prediction_interval
                if config.cache_size:
                    updates['cache_size'] = config.cache_size
                if config.enable_monitoring is not None:
                    updates['enable_monitoring'] = config.enable_monitoring
                
                return {
                    "message": "Configuration update requested",
                    "updates": updates,
                    "note": "Restart required for changes to take effect"
                }
            except Exception as e:
                logger.error(f"Failed to update config: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update config: {e}")
        
        @self.app.post("/cache/clear")
        async def clear_cache():
            """Clear prediction cache."""
            try:
                if self.predictor.prediction_engine:
                    self.predictor.prediction_engine.clear_cache()
                return {"message": "Cache cleared successfully"}
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")
        
        @self.app.get("/model/info", response_model=Dict[str, Any])
        async def get_model_info():
            """Get model information."""
            try:
                if not self.predictor.model_server:
                    raise HTTPException(status_code=503, detail="Model server not available")
                
                return self.predictor.model_server.get_model_info()
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get model info: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get model info: {e}")
        
        # WebSocket endpoint for real-time predictions (if needed)
        @self.app.websocket("/ws/predictions")
        async def websocket_predictions(websocket):
            """WebSocket endpoint for real-time prediction streaming."""
            await websocket.accept()
            
            def send_prediction(prediction: PredictionResult):
                try:
                    asyncio.create_task(websocket.send_json({
                        "type": "prediction",
                        "data": {
                            "timestamp": prediction.timestamp.isoformat(),
                            "predicted_direction": prediction.predicted_direction,
                            "confidence": prediction.confidence,
                            "predicted_change": prediction.predicted_change,
                            "directional_probability": prediction.directional_probability,
                            "model_name": prediction.model_name,
                            "inference_time_ms": prediction.inference_time_ms
                        }
                    }))
                except Exception as e:
                    logger.error(f"WebSocket send failed: {e}")
            
            # Subscribe to predictions
            self.predictor.subscribe_to_predictions(send_prediction)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except Exception as e:
                logger.info(f"WebSocket disconnected: {e}")
            finally:
                # Unsubscribe when client disconnects
                if send_prediction in self.predictor.prediction_callbacks:
                    self.predictor.prediction_callbacks.remove(send_prediction)
    
    async def start_server(self):
        """Start the API server."""
        logger.info(f"Starting API server on {self.host}:{self.port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def run(self):
        """Run the API server (blocking)."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# Convenience functions for creating and running API server
def create_api_server(predictor_config: PredictorConfig, 
                     host: str = "0.0.0.0", 
                     port: int = 8000) -> APIServer:
    """Create an API server with predictor."""
    predictor = RealTimePredictor(predictor_config)
    return APIServer(predictor, host, port)


async def run_api_server_async(predictor_config: PredictorConfig, 
                              host: str = "0.0.0.0", 
                              port: int = 8000):
    """Run API server asynchronously."""
    predictor = RealTimePredictor(predictor_config)
    api_server = APIServer(predictor, host, port)
    
    # Start predictor first
    await predictor.start()
    
    try:
        # Start API server
        await api_server.start_server()
    finally:
        # Cleanup
        await predictor.stop()


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = PredictorConfig(
        symbol='BTCUSD',
        data_source_type='synthetic',
        update_frequency=2.0,
        prediction_interval=1.0,
        enable_monitoring=True
    )
    
    # Option 1: Run with asyncio
    async def demo():
        await run_api_server_async(config, host="localhost", port=8000)
    
    # Option 2: Create and run directly
    def demo_sync():
        predictor = RealTimePredictor(config)
        api_server = APIServer(predictor, host="localhost", port=8000)
        
        print(f"Starting API server at http://localhost:8000")
        print("API documentation available at: http://localhost:8000/docs")
        print("Press Ctrl+C to stop...")
        
        api_server.run()
    
    # Run demo
    try:
        demo_sync()
    except KeyboardInterrupt:
        print("\nAPI server stopped")