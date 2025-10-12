"""
Trading Strategy Framework for Backtesting

This module provides base classes and implementations for trading strategies
that convert model predictions into trading signals for backtesting.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import logging

from src.backtesting.engine import MarketData, OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal representation."""
    timestamp: datetime
    symbol: str
    signal_strength: float  # -1 to 1, where -1 is strong sell, 1 is strong buy
    confidence: float  # 0 to 1
    prediction: float  # Raw model prediction
    features: Optional[Dict[str, float]] = None
    
    @property
    def is_buy_signal(self) -> bool:
        """Check if this is a buy signal."""
        return self.signal_strength > 0
    
    @property
    def is_sell_signal(self) -> bool:
        """Check if this is a sell signal."""
        return self.signal_strength < 0
    
    @property
    def is_hold_signal(self) -> bool:
        """Check if this is a hold signal."""
        return self.signal_strength == 0


class TradingStrategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    
    def __init__(self, 
                 name: str,
                 min_signal_strength: float = 0.1,
                 min_confidence: float = 0.5,
                 max_position_size: float = 1.0):
        """
        Initialize trading strategy.
        
        Args:
            name: Strategy name
            min_signal_strength: Minimum signal strength to trade
            min_confidence: Minimum confidence to trade
            max_position_size: Maximum position size (as fraction of portfolio)
        """
        self.name = name
        self.min_signal_strength = min_signal_strength
        self.min_confidence = min_confidence
        self.max_position_size = max_position_size
        
        # State tracking
        self.current_positions = {}
        self.signal_history = []
        self.performance_history = []
        
        logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def generate_signal(self, 
                       market_data: MarketData,
                       model_prediction: Optional[float] = None,
                       model_confidence: Optional[float] = None) -> Signal:
        """
        Generate trading signal from market data and model prediction.
        
        Args:
            market_data: Current market data
            model_prediction: Model prediction value
            model_confidence: Model confidence score
            
        Returns:
            Trading signal
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, 
                               signal: Signal,
                               portfolio_value: float,
                               current_position: float = 0.0) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_position: Current position size
            
        Returns:
            Position size to trade
        """
        pass
    
    def should_trade(self, signal: Signal) -> bool:
        """
        Determine if we should trade based on signal criteria.
        
        Args:
            signal: Trading signal
            
        Returns:
            True if should trade
        """
        return (abs(signal.signal_strength) >= self.min_signal_strength and 
                signal.confidence >= self.min_confidence)
    
    def update_position(self, symbol: str, position_change: float):
        """Update current position tracking."""
        if symbol not in self.current_positions:
            self.current_positions[symbol] = 0.0
        self.current_positions[symbol] += position_change
    
    def get_current_position(self, symbol: str) -> float:
        """Get current position for symbol."""
        return self.current_positions.get(symbol, 0.0)


class PredictionStrategy(TradingStrategy):
    """
    Strategy that trades based on raw model predictions.
    """
    
    def __init__(self,
                 name: str = "PredictionStrategy",
                 prediction_threshold: float = 0.001,
                 position_scaling: float = 1000.0,
                 **kwargs):
        """
        Initialize prediction-based strategy.
        
        Args:
            prediction_threshold: Minimum prediction magnitude to trade
            position_scaling: Scaling factor for position size
        """
        super().__init__(name, **kwargs)
        self.prediction_threshold = prediction_threshold
        self.position_scaling = position_scaling
    
    def generate_signal(self, 
                       market_data: MarketData,
                       model_prediction: Optional[float] = None,
                       model_confidence: Optional[float] = None) -> Signal:
        """Generate signal based on raw prediction."""
        if model_prediction is None:
            model_prediction = 0.0
        if model_confidence is None:
            model_confidence = 0.5
        
        # Convert prediction to signal strength
        if abs(model_prediction) >= self.prediction_threshold:
            # Normalize prediction to [-1, 1] range
            signal_strength = np.tanh(model_prediction / self.prediction_threshold)
        else:
            signal_strength = 0.0
        
        return Signal(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            signal_strength=signal_strength,
            confidence=model_confidence,
            prediction=model_prediction,
            features={'spread_bps': market_data.spread_bps}
        )
    
    def calculate_position_size(self, 
                               signal: Signal,
                               portfolio_value: float,
                               current_position: float = 0.0) -> float:
        """Calculate position size based on signal strength and confidence."""
        base_size = (signal.signal_strength * signal.confidence * 
                    self.position_scaling * self.max_position_size)
        
        # Adjust for current position
        target_position = base_size
        position_change = target_position - current_position
        
        return position_change


class MomentumStrategy(TradingStrategy):
    """
    Momentum-based strategy that trades in direction of recent price moves.
    """
    
    def __init__(self,
                 name: str = "MomentumStrategy", 
                 lookback_periods: int = 5,
                 momentum_threshold: float = 0.001,
                 **kwargs):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_periods: Number of periods to look back for momentum
            momentum_threshold: Minimum momentum to generate signal
        """
        super().__init__(name, **kwargs)
        self.lookback_periods = lookback_periods
        self.momentum_threshold = momentum_threshold
        self.price_history = []
    
    def generate_signal(self, 
                       market_data: MarketData,
                       model_prediction: Optional[float] = None,
                       model_confidence: Optional[float] = None) -> Signal:
        """Generate signal based on price momentum."""
        # Update price history
        self.price_history.append(market_data.mid_price)
        if len(self.price_history) > self.lookback_periods:
            self.price_history.pop(0)
        
        # Calculate momentum
        if len(self.price_history) >= 2:
            momentum = (self.price_history[-1] - self.price_history[0]) / self.price_history[0]
        else:
            momentum = 0.0
        
        # Generate signal
        if abs(momentum) >= self.momentum_threshold:
            signal_strength = np.sign(momentum) * min(1.0, abs(momentum) / self.momentum_threshold)
        else:
            signal_strength = 0.0
        
        # Use model confidence if available, otherwise use momentum magnitude
        confidence = model_confidence if model_confidence is not None else min(1.0, abs(momentum) * 10)
        
        return Signal(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            signal_strength=signal_strength,
            confidence=confidence,
            prediction=model_prediction or momentum,
            features={'momentum': momentum, 'lookback': len(self.price_history)}
        )
    
    def calculate_position_size(self, 
                               signal: Signal,
                               portfolio_value: float,
                               current_position: float = 0.0) -> float:
        """Calculate position size based on momentum strength."""
        # Fixed position size based on signal
        base_size = signal.signal_strength * 1000 * self.max_position_size
        position_change = base_size - current_position
        
        return position_change


class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion strategy that trades against recent price moves.
    """
    
    def __init__(self,
                 name: str = "MeanReversionStrategy",
                 lookback_periods: int = 20,
                 reversion_threshold: float = 2.0,  # Standard deviations
                 **kwargs):
        """
        Initialize mean reversion strategy.
        
        Args:
            lookback_periods: Number of periods for mean calculation
            reversion_threshold: Threshold in standard deviations
        """
        super().__init__(name, **kwargs)
        self.lookback_periods = lookback_periods
        self.reversion_threshold = reversion_threshold
        self.price_history = []
    
    def generate_signal(self, 
                       market_data: MarketData,
                       model_prediction: Optional[float] = None,
                       model_confidence: Optional[float] = None) -> Signal:
        """Generate signal based on mean reversion."""
        # Update price history
        self.price_history.append(market_data.mid_price)
        if len(self.price_history) > self.lookback_periods:
            self.price_history.pop(0)
        
        # Calculate mean and standard deviation
        if len(self.price_history) >= 3:
            mean_price = np.mean(self.price_history)
            std_price = np.std(self.price_history)
            
            if std_price > 0:
                z_score = (market_data.mid_price - mean_price) / std_price
            else:
                z_score = 0.0
        else:
            z_score = 0.0
        
        # Generate mean reversion signal (opposite of z-score)
        if abs(z_score) >= self.reversion_threshold:
            signal_strength = -np.sign(z_score) * min(1.0, abs(z_score) / self.reversion_threshold)
        else:
            signal_strength = 0.0
        
        confidence = model_confidence if model_confidence is not None else min(1.0, abs(z_score) / 3.0)
        
        return Signal(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            signal_strength=signal_strength,
            confidence=confidence,
            prediction=model_prediction or z_score,
            features={'z_score': z_score, 'mean_price': mean_price if len(self.price_history) >= 3 else 0}
        )
    
    def calculate_position_size(self, 
                               signal: Signal,
                               portfolio_value: float,
                               current_position: float = 0.0) -> float:
        """Calculate position size based on reversion signal."""
        base_size = signal.signal_strength * 1000 * self.max_position_size
        position_change = base_size - current_position
        
        return position_change


class MLStrategy(TradingStrategy):
    """
    Machine learning strategy that uses model predictions with additional features.
    """
    
    def __init__(self,
                 name: str = "MLStrategy",
                 model: Optional[torch.nn.Module] = None,
                 feature_scaler: Optional[Any] = None,
                 signal_transformer: Optional[Callable] = None,
                 **kwargs):
        """
        Initialize ML strategy.
        
        Args:
            model: Trained ML model
            feature_scaler: Feature scaler (from training)
            signal_transformer: Function to transform model output to signal
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.feature_scaler = feature_scaler
        self.signal_transformer = signal_transformer or self._default_signal_transformer
        
        # Model state
        if self.model:
            self.model.eval()
    
    def _default_signal_transformer(self, prediction: float, confidence: float) -> Tuple[float, float]:
        """Default transformation from model output to trading signal."""
        # Simple transformation - can be made more sophisticated
        signal_strength = np.tanh(prediction * 10)  # Scale and bound to [-1, 1]
        return signal_strength, confidence
    
    def generate_signal(self, 
                       market_data: MarketData,
                       model_prediction: Optional[float] = None,
                       model_confidence: Optional[float] = None) -> Signal:
        """Generate signal using ML model."""
        
        # If model is provided and features are available, use model prediction
        if self.model is not None and market_data.features is not None:
            try:
                with torch.no_grad():
                    features = torch.tensor(market_data.features, dtype=torch.float32)
                    if len(features.shape) == 1:
                        features = features.unsqueeze(0)  # Add batch dimension
                    
                    output = self.model(features)
                    
                    if isinstance(output, dict):
                        prediction = output['predictions'].item()
                        confidence = output.get('confidence', torch.tensor([0.5])).item()
                    else:
                        prediction = output.item()
                        confidence = 0.5
                
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                prediction = model_prediction or 0.0
                confidence = model_confidence or 0.5
        else:
            prediction = model_prediction or 0.0
            confidence = model_confidence or 0.5
        
        # Transform to signal
        signal_strength, final_confidence = self.signal_transformer(prediction, confidence)
        
        return Signal(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            signal_strength=signal_strength,
            confidence=final_confidence,
            prediction=prediction,
            features={
                'spread_bps': market_data.spread_bps,
                'volume': market_data.volume,
                'bid_ask_ratio': market_data.bid_size / (market_data.ask_size + 1e-8)
            }
        )
    
    def calculate_position_size(self, 
                               signal: Signal,
                               portfolio_value: float,
                               current_position: float = 0.0) -> float:
        """Calculate position size using Kelly criterion approximation."""
        # Simple Kelly-like position sizing
        win_prob = (signal.confidence + 1) / 2  # Convert confidence to win probability
        
        if win_prob > 0.5:
            # Kelly fraction approximation
            edge = 2 * win_prob - 1
            kelly_fraction = edge * signal.confidence
            
            # Scale by signal strength and apply maximum position constraint
            target_fraction = kelly_fraction * abs(signal.signal_strength) * self.max_position_size
            target_fraction = np.clip(target_fraction, -self.max_position_size, self.max_position_size)
            
            target_size = np.sign(signal.signal_strength) * target_fraction * portfolio_value / 100  # Assuming price ~100
            position_change = target_size - current_position
            
            return position_change
        else:
            # If low confidence, reduce position
            return -current_position * 0.5


class EnsembleStrategy(TradingStrategy):
    """
    Ensemble strategy that combines multiple strategies.
    """
    
    def __init__(self,
                 name: str = "EnsembleStrategy",
                 strategies: List[TradingStrategy] = None,
                 weights: Optional[List[float]] = None,
                 **kwargs):
        """
        Initialize ensemble strategy.
        
        Args:
            strategies: List of strategies to combine
            weights: Weights for each strategy (equal weights if None)
        """
        super().__init__(name, **kwargs)
        self.strategies = strategies or []
        
        if weights is None:
            self.weights = [1.0 / len(self.strategies)] * len(self.strategies)
        else:
            self.weights = weights
            
        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Initialized ensemble with {len(self.strategies)} strategies")
    
    def generate_signal(self, 
                       market_data: MarketData,
                       model_prediction: Optional[float] = None,
                       model_confidence: Optional[float] = None) -> Signal:
        """Generate ensemble signal by combining individual strategy signals."""
        if not self.strategies:
            return Signal(
                timestamp=market_data.timestamp,
                symbol=market_data.symbol,
                signal_strength=0.0,
                confidence=0.0,
                prediction=0.0
            )
        
        # Get signals from all strategies
        signals = []
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(market_data, model_prediction, model_confidence)
                signals.append(signal)
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")
                # Create neutral signal for failed strategy
                signals.append(Signal(
                    timestamp=market_data.timestamp,
                    symbol=market_data.symbol,
                    signal_strength=0.0,
                    confidence=0.0,
                    prediction=0.0
                ))
        
        # Combine signals using weights
        combined_strength = sum(s.signal_strength * w for s, w in zip(signals, self.weights))
        combined_confidence = sum(s.confidence * w for s, w in zip(signals, self.weights))
        combined_prediction = sum(s.prediction * w for s, w in zip(signals, self.weights))
        
        # Combine features
        combined_features = {}
        for signal in signals:
            if signal.features:
                for key, value in signal.features.items():
                    if key not in combined_features:
                        combined_features[key] = []
                    combined_features[key].append(value)
        
        # Average feature values
        for key, values in combined_features.items():
            combined_features[key] = np.mean(values)
        
        return Signal(
            timestamp=market_data.timestamp,
            symbol=market_data.symbol,
            signal_strength=combined_strength,
            confidence=combined_confidence,
            prediction=combined_prediction,
            features=combined_features
        )
    
    def calculate_position_size(self, 
                               signal: Signal,
                               portfolio_value: float,
                               current_position: float = 0.0) -> float:
        """Calculate ensemble position size."""
        # Use first strategy's position sizing as default
        if self.strategies:
            return self.strategies[0].calculate_position_size(signal, portfolio_value, current_position)
        else:
            return 0.0


if __name__ == "__main__":
    # Test the strategy framework
    print("Testing Trading Strategy Framework...")
    
    # Create sample market data
    market_data = MarketData(
        timestamp=datetime.now(),
        symbol="TEST",
        bid_price=99.95,
        ask_price=100.05,
        bid_size=1000,
        ask_size=1000,
        last_price=100.00,
        volume=10000,
        features=np.random.randn(46)
    )
    
    # Test prediction strategy
    pred_strategy = PredictionStrategy()
    signal = pred_strategy.generate_signal(market_data, model_prediction=0.002, model_confidence=0.8)
    position_size = pred_strategy.calculate_position_size(signal, 100000, 0)
    
    print(f"Prediction strategy signal: {signal.signal_strength:.3f} (confidence: {signal.confidence:.3f})")
    print(f"Position size: {position_size:.2f}")
    
    # Test momentum strategy
    momentum_strategy = MomentumStrategy()
    for i in range(5):
        test_data = MarketData(
            timestamp=datetime.now(),
            symbol="TEST",
            bid_price=100 + i * 0.1,
            ask_price=100.1 + i * 0.1,
            bid_size=1000,
            ask_size=1000,
            last_price=100.05 + i * 0.1,
            volume=10000
        )
        signal = momentum_strategy.generate_signal(test_data)
    
    print(f"Momentum strategy signal: {signal.signal_strength:.3f}")
    
    # Test ensemble
    ensemble = EnsembleStrategy(strategies=[pred_strategy, momentum_strategy])
    ensemble_signal = ensemble.generate_signal(market_data, model_prediction=0.001, model_confidence=0.7)
    
    print(f"Ensemble signal: {ensemble_signal.signal_strength:.3f}")
    
    print("✅ Trading strategy framework test passed!")