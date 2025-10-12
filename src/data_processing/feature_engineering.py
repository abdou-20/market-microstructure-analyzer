"""
Feature Engineering Module

This module extracts microstructure features from order book data including:
- Order Flow Imbalance (OFI)
- Bid-ask spread features
- Volume clustering metrics
- Microstructure indicators
- Price and volume features
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from .order_book_parser import OrderBookSnapshot

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Container for extracted features from order book data."""
    timestamp: datetime
    symbol: str
    features: Dict[str, float]
    target: Optional[float] = None  # For supervised learning
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'target': self.target
        }
        result.update(self.features)
        return result


class FeatureEngineering:
    """
    Main feature engineering class for extracting microstructure features
    from order book snapshots.
    """
    
    def __init__(self,
                 lookback_window: int = 10,
                 prediction_horizon: int = 5,
                 max_levels: int = 10):
        """
        Initialize feature engineering.
        
        Args:
            lookback_window: Number of snapshots to look back for features
            prediction_horizon: Number of snapshots ahead for target calculation
            max_levels: Maximum number of order book levels to consider
        """
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.max_levels = max_levels
        self.snapshot_buffer: List[OrderBookSnapshot] = []
        
    def extract_features(self, snapshots: List[OrderBookSnapshot]) -> List[FeatureVector]:
        """
        Extract features from a sequence of order book snapshots.
        
        Args:
            snapshots: List of OrderBookSnapshot objects
            
        Returns:
            List of FeatureVector objects
        """
        features = []
        
        for i in range(self.lookback_window, len(snapshots) - self.prediction_horizon):
            current_snapshot = snapshots[i]
            historical_snapshots = snapshots[i - self.lookback_window:i + 1]
            
            # Calculate target (future price movement)
            future_snapshot = snapshots[i + self.prediction_horizon]
            target = self._calculate_target(current_snapshot, future_snapshot)
            
            # Extract all features
            feature_dict = {}
            
            # Basic order book features
            feature_dict.update(self._extract_basic_features(current_snapshot))
            
            # Order flow imbalance features
            feature_dict.update(self._extract_ofi_features(historical_snapshots))
            
            # Spread features
            feature_dict.update(self._extract_spread_features(current_snapshot, historical_snapshots))
            
            # Volume features
            feature_dict.update(self._extract_volume_features(current_snapshot, historical_snapshots))
            
            # Price features
            feature_dict.update(self._extract_price_features(historical_snapshots))
            
            # Microstructure features
            feature_dict.update(self._extract_microstructure_features(current_snapshot, historical_snapshots))
            
            # Technical indicators
            feature_dict.update(self._extract_technical_features(historical_snapshots))
            
            feature_vector = FeatureVector(
                timestamp=current_snapshot.timestamp,
                symbol=current_snapshot.symbol,
                features=feature_dict,
                target=target
            )
            
            features.append(feature_vector)
        
        return features
    
    def _calculate_target(self, current: OrderBookSnapshot, future: OrderBookSnapshot) -> float:
        """
        Calculate target variable (future price movement).
        
        Args:
            current: Current order book snapshot
            future: Future order book snapshot
            
        Returns:
            Price movement as percentage change
        """
        if current.mid_price == 0:
            return 0.0
        
        return (future.mid_price - current.mid_price) / current.mid_price
    
    def _extract_basic_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Extract basic order book features."""
        features = {}
        
        # Basic price and spread features
        features['mid_price'] = snapshot.mid_price
        features['best_bid'] = snapshot.best_bid
        features['best_ask'] = snapshot.best_ask
        features['spread_absolute'] = snapshot.spread
        features['spread_relative'] = snapshot.spread / snapshot.mid_price if snapshot.mid_price > 0 else 0
        
        # Bid and ask quantities at best levels
        if snapshot.bids and snapshot.asks:
            features['best_bid_qty'] = snapshot.bids[0][1]
            features['best_ask_qty'] = snapshot.asks[0][1]
            features['bid_ask_qty_ratio'] = snapshot.bids[0][1] / snapshot.asks[0][1] if snapshot.asks[0][1] > 0 else 0
        else:
            features['best_bid_qty'] = 0.0
            features['best_ask_qty'] = 0.0
            features['bid_ask_qty_ratio'] = 0.0
        
        # Total volume at all levels
        total_bid_volume = sum(qty for _, qty in snapshot.bids)
        total_ask_volume = sum(qty for _, qty in snapshot.asks)
        
        features['total_bid_volume'] = total_bid_volume
        features['total_ask_volume'] = total_ask_volume
        features['volume_imbalance'] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
        
        return features
    
    def _extract_ofi_features(self, snapshots: List[OrderBookSnapshot]) -> Dict[str, float]:
        """
        Extract Order Flow Imbalance (OFI) features.
        
        OFI = Σ(bid_volume_changes) - Σ(ask_volume_changes)
        """
        features = {}
        
        if len(snapshots) < 2:
            return {'ofi_1': 0.0, 'ofi_5': 0.0, 'ofi_10': 0.0}
        
        # Calculate OFI for different horizons
        for horizon in [1, 5, min(10, len(snapshots) - 1)]:
            if horizon >= len(snapshots):
                features[f'ofi_{horizon}'] = 0.0
                continue
                
            current = snapshots[-1]
            previous = snapshots[-1 - horizon]
            
            # Create price-to-volume mappings
            current_bids = {price: qty for price, qty in current.bids}
            current_asks = {price: qty for price, qty in current.asks}
            previous_bids = {price: qty for price, qty in previous.bids}
            previous_asks = {price: qty for price, qty in previous.asks}
            
            # Calculate bid volume changes
            bid_volume_change = 0.0
            all_bid_prices = set(current_bids.keys()) | set(previous_bids.keys())
            for price in all_bid_prices:
                current_qty = current_bids.get(price, 0.0)
                previous_qty = previous_bids.get(price, 0.0)
                bid_volume_change += current_qty - previous_qty
            
            # Calculate ask volume changes
            ask_volume_change = 0.0
            all_ask_prices = set(current_asks.keys()) | set(previous_asks.keys())
            for price in all_ask_prices:
                current_qty = current_asks.get(price, 0.0)
                previous_qty = previous_asks.get(price, 0.0)
                ask_volume_change += current_qty - previous_qty
            
            # OFI calculation
            ofi = bid_volume_change - ask_volume_change
            features[f'ofi_{horizon}'] = ofi
        
        return features
    
    def _extract_spread_features(self, current: OrderBookSnapshot, snapshots: List[OrderBookSnapshot]) -> Dict[str, float]:
        """Extract bid-ask spread related features."""
        features = {}
        
        # Current spread features
        features['spread_bps'] = (current.spread / current.mid_price) * 10000 if current.mid_price > 0 else 0
        
        # Historical spread statistics
        spreads = [s.spread for s in snapshots if s.spread > 0]
        relative_spreads = [s.spread / s.mid_price for s in snapshots if s.mid_price > 0]
        
        if spreads:
            features['spread_mean'] = np.mean(spreads)
            features['spread_std'] = np.std(spreads)
            features['spread_min'] = np.min(spreads)
            features['spread_max'] = np.max(spreads)
        else:
            features.update({'spread_mean': 0, 'spread_std': 0, 'spread_min': 0, 'spread_max': 0})
        
        if relative_spreads:
            features['relative_spread_mean'] = np.mean(relative_spreads)
            features['relative_spread_std'] = np.std(relative_spreads)
        else:
            features.update({'relative_spread_mean': 0, 'relative_spread_std': 0})
        
        # Effective spread (using mid-price)
        if len(snapshots) >= 2:
            price_changes = [snapshots[i].mid_price - snapshots[i-1].mid_price 
                           for i in range(1, len(snapshots)) 
                           if snapshots[i].mid_price > 0 and snapshots[i-1].mid_price > 0]
            if price_changes:
                features['effective_spread'] = 2 * np.mean(np.abs(price_changes))
            else:
                features['effective_spread'] = 0.0
        else:
            features['effective_spread'] = 0.0
        
        return features
    
    def _extract_volume_features(self, current: OrderBookSnapshot, snapshots: List[OrderBookSnapshot]) -> Dict[str, float]:
        """Extract volume-based features."""
        features = {}
        
        # Volume-weighted prices
        if current.bids and current.asks:
            # Volume-weighted bid price
            total_bid_value = sum(price * qty for price, qty in current.bids)
            total_bid_qty = sum(qty for _, qty in current.bids)
            vwap_bid = total_bid_value / total_bid_qty if total_bid_qty > 0 else 0
            
            # Volume-weighted ask price
            total_ask_value = sum(price * qty for price, qty in current.asks)
            total_ask_qty = sum(qty for _, qty in current.asks)
            vwap_ask = total_ask_value / total_ask_qty if total_ask_qty > 0 else 0
            
            features['vwap_bid'] = vwap_bid
            features['vwap_ask'] = vwap_ask
            features['vwap_mid'] = (vwap_bid + vwap_ask) / 2
        else:
            features.update({'vwap_bid': 0, 'vwap_ask': 0, 'vwap_mid': 0})
        
        # Volume clustering metrics
        volumes = []
        for snapshot in snapshots:
            total_volume = sum(qty for _, qty in snapshot.bids + snapshot.asks)
            volumes.append(total_volume)
        
        if volumes:
            features['volume_mean'] = np.mean(volumes)
            features['volume_std'] = np.std(volumes)
            features['volume_current_zscore'] = (volumes[-1] - np.mean(volumes)) / np.std(volumes) if np.std(volumes) > 0 else 0
        else:
            features.update({'volume_mean': 0, 'volume_std': 0, 'volume_current_zscore': 0})
        
        # Depth imbalance at different levels
        for level in range(1, min(6, len(current.bids), len(current.asks)) + 1):
            bid_volume = sum(qty for _, qty in current.bids[:level])
            ask_volume = sum(qty for _, qty in current.asks[:level])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
            else:
                imbalance = 0.0
            
            features[f'depth_imbalance_l{level}'] = imbalance
        
        return features
    
    def _extract_price_features(self, snapshots: List[OrderBookSnapshot]) -> Dict[str, float]:
        """Extract price-based features."""
        features = {}
        
        mid_prices = [s.mid_price for s in snapshots if s.mid_price > 0]
        
        if len(mid_prices) < 2:
            return {'price_return_1': 0, 'price_return_5': 0, 'price_volatility': 0, 'price_trend': 0}
        
        # Price returns
        if len(mid_prices) >= 2:
            features['price_return_1'] = (mid_prices[-1] - mid_prices[-2]) / mid_prices[-2]
        else:
            features['price_return_1'] = 0.0
        
        if len(mid_prices) >= 6:
            features['price_return_5'] = (mid_prices[-1] - mid_prices[-6]) / mid_prices[-6]
        else:
            features['price_return_5'] = 0.0
        
        # Price volatility (rolling standard deviation of returns)
        if len(mid_prices) >= 3:
            returns = [(mid_prices[i] - mid_prices[i-1]) / mid_prices[i-1] 
                      for i in range(1, len(mid_prices))]
            features['price_volatility'] = np.std(returns)
        else:
            features['price_volatility'] = 0.0
        
        # Price trend (linear regression slope)
        if len(mid_prices) >= 3:
            x = np.arange(len(mid_prices))
            slope = np.polyfit(x, mid_prices, 1)[0]
            features['price_trend'] = slope / mid_prices[-1] if mid_prices[-1] > 0 else 0
        else:
            features['price_trend'] = 0.0
        
        return features
    
    def _extract_microstructure_features(self, current: OrderBookSnapshot, snapshots: List[OrderBookSnapshot]) -> Dict[str, float]:
        """Extract additional microstructure indicators."""
        features = {}
        
        # Trade intensity (inverse of time between snapshots)
        if len(snapshots) >= 2:
            time_diffs = [(snapshots[i].timestamp - snapshots[i-1].timestamp).total_seconds() 
                         for i in range(1, len(snapshots))]
            avg_time_diff = np.mean(time_diffs) if time_diffs else 1.0
            features['trade_intensity'] = 1.0 / avg_time_diff if avg_time_diff > 0 else 0.0
        else:
            features['trade_intensity'] = 0.0
        
        # Arrival rates (changes in order book per second)
        if len(snapshots) >= 2:
            total_changes = 0
            total_time = 0
            
            for i in range(1, len(snapshots)):
                prev = snapshots[i-1]
                curr = snapshots[i]
                
                # Count changes in bid levels
                prev_bids = set((price, qty) for price, qty in prev.bids)
                curr_bids = set((price, qty) for price, qty in curr.bids)
                bid_changes = len(prev_bids.symmetric_difference(curr_bids))
                
                # Count changes in ask levels
                prev_asks = set((price, qty) for price, qty in prev.asks)
                curr_asks = set((price, qty) for price, qty in curr.asks)
                ask_changes = len(prev_asks.symmetric_difference(curr_asks))
                
                total_changes += bid_changes + ask_changes
                total_time += (curr.timestamp - prev.timestamp).total_seconds()
            
            features['arrival_rate'] = total_changes / total_time if total_time > 0 else 0.0
        else:
            features['arrival_rate'] = 0.0
        
        # Resilience (how quickly spread returns to normal after widening)
        spreads = [s.spread for s in snapshots]
        if len(spreads) >= 3:
            spread_changes = [spreads[i] - spreads[i-1] for i in range(1, len(spreads))]
            features['spread_resilience'] = -np.corrcoef(spread_changes[:-1], spread_changes[1:])[0, 1] if len(spread_changes) > 1 else 0
        else:
            features['spread_resilience'] = 0.0
        
        return features
    
    def _extract_technical_features(self, snapshots: List[OrderBookSnapshot]) -> Dict[str, float]:
        """Extract technical analysis features."""
        features = {}
        
        mid_prices = [s.mid_price for s in snapshots if s.mid_price > 0]
        
        if len(mid_prices) < 5:
            return {'sma_5': 0, 'rsi': 50, 'momentum': 0, 'price_acceleration': 0}
        
        # Simple moving average
        features['sma_5'] = np.mean(mid_prices[-5:])
        features['price_vs_sma'] = (mid_prices[-1] - features['sma_5']) / features['sma_5'] if features['sma_5'] > 0 else 0
        
        # RSI (Relative Strength Index)
        if len(mid_prices) >= 14:
            price_changes = [mid_prices[i] - mid_prices[i-1] for i in range(1, len(mid_prices))]
            gains = [change if change > 0 else 0 for change in price_changes]
            losses = [-change if change < 0 else 0 for change in price_changes]
            
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            
            if avg_loss == 0:
                features['rsi'] = 100
            else:
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
        else:
            features['rsi'] = 50  # Neutral RSI
        
        # Momentum
        if len(mid_prices) >= 10:
            features['momentum'] = mid_prices[-1] - mid_prices[-10]
        else:
            features['momentum'] = 0
        
        # Price acceleration (second derivative)
        if len(mid_prices) >= 3:
            first_diff = [mid_prices[i] - mid_prices[i-1] for i in range(1, len(mid_prices))]
            if len(first_diff) >= 2:
                second_diff = [first_diff[i] - first_diff[i-1] for i in range(1, len(first_diff))]
                features['price_acceleration'] = np.mean(second_diff)
            else:
                features['price_acceleration'] = 0
        else:
            features['price_acceleration'] = 0
        
        return features


def create_feature_matrix(feature_vectors: List[FeatureVector]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convert list of FeatureVector objects to feature matrix and target series.
    
    Args:
        feature_vectors: List of FeatureVector objects
        
    Returns:
        Tuple of (features DataFrame, targets Series)
    """
    if not feature_vectors:
        return pd.DataFrame(), pd.Series()
    
    # Extract features
    feature_data = []
    targets = []
    timestamps = []
    symbols = []
    
    for fv in feature_vectors:
        feature_data.append(fv.features)
        targets.append(fv.target)
        timestamps.append(fv.timestamp)
        symbols.append(fv.symbol)
    
    # Create DataFrame
    features_df = pd.DataFrame(feature_data)
    features_df['timestamp'] = timestamps
    features_df['symbol'] = symbols
    
    targets_series = pd.Series(targets, name='target')
    
    return features_df, targets_series


if __name__ == "__main__":
    # Example usage
    from .order_book_parser import create_synthetic_order_book_data
    
    # Generate synthetic data
    snapshots = create_synthetic_order_book_data(num_snapshots=100)
    
    # Extract features
    feature_engineer = FeatureEngineering(lookback_window=10, prediction_horizon=5)
    feature_vectors = feature_engineer.extract_features(snapshots)
    
    print(f"Extracted {len(feature_vectors)} feature vectors")
    
    if feature_vectors:
        sample_features = feature_vectors[0]
        print(f"Sample feature vector has {len(sample_features.features)} features:")
        for name, value in list(sample_features.features.items())[:10]:
            print(f"  {name}: {value:.6f}")
        
        # Create feature matrix
        features_df, targets = create_feature_matrix(feature_vectors)
        print(f"Feature matrix shape: {features_df.shape}")
        print(f"Target series length: {len(targets)}")