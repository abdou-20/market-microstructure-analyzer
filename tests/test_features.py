"""
Unit tests for feature engineering.

Tests for order book parsing and feature extraction.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from src.data_processing.order_book_parser import (
    OrderBookParser, OrderBookSnapshot, create_synthetic_order_book_data
)
from src.data_processing.feature_engineering import (
    FeatureEngineering, FeatureVector, create_feature_matrix
)


class TestOrderBookParser:
    """Test cases for order book parser."""
    
    @pytest.fixture
    def sample_bids(self) -> List[tuple]:
        """Sample bid data."""
        return [(50000.0, 1.5), (49999.0, 2.0), (49998.0, 1.8), (49997.0, 2.5)]
    
    @pytest.fixture
    def sample_asks(self) -> List[tuple]:
        """Sample ask data."""
        return [(50001.0, 1.2), (50002.0, 1.8), (50003.0, 2.1), (50004.0, 1.9)]
    
    @pytest.fixture
    def sample_snapshot(self, sample_bids, sample_asks) -> OrderBookSnapshot:
        """Sample order book snapshot."""
        return OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            bids=sample_bids,
            asks=sample_asks,
            best_bid=0.0,
            best_ask=0.0,
            spread=0.0,
            mid_price=0.0
        )
    
    def test_order_book_snapshot_creation(self, sample_snapshot):
        """Test order book snapshot creation and derived fields."""
        assert sample_snapshot.symbol == "BTC-USD"
        assert sample_snapshot.best_bid == 50000.0
        assert sample_snapshot.best_ask == 50001.0
        assert sample_snapshot.spread == 1.0
        assert sample_snapshot.mid_price == 50000.5
    
    def test_order_book_snapshot_to_dict(self, sample_snapshot):
        """Test conversion to dictionary."""
        snapshot_dict = sample_snapshot.to_dict()
        
        assert isinstance(snapshot_dict, dict)
        assert 'timestamp' in snapshot_dict
        assert 'symbol' in snapshot_dict
        assert 'bids' in snapshot_dict
        assert 'asks' in snapshot_dict
        assert 'best_bid' in snapshot_dict
        assert 'best_ask' in snapshot_dict
        assert 'spread' in snapshot_dict
        assert 'mid_price' in snapshot_dict
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = OrderBookParser(max_levels=10, validate_data=True, sort_levels=True)
        
        assert parser.max_levels == 10
        assert parser.validate_data == True
        assert parser.sort_levels == True
        assert parser.processed_count == 0
        assert parser.error_count == 0
    
    def test_synthetic_data_generation(self):
        """Test synthetic order book data generation."""
        snapshots = create_synthetic_order_book_data(
            symbol="TEST-USD",
            num_snapshots=100,
            start_price=1000.0,
            volatility=0.01,
            num_levels=5
        )
        
        assert len(snapshots) == 100
        assert all(s.symbol == "TEST-USD" for s in snapshots)
        assert all(len(s.bids) == 5 for s in snapshots)
        assert all(len(s.asks) == 5 for s in snapshots)
        assert all(s.best_bid > 0 for s in snapshots)
        assert all(s.best_ask > s.best_bid for s in snapshots)
    
    def test_parser_statistics(self):
        """Test parser statistics tracking."""
        parser = OrderBookParser()
        
        # Generate and parse some data
        snapshots = create_synthetic_order_book_data(num_snapshots=50)
        processed_snapshots = list(parser._validate_snapshots(snapshots))
        
        stats = parser.get_statistics()
        assert 'processed_count' in stats
        assert 'error_count' in stats
        assert 'success_rate' in stats
        assert stats['processed_count'] >= 0
        assert stats['error_count'] >= 0
    
    def _validate_snapshots(self, parser, snapshots):
        """Helper method to validate snapshots."""
        for snapshot in snapshots:
            if parser._validate_snapshot(snapshot):
                parser.processed_count += 1
                yield snapshot
            else:
                parser.error_count += 1


class TestFeatureEngineering:
    """Test cases for feature engineering."""
    
    @pytest.fixture
    def sample_snapshots(self) -> List[OrderBookSnapshot]:
        """Sample order book snapshots for testing."""
        return create_synthetic_order_book_data(
            num_snapshots=50,
            start_price=50000.0,
            volatility=0.001,
            num_levels=10
        )
    
    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineering:
        """Feature engineering instance."""
        return FeatureEngineering(
            lookback_window=10,
            prediction_horizon=5,
            max_levels=10
        )
    
    def test_feature_engineering_initialization(self, feature_engineer):
        """Test feature engineering initialization."""
        assert feature_engineer.lookback_window == 10
        assert feature_engineer.prediction_horizon == 5
        assert feature_engineer.max_levels == 10
        assert len(feature_engineer.snapshot_buffer) == 0
    
    def test_feature_extraction(self, feature_engineer, sample_snapshots):
        """Test feature extraction from snapshots."""
        feature_vectors = feature_engineer.extract_features(sample_snapshots)
        
        assert len(feature_vectors) > 0
        assert all(isinstance(fv, FeatureVector) for fv in feature_vectors)
        
        # Check that each feature vector has the expected structure
        for fv in feature_vectors:
            assert isinstance(fv.timestamp, datetime)
            assert isinstance(fv.symbol, str)
            assert isinstance(fv.features, dict)
            assert isinstance(fv.target, (float, type(None)))
    
    def test_basic_features(self, feature_engineer, sample_snapshots):
        """Test basic order book features."""
        feature_vectors = feature_engineer.extract_features(sample_snapshots)
        
        if feature_vectors:
            features = feature_vectors[0].features
            
            # Check presence of basic features
            expected_basic_features = [
                'mid_price', 'best_bid', 'best_ask', 'spread_absolute', 'spread_relative',
                'best_bid_qty', 'best_ask_qty', 'bid_ask_qty_ratio',
                'total_bid_volume', 'total_ask_volume', 'volume_imbalance'
            ]
            
            for feature in expected_basic_features:
                assert feature in features, f"Missing basic feature: {feature}"
                assert isinstance(features[feature], (int, float, np.integer, np.floating))
    
    def test_ofi_features(self, feature_engineer, sample_snapshots):
        """Test Order Flow Imbalance features."""
        feature_vectors = feature_engineer.extract_features(sample_snapshots)
        
        if feature_vectors:
            features = feature_vectors[0].features
            
            # Check OFI features
            ofi_features = ['ofi_1', 'ofi_5', 'ofi_10']
            for feature in ofi_features:
                assert feature in features, f"Missing OFI feature: {feature}"
                assert isinstance(features[feature], (int, float, np.integer, np.floating))
    
    def test_spread_features(self, feature_engineer, sample_snapshots):
        """Test spread-related features."""
        feature_vectors = feature_engineer.extract_features(sample_snapshots)
        
        if feature_vectors:
            features = feature_vectors[0].features
            
            # Check spread features
            spread_features = [
                'spread_bps', 'spread_mean', 'spread_std', 'spread_min', 'spread_max',
                'relative_spread_mean', 'relative_spread_std', 'effective_spread'
            ]
            
            for feature in spread_features:
                assert feature in features, f"Missing spread feature: {feature}"
                assert isinstance(features[feature], (int, float, np.integer, np.floating))
    
    def test_volume_features(self, feature_engineer, sample_snapshots):
        """Test volume-based features."""
        feature_vectors = feature_engineer.extract_features(sample_snapshots)
        
        if feature_vectors:
            features = feature_vectors[0].features
            
            # Check volume features
            volume_features = [
                'vwap_bid', 'vwap_ask', 'vwap_mid',
                'volume_mean', 'volume_std', 'volume_current_zscore'
            ]
            
            for feature in volume_features:
                assert feature in features, f"Missing volume feature: {feature}"
                assert isinstance(features[feature], (int, float, np.integer, np.floating))
            
            # Check depth imbalance features
            for level in range(1, 6):
                feature_name = f'depth_imbalance_l{level}'
                assert feature_name in features, f"Missing depth imbalance feature: {feature_name}"
    
    def test_price_features(self, feature_engineer, sample_snapshots):
        """Test price-based features."""
        feature_vectors = feature_engineer.extract_features(sample_snapshots)
        
        if feature_vectors:
            features = feature_vectors[0].features
            
            # Check price features
            price_features = [
                'price_return_1', 'price_return_5', 'price_volatility', 'price_trend'
            ]
            
            for feature in price_features:
                assert feature in features, f"Missing price feature: {feature}"
                assert isinstance(features[feature], (int, float, np.integer, np.floating))
    
    def test_microstructure_features(self, feature_engineer, sample_snapshots):
        """Test microstructure features."""
        feature_vectors = feature_engineer.extract_features(sample_snapshots)
        
        if feature_vectors:
            features = feature_vectors[0].features
            
            # Check microstructure features
            micro_features = [
                'trade_intensity', 'arrival_rate', 'spread_resilience'
            ]
            
            for feature in micro_features:
                assert feature in features, f"Missing microstructure feature: {feature}"
                assert isinstance(features[feature], (int, float, np.integer, np.floating))
    
    def test_technical_features(self, feature_engineer, sample_snapshots):
        """Test technical analysis features."""
        feature_vectors = feature_engineer.extract_features(sample_snapshots)
        
        if feature_vectors:
            features = feature_vectors[0].features
            
            # Check technical features
            technical_features = [
                'sma_5', 'price_vs_sma', 'rsi', 'momentum', 'price_acceleration'
            ]
            
            for feature in technical_features:
                assert feature in features, f"Missing technical feature: {feature}"
                assert isinstance(features[feature], (int, float, np.integer, np.floating))
    
    def test_target_calculation(self, feature_engineer, sample_snapshots):
        """Test target calculation."""
        feature_vectors = feature_engineer.extract_features(sample_snapshots)
        
        for fv in feature_vectors:
            if fv.target is not None:
                assert isinstance(fv.target, (float, np.floating))
                # Target should be a reasonable percentage change
                assert -1.0 <= fv.target <= 1.0, f"Target out of reasonable range: {fv.target}"
    
    def test_feature_vector_to_dict(self, feature_engineer, sample_snapshots):
        """Test feature vector conversion to dictionary."""
        feature_vectors = feature_engineer.extract_features(sample_snapshots)
        
        if feature_vectors:
            fv_dict = feature_vectors[0].to_dict()
            
            assert isinstance(fv_dict, dict)
            assert 'timestamp' in fv_dict
            assert 'symbol' in fv_dict
            assert 'target' in fv_dict
            
            # All features should be in the dictionary
            for feature_name, feature_value in feature_vectors[0].features.items():
                assert feature_name in fv_dict
                assert fv_dict[feature_name] == feature_value


class TestFeatureMatrix:
    """Test cases for feature matrix creation."""
    
    @pytest.fixture
    def sample_feature_vectors(self) -> List[FeatureVector]:
        """Sample feature vectors."""
        snapshots = create_synthetic_order_book_data(num_snapshots=30)
        feature_engineer = FeatureEngineering(lookback_window=5, prediction_horizon=3)
        return feature_engineer.extract_features(snapshots)
    
    def test_create_feature_matrix(self, sample_feature_vectors):
        """Test feature matrix creation."""
        features_df, targets_series = create_feature_matrix(sample_feature_vectors)
        
        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(targets_series, pd.Series)
        assert len(features_df) == len(targets_series)
        assert len(features_df) == len(sample_feature_vectors)
        
        # Check that timestamps and symbols are included
        assert 'timestamp' in features_df.columns
        assert 'symbol' in features_df.columns
        
        # Check that all feature columns are numeric (except timestamp and symbol)
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        assert len(numeric_columns) > 0
    
    def test_empty_feature_vectors(self):
        """Test feature matrix creation with empty input."""
        features_df, targets_series = create_feature_matrix([])
        
        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(targets_series, pd.Series)
        assert len(features_df) == 0
        assert len(targets_series) == 0
    
    def test_feature_matrix_consistency(self, sample_feature_vectors):
        """Test feature matrix consistency."""
        features_df, targets_series = create_feature_matrix(sample_feature_vectors)
        
        # Check that all rows have the same number of features
        feature_columns = [col for col in features_df.columns 
                          if col not in ['timestamp', 'symbol']]
        
        for col in feature_columns:
            assert not features_df[col].isna().all(), f"Column {col} is all NaN"
        
        # Check that targets are not all NaN
        assert not targets_series.isna().all(), "All targets are NaN"


class TestFeatureValidation:
    """Test cases for feature validation and edge cases."""
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Create very few snapshots
        snapshots = create_synthetic_order_book_data(num_snapshots=5)
        feature_engineer = FeatureEngineering(lookback_window=10, prediction_horizon=5)
        
        feature_vectors = feature_engineer.extract_features(snapshots)
        
        # Should return empty list or handle gracefully
        assert isinstance(feature_vectors, list)
    
    def test_zero_prices(self):
        """Test handling of zero prices."""
        # Create snapshot with zero prices (edge case)
        timestamp = datetime.now()
        bids = [(0.0, 1.0)]
        asks = [(0.0, 1.0)]
        
        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            symbol="TEST",
            bids=bids,
            asks=asks,
            best_bid=0.0,
            best_ask=0.0,
            spread=0.0,
            mid_price=0.0
        )
        
        # Should handle gracefully without errors
        feature_engineer = FeatureEngineering()
        # This should not raise an exception
        basic_features = feature_engineer._extract_basic_features(snapshot)
        assert isinstance(basic_features, dict)
    
    def test_empty_order_book(self):
        """Test handling of empty order book."""
        timestamp = datetime.now()
        
        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            symbol="TEST",
            bids=[],
            asks=[],
            best_bid=0.0,
            best_ask=0.0,
            spread=0.0,
            mid_price=0.0
        )
        
        feature_engineer = FeatureEngineering()
        basic_features = feature_engineer._extract_basic_features(snapshot)
        
        # Should return features with appropriate default values
        assert isinstance(basic_features, dict)
        assert basic_features['best_bid_qty'] == 0.0
        assert basic_features['best_ask_qty'] == 0.0
        assert basic_features['total_bid_volume'] == 0.0
        assert basic_features['total_ask_volume'] == 0.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])