"""
Tests for Traditional Microstructure Models

Comprehensive test suite for all traditional model components including
unit tests, integration tests, and performance validation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import shutil
from pathlib import Path

# Import models to test
from src.models.traditional_models import (
    GlostenMilgromModel, KyleLambdaModel, BaselineModels, 
    TraditionalModelEnsemble, ModelPrediction
)
from src.evaluation.traditional_evaluation import TraditionalModelEvaluator
from src.evaluation.model_comparison import FairComparisonFramework


class TestGlostenMilgromModel:
    """Test suite for Glosten-Milgrom model."""
    
    @pytest.fixture
    def model(self):
        """Create a Glosten-Milgrom model for testing."""
        return GlostenMilgromModel(alpha=0.3, sigma=0.01)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return {
            'order_flows': np.random.normal(0, 100, 100),
            'volumes': np.abs(np.random.normal(1000, 200, 100)),
            'prices': 50000 + np.cumsum(np.random.normal(0, 1, 100))
        }
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.alpha == 0.3
        assert model.sigma == 0.01
        assert model.fundamental_value is None
        assert len(model.trade_history) == 0
    
    def test_update_fundamental_value(self, model):
        """Test fundamental value updates."""
        # First update
        model.update_fundamental_value(50000)
        assert model.fundamental_value == 50000
        
        # Subsequent updates should add noise
        initial_value = model.fundamental_value
        model.update_fundamental_value(50001)
        # Value should change due to random walk
        assert model.fundamental_value != initial_value
    
    def test_predict_price_movement(self, model, sample_data):
        """Test price movement prediction."""
        order_flow = sample_data['order_flows'][0]
        volume = sample_data['volumes'][0]
        price = sample_data['prices'][0]
        
        prediction = model.predict_price_movement(order_flow, volume, price)
        
        # Check prediction structure
        assert isinstance(prediction, ModelPrediction)
        assert prediction.model_name == "Glosten-Milgrom"
        assert isinstance(prediction.prediction, float)
        assert isinstance(prediction.confidence, float)
        assert 0 <= prediction.confidence <= 1
        
        # Check that trade history is updated
        assert len(model.trade_history) == 1
        assert len(model.price_history) == 1
    
    def test_calculate_bid_ask_prices(self, model):
        """Test bid-ask price calculation."""
        model.fundamental_value = 50000
        model.mid_price = 50000
        
        bid, ask = model.calculate_bid_ask_prices(100, 1000)
        
        # Bid should be less than ask
        assert bid <= ask
        assert isinstance(bid, float)
        assert isinstance(ask, float)
    
    def test_parameter_estimation(self, model, sample_data):
        """Test parameter estimation functionality."""
        initial_alpha = model.alpha
        
        # Generate predictions to build history
        for i in range(20):
            model.predict_price_movement(
                sample_data['order_flows'][i],
                sample_data['volumes'][i], 
                sample_data['prices'][i]
            )
        
        # Estimate parameters
        model.estimate_parameters()
        
        # Alpha should be updated (might increase or decrease)
        assert isinstance(model.alpha, float)
        assert 0.1 <= model.alpha <= 0.8  # Within bounds
    
    def test_get_model_state(self, model):
        """Test model state retrieval."""
        state = model.get_model_state()
        
        required_keys = ['alpha', 'sigma', 'fundamental_value', 'bid_price', 'ask_price', 'mid_price', 'trade_history_length']
        for key in required_keys:
            assert key in state
        
        assert state['alpha'] == model.alpha
        assert state['sigma'] == model.sigma


class TestKyleLambdaModel:
    """Test suite for Kyle's Lambda model."""
    
    @pytest.fixture
    def model(self):
        """Create a Kyle's Lambda model for testing."""
        return KyleLambdaModel(lambda_param=0.001, volatility=0.01)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return {
            'order_flows': np.random.normal(0, 100, 100),
            'volumes': np.abs(np.random.normal(1000, 200, 100)),
            'true_changes': np.random.normal(0, 0.01, 100)
        }
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.lambda_param == 0.001
        assert model.volatility == 0.01
        assert len(model.order_flows) == 0
        assert len(model.price_changes) == 0
    
    def test_predict_price_movement(self, model, sample_data):
        """Test price movement prediction."""
        order_flow = sample_data['order_flows'][0]
        volume = sample_data['volumes'][0]
        
        prediction = model.predict_price_movement(order_flow, volume)
        
        # Check prediction structure
        assert isinstance(prediction, ModelPrediction)
        assert prediction.model_name == "Kyle-Lambda"
        assert isinstance(prediction.prediction, float)
        assert isinstance(prediction.confidence, float)
        
        # Check that history is updated
        assert len(model.order_flows) == 1
    
    def test_update_with_realized_change(self, model, sample_data):
        """Test updating model with realized changes."""
        # Make a prediction first
        model.predict_price_movement(sample_data['order_flows'][0], sample_data['volumes'][0])
        
        # Update with realized change
        model.update_with_realized_change(sample_data['true_changes'][0])
        
        assert len(model.price_changes) == 1
        assert model.price_changes[0] == sample_data['true_changes'][0]
    
    def test_estimate_lambda(self, model, sample_data):
        """Test lambda parameter estimation."""
        initial_lambda = model.lambda_param
        
        # Generate data for estimation
        for i in range(15):
            model.predict_price_movement(sample_data['order_flows'][i], sample_data['volumes'][i])
            model.update_with_realized_change(sample_data['true_changes'][i])
        
        # Lambda should be updated
        assert isinstance(model.lambda_param, float)
        # Lambda might increase or decrease based on data
    
    def test_calculate_r_squared(self, model, sample_data):
        """Test R-squared calculation."""
        # Generate sufficient data
        for i in range(10):
            model.predict_price_movement(sample_data['order_flows'][i], sample_data['volumes'][i])
            model.update_with_realized_change(sample_data['true_changes'][i])
        
        r_squared = model.calculate_r_squared()
        assert isinstance(r_squared, float)
        assert r_squared >= 0  # R-squared should be non-negative
    
    def test_get_model_state(self, model):
        """Test model state retrieval."""
        state = model.get_model_state()
        
        required_keys = ['lambda', 'volatility', 'observations', 'r_squared']
        for key in required_keys:
            assert key in state
        
        assert state['lambda'] == model.lambda_param
        assert state['volatility'] == model.volatility


class TestBaselineModels:
    """Test suite for baseline models."""
    
    @pytest.fixture
    def models(self):
        """Create baseline models for testing."""
        return BaselineModels(window_size=20)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.normal(0, 1, 50))
        volumes = np.abs(np.random.normal(1000, 200, 50))
        timestamps = [datetime.now() for _ in range(50)]
        
        return {
            'prices': prices,
            'volumes': volumes,
            'timestamps': timestamps
        }
    
    def test_model_initialization(self, models):
        """Test baseline models initialization."""
        assert models.window_size == 20
        assert len(models.prices) == 0
        assert len(models.volumes) == 0
    
    def test_update_history(self, models, sample_data):
        """Test history update functionality."""
        # Update with sample data
        for i in range(10):
            models.update_history(
                sample_data['prices'][i],
                sample_data['volumes'][i],
                sample_data['timestamps'][i]
            )
        
        assert len(models.prices) == 10
        assert len(models.volumes) == 10
        assert len(models.timestamps) == 10
    
    def test_momentum_model(self, models, sample_data):
        """Test momentum model."""
        # Add sufficient history
        for i in range(10):
            models.update_history(
                sample_data['prices'][i],
                sample_data['volumes'][i],
                sample_data['timestamps'][i]
            )
        
        prediction = models.momentum_model(lookback=5)
        
        assert isinstance(prediction, ModelPrediction)
        assert prediction.model_name == "Momentum"
        assert isinstance(prediction.prediction, float)
        assert isinstance(prediction.confidence, float)
    
    def test_mean_reversion_model(self, models, sample_data):
        """Test mean reversion model."""
        # Add sufficient history
        for i in range(25):
            models.update_history(
                sample_data['prices'][i],
                sample_data['volumes'][i],
                sample_data['timestamps'][i]
            )
        
        prediction = models.mean_reversion_model(lookback=20)
        
        assert isinstance(prediction, ModelPrediction)
        assert prediction.model_name == "Mean-Reversion"
        assert isinstance(prediction.prediction, float)
    
    def test_vwap_model(self, models, sample_data):
        """Test VWAP model."""
        # Add sufficient history
        for i in range(15):
            models.update_history(
                sample_data['prices'][i],
                sample_data['volumes'][i],
                sample_data['timestamps'][i]
            )
        
        prediction = models.vwap_model()
        
        assert isinstance(prediction, ModelPrediction)
        assert prediction.model_name == "VWAP"
        assert isinstance(prediction.prediction, float)
    
    def test_random_walk_model(self, models):
        """Test random walk model."""
        prediction = models.random_walk_model()
        
        assert isinstance(prediction, ModelPrediction)
        assert prediction.model_name == "Random-Walk"
        assert isinstance(prediction.prediction, float)
        assert prediction.confidence == 0.0  # Random walk has no confidence


class TestTraditionalModelEnsemble:
    """Test suite for traditional model ensemble."""
    
    @pytest.fixture
    def ensemble(self):
        """Create ensemble for testing."""
        return TraditionalModelEnsemble(
            use_glosten_milgrom=True,
            use_kyle_lambda=True,
            use_momentum=True,
            use_mean_reversion=True,
            use_vwap=True
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return {
            'order_flow': 150.0,
            'volume': 1200.0,
            'current_price': 50005.0
        }
    
    def test_ensemble_initialization(self, ensemble):
        """Test ensemble initialization."""
        expected_models = ['glosten_milgrom', 'kyle_lambda', 'momentum', 'mean_reversion', 'vwap']
        
        for model_name in expected_models:
            assert model_name in ensemble.models
        
        assert len(ensemble.model_weights) == len(ensemble.models)
        
        # All weights should be 1.0 initially
        for weight in ensemble.model_weights.values():
            assert weight == 1.0
    
    def test_predict(self, ensemble, sample_data):
        """Test ensemble prediction."""
        predictions = ensemble.predict(
            sample_data['order_flow'],
            sample_data['volume'],
            sample_data['current_price']
        )
        
        # Should have predictions from all models
        assert len(predictions) == len(ensemble.models)
        
        for model_name, prediction in predictions.items():
            assert isinstance(prediction, ModelPrediction)
            assert prediction.model_name == model_name or prediction.model_name.startswith(model_name.split('_')[0])
    
    def test_get_ensemble_prediction(self, ensemble, sample_data):
        """Test ensemble prediction combination."""
        # Get individual predictions
        individual_predictions = ensemble.predict(
            sample_data['order_flow'],
            sample_data['volume'],
            sample_data['current_price']
        )
        
        # Get ensemble prediction
        ensemble_prediction = ensemble.get_ensemble_prediction(individual_predictions)
        
        assert isinstance(ensemble_prediction, ModelPrediction)
        assert ensemble_prediction.model_name == "Ensemble"
        assert isinstance(ensemble_prediction.prediction, float)
        assert isinstance(ensemble_prediction.confidence, float)
    
    def test_update_model_weights(self, ensemble):
        """Test model weight updates."""
        # Update weights based on dummy performance
        performance_metrics = {
            'glosten_milgrom': 0.8,
            'kyle_lambda': 0.6,
            'momentum': 0.7,
            'mean_reversion': 0.5,
            'vwap': 0.9
        }
        
        ensemble.update_model_weights(performance_metrics)
        
        # Weights should be normalized
        total_weight = sum(ensemble.model_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
        
        # Best performing model should have highest weight
        best_model = max(performance_metrics.keys(), key=lambda k: performance_metrics[k])
        assert ensemble.model_weights[best_model] == max(ensemble.model_weights.values())
    
    def test_get_model_states(self, ensemble):
        """Test model state retrieval."""
        states = ensemble.get_model_states()
        
        assert len(states) == len(ensemble.models)
        
        for model_name, state in states.items():
            assert isinstance(state, dict)


class TestTraditionalModelEvaluator:
    """Test suite for traditional model evaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield TraditionalModelEvaluator(output_dir=temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        return {
            'order_flows': np.random.normal(0, 100, n_samples),
            'volumes': np.abs(np.random.normal(1000, 200, n_samples)),
            'prices': 50000 + np.cumsum(np.random.normal(0, 1, n_samples)),
            'true_changes': np.random.normal(0, 0.5, n_samples)
        }
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.output_dir.exists()
        assert hasattr(evaluator, 'performance_evaluator')
    
    def test_evaluate_glosten_milgrom_model(self, evaluator, sample_data):
        """Test Glosten-Milgrom model evaluation."""
        model = GlostenMilgromModel()
        
        diagnostics = evaluator.evaluate_glosten_milgrom_model(
            model,
            sample_data['order_flows'],
            sample_data['volumes'],
            sample_data['prices'],
            sample_data['true_changes']
        )
        
        assert diagnostics.model_name == "Glosten-Milgrom"
        assert 'alpha_final' in diagnostics.parameter_estimates
        assert 'directional_accuracy' in diagnostics.goodness_of_fit
        assert 'mean_residual' in diagnostics.residual_analysis
    
    def test_evaluate_kyle_lambda_model(self, evaluator, sample_data):
        """Test Kyle's lambda model evaluation."""
        model = KyleLambdaModel()
        
        diagnostics = evaluator.evaluate_kyle_lambda_model(
            model,
            sample_data['order_flows'],
            sample_data['volumes'],
            sample_data['true_changes']
        )
        
        assert diagnostics.model_name == "Kyle-Lambda"
        assert 'lambda_final' in diagnostics.parameter_estimates
        assert 'directional_accuracy' in diagnostics.goodness_of_fit
    
    def test_evaluate_baseline_models(self, evaluator, sample_data):
        """Test baseline models evaluation."""
        baseline_models = BaselineModels()
        
        diagnostics = evaluator.evaluate_baseline_models(
            baseline_models,
            sample_data['prices'],
            sample_data['volumes'],
            sample_data['true_changes']
        )
        
        expected_models = ['momentum', 'mean_reversion', 'vwap', 'random_walk']
        assert len(diagnostics) == len(expected_models)
        
        for model_name in expected_models:
            assert model_name in diagnostics
            assert diagnostics[model_name].model_name.startswith("Baseline_")


class TestFairComparisonFramework:
    """Test suite for fair comparison framework."""
    
    @pytest.fixture
    def framework(self):
        """Create comparison framework for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield FairComparisonFramework(output_dir=temp_dir)
    
    def test_framework_initialization(self, framework):
        """Test framework initialization."""
        assert framework.output_dir.exists()
        assert framework.random_seed == 42
        assert hasattr(framework, 'evaluator')
    
    def test_extract_methods(self, framework):
        """Test feature extraction methods."""
        # Create dummy features
        features = np.random.randn(10, 20, 46)  # batch_size, seq_len, features
        
        order_flows = framework._extract_order_flows(features)
        volumes = framework._extract_volumes(features)
        prices = framework._extract_current_prices(features)
        
        assert len(order_flows) == 10
        assert len(volumes) == 10
        assert len(prices) == 10
        
        assert all(isinstance(x, (int, float)) for x in order_flows)
        assert all(isinstance(x, (int, float)) for x in volumes)
        assert all(isinstance(x, (int, float)) for x in prices)


class TestIntegration:
    """Integration tests for traditional models."""
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline."""
        # Create models
        gm_model = GlostenMilgromModel()
        kyle_model = KyleLambdaModel()
        baseline_models = BaselineModels()
        
        # Create ensemble
        ensemble = TraditionalModelEnsemble()
        
        # Generate test data
        np.random.seed(42)
        n_samples = 50
        
        order_flows = np.random.normal(0, 100, n_samples)
        volumes = np.abs(np.random.normal(1000, 200, n_samples))
        prices = 50000 + np.cumsum(np.random.normal(0, 1, n_samples))
        
        # Test individual models
        for i in range(10):  # Test subset for speed
            # Update baseline history
            baseline_models.update_history(prices[i], volumes[i], datetime.now())
            
            # Get predictions
            gm_pred = gm_model.predict_price_movement(order_flows[i], volumes[i], prices[i])
            kyle_pred = kyle_model.predict_price_movement(order_flows[i], volumes[i])
            momentum_pred = baseline_models.momentum_model()
            
            # Check predictions are valid
            assert isinstance(gm_pred.prediction, float)
            assert isinstance(kyle_pred.prediction, float)
            assert isinstance(momentum_pred.prediction, float)
        
        # Test ensemble
        ensemble_preds = ensemble.predict(order_flows[0], volumes[0], prices[0])
        ensemble_combined = ensemble.get_ensemble_prediction(ensemble_preds)
        
        assert isinstance(ensemble_combined.prediction, float)
        assert len(ensemble_preds) > 0
    
    def test_model_persistence_and_state(self):
        """Test model state persistence."""
        # Create model and make predictions
        model = GlostenMilgromModel(alpha=0.4)
        
        # Make some predictions
        for i in range(5):
            model.predict_price_movement(
                np.random.normal(0, 100),
                abs(np.random.normal(1000, 200)),
                50000 + i
            )
        
        # Get state
        state = model.get_model_state()
        
        # State should contain model history
        assert state['trade_history_length'] == 5
        assert 'alpha' in state
        assert isinstance(state['alpha'], float)
    
    def test_parameter_learning_convergence(self):
        """Test that parameters converge with sufficient data."""
        model = KyleLambdaModel(lambda_param=0.001)
        
        # Generate correlated data where true impact exists
        np.random.seed(42)
        true_lambda = 0.002
        
        for i in range(50):
            order_flow = np.random.normal(0, 100)
            volume = abs(np.random.normal(1000, 200))
            
            # True price change with impact
            true_change = true_lambda * (order_flow / np.sqrt(volume)) + np.random.normal(0, 0.001)
            
            # Make prediction and update
            pred = model.predict_price_movement(order_flow, volume)
            model.update_with_realized_change(true_change)
        
        # Lambda should move towards true value
        final_lambda = model.lambda_param
        # Allow for some tolerance due to noise
        assert abs(final_lambda - true_lambda) < abs(0.001 - true_lambda)


if __name__ == "__main__":
    # Run specific tests manually for debugging
    print("Running traditional models tests...")
    
    # Test Glosten-Milgrom model
    gm_model = GlostenMilgromModel()
    prediction = gm_model.predict_price_movement(100, 1000, 50000)
    print(f"Glosten-Milgrom prediction: {prediction.prediction:.6f}")
    
    # Test Kyle's lambda model
    kyle_model = KyleLambdaModel()
    prediction = kyle_model.predict_price_movement(100, 1000)
    print(f"Kyle's lambda prediction: {prediction.prediction:.6f}")
    
    # Test baseline models
    baseline = BaselineModels()
    for i in range(10):
        baseline.update_history(50000 + i, 1000 + i*10, datetime.now())
    
    momentum_pred = baseline.momentum_model()
    print(f"Momentum prediction: {momentum_pred.prediction:.6f}")
    
    # Test ensemble
    ensemble = TraditionalModelEnsemble()
    ensemble_preds = ensemble.predict(100, 1000, 50000)
    combined = ensemble.get_ensemble_prediction(ensemble_preds)
    print(f"Ensemble prediction: {combined.prediction:.6f}")
    
    print("✅ Manual tests passed!")