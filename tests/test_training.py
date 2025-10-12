"""
Unit tests for training pipeline.

Tests for loss functions, trainer, and validation.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from src.training.loss_functions import (
    DirectionalAccuracyLoss, SharpeRatioLoss, HuberLoss, QuantileLoss,
    TrendAwareLoss, VolatilityAwareLoss, CombinedLoss, create_loss_function
)
from src.training.validation import (
    PerformanceEvaluator, ValidationMetrics, WalkForwardValidator
)


class TestLossFunctions:
    """Test cases for custom loss functions."""
    
    @pytest.fixture
    def sample_predictions(self) -> torch.Tensor:
        """Sample predictions."""
        return torch.randn(32, 1) * 0.01  # Small price movements
    
    @pytest.fixture
    def sample_targets(self) -> torch.Tensor:
        """Sample targets."""
        return torch.randn(32, 1) * 0.01
    
    def test_directional_accuracy_loss(self, sample_predictions, sample_targets):
        """Test directional accuracy loss."""
        loss_fn = DirectionalAccuracyLoss(margin=0.001)
        
        loss = loss_fn(sample_predictions, sample_targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert 0.0 <= loss.item() <= 1.0  # Loss should be between 0 and 1
    
    def test_directional_accuracy_loss_perfect_predictions(self):
        """Test directional accuracy loss with perfect predictions."""
        loss_fn = DirectionalAccuracyLoss(margin=0.0)
        
        # Perfect predictions (same signs)
        predictions = torch.tensor([0.01, -0.01, 0.02, -0.015]).unsqueeze(1)
        targets = torch.tensor([0.005, -0.008, 0.012, -0.007]).unsqueeze(1)
        
        loss = loss_fn(predictions, targets)
        
        assert loss.item() == 0.0  # Perfect directional accuracy
    
    def test_sharpe_ratio_loss(self, sample_predictions, sample_targets):
        """Test Sharpe ratio loss."""
        loss_fn = SharpeRatioLoss(risk_free_rate=0.0)
        
        loss = loss_fn(sample_predictions, sample_targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad  # Should be differentiable
    
    def test_huber_loss(self, sample_predictions, sample_targets):
        """Test Huber loss."""
        loss_fn = HuberLoss(delta=0.005)
        
        loss = loss_fn(sample_predictions, sample_targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0  # Loss should be non-negative
    
    def test_huber_loss_behavior(self):
        """Test Huber loss behavior with different residuals."""
        loss_fn = HuberLoss(delta=1.0)
        
        # Small residual (should behave like MSE)
        predictions = torch.tensor([0.0])
        targets = torch.tensor([0.5])  # residual = 0.5 < delta
        
        loss_small = loss_fn(predictions, targets)
        expected_mse = 0.5 * (0.5 ** 2)
        assert abs(loss_small.item() - expected_mse) < 1e-6
        
        # Large residual (should behave like MAE)
        targets = torch.tensor([2.0])  # residual = 2.0 > delta
        loss_large = loss_fn(predictions, targets)
        expected_mae = 1.0 * 2.0 - 0.5 * (1.0 ** 2)
        assert abs(loss_large.item() - expected_mae) < 1e-6
    
    def test_quantile_loss(self, sample_predictions, sample_targets):
        """Test quantile loss."""
        loss_fn = QuantileLoss(quantile=0.5)  # Median
        
        loss = loss_fn(sample_predictions, sample_targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0
    
    def test_quantile_loss_different_quantiles(self):
        """Test quantile loss with different quantiles."""
        predictions = torch.tensor([0.0, 0.0, 0.0])
        targets = torch.tensor([1.0, -1.0, 0.5])
        
        # Test different quantiles
        for quantile in [0.1, 0.5, 0.9]:
            loss_fn = QuantileLoss(quantile=quantile)
            loss = loss_fn(predictions, targets)
            assert loss.item() >= 0.0
    
    def test_trend_aware_loss(self, sample_predictions, sample_targets):
        """Test trend-aware loss."""
        loss_fn = TrendAwareLoss(trend_window=5, trend_threshold=0.01, trend_penalty=2.0)
        
        # Create historical targets
        historical_targets = torch.randn(10) * 0.01
        
        loss = loss_fn(sample_predictions, sample_targets, historical_targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0
    
    def test_volatility_aware_loss(self, sample_predictions, sample_targets):
        """Test volatility-aware loss."""
        loss_fn = VolatilityAwareLoss(volatility_window=10, base_loss_type='mse')
        
        # Create historical targets
        historical_targets = torch.randn(15) * 0.01
        
        loss = loss_fn(sample_predictions, sample_targets, historical_targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0
    
    def test_combined_loss(self, sample_predictions, sample_targets):
        """Test combined loss function."""
        loss_configs = {
            'mse': {'weight': 1.0, 'params': {}},
            'directional': {'weight': 0.5, 'params': {'margin': 0.001}},
            'huber': {'weight': 0.3, 'params': {'delta': 0.005}}
        }
        
        loss_fn = CombinedLoss(loss_configs)
        
        total_loss, individual_losses = loss_fn(sample_predictions, sample_targets)
        
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(individual_losses, dict)
        assert 'mse' in individual_losses
        assert 'directional' in individual_losses
        assert 'huber' in individual_losses
        
        # Check that total loss is combination of individual losses
        expected_total = (1.0 * individual_losses['mse'] + 
                         0.5 * individual_losses['directional'] + 
                         0.3 * individual_losses['huber'])
        assert abs(total_loss.item() - expected_total.item()) < 1e-6
    
    def test_create_loss_function(self):
        """Test loss function factory."""
        # Test MSE
        mse_config = {'loss_function': 'mse'}
        mse_loss = create_loss_function(mse_config)
        assert isinstance(mse_loss, torch.nn.MSELoss)
        
        # Test custom loss
        directional_config = {
            'loss_function': 'directional',
            'loss_params': {'margin': 0.001}
        }
        dir_loss = create_loss_function(directional_config)
        assert isinstance(dir_loss, DirectionalAccuracyLoss)
        
        # Test combined loss
        combined_config = {
            'loss_function': 'combined',
            'loss_params': {
                'loss_configs': {
                    'mse': {'weight': 1.0, 'params': {}},
                    'directional': {'weight': 0.5, 'params': {}}
                }
            }
        }
        combined_loss = create_loss_function(combined_config)
        assert isinstance(combined_loss, CombinedLoss)


class TestPerformanceEvaluator:
    """Test cases for performance evaluator."""
    
    @pytest.fixture
    def evaluator(self) -> PerformanceEvaluator:
        """Performance evaluator instance."""
        return PerformanceEvaluator(risk_free_rate=0.0)
    
    @pytest.fixture
    def sample_data(self) -> tuple:
        """Sample predictions and targets."""
        np.random.seed(42)  # For reproducibility
        predictions = np.random.randn(100) * 0.01
        targets = predictions + np.random.randn(100) * 0.005  # Add some noise
        return predictions, targets
    
    def test_evaluate_predictions(self, evaluator, sample_data):
        """Test prediction evaluation."""
        predictions, targets = sample_data
        
        metrics = evaluator.evaluate_predictions(predictions, targets)
        
        assert isinstance(metrics, ValidationMetrics)
        assert hasattr(metrics, 'mse')
        assert hasattr(metrics, 'mae')
        assert hasattr(metrics, 'rmse')
        assert hasattr(metrics, 'directional_accuracy')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'win_rate')
        assert hasattr(metrics, 'profit_factor')
        assert hasattr(metrics, 'calmar_ratio')
        
        # Check that metrics are reasonable
        assert metrics.mse >= 0.0
        assert metrics.mae >= 0.0
        assert metrics.rmse >= 0.0
        assert 0.0 <= metrics.directional_accuracy <= 1.0
        assert 0.0 <= metrics.win_rate <= 1.0
        assert metrics.max_drawdown >= 0.0
    
    def test_perfect_predictions(self, evaluator):
        """Test evaluation with perfect predictions."""
        targets = np.array([0.01, -0.01, 0.005, -0.008, 0.012])
        predictions = targets.copy()  # Perfect predictions
        
        metrics = evaluator.evaluate_predictions(predictions, targets)
        
        assert metrics.mse == 0.0
        assert metrics.mae == 0.0
        assert metrics.rmse == 0.0
        assert metrics.directional_accuracy == 1.0
    
    def test_random_predictions(self, evaluator):
        """Test evaluation with random predictions."""
        np.random.seed(123)
        targets = np.random.randn(50) * 0.01
        predictions = np.random.randn(50) * 0.01  # Uncorrelated predictions
        
        metrics = evaluator.evaluate_predictions(predictions, targets)
        
        # Directional accuracy should be around 0.5 for random predictions
        assert 0.3 <= metrics.directional_accuracy <= 0.7
    
    def test_trading_returns_simulation(self, evaluator):
        """Test trading returns simulation."""
        predictions = np.array([0.01, -0.01, 0.005, -0.008])
        targets = np.array([0.008, -0.012, 0.003, -0.006])
        
        trading_returns = evaluator._simulate_trading_returns(predictions, targets)
        
        # Check that returns match expected trading strategy
        expected_returns = np.array([0.008, 0.012, 0.003, 0.006])  # All positive (good predictions)
        np.testing.assert_array_almost_equal(trading_returns, expected_returns)
    
    def test_sharpe_ratio_calculation(self, evaluator):
        """Test Sharpe ratio calculation."""
        # Create returns with known properties
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.008])
        
        sharpe = evaluator._calculate_sharpe_ratio(returns)
        
        # Manual calculation
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        expected_sharpe = (mean_return - 0.0) / std_return * np.sqrt(252)
        
        assert abs(sharpe - expected_sharpe) < 1e-10
    
    def test_max_drawdown_calculation(self, evaluator):
        """Test maximum drawdown calculation."""
        # Create returns that lead to a known drawdown
        returns = np.array([0.1, -0.05, -0.1, 0.05, 0.02])
        
        max_dd = evaluator._calculate_max_drawdown(returns)
        
        # Calculate expected drawdown manually
        cumulative = np.cumprod(1 + returns)  # [1.1, 1.045, 0.9405, 0.98753, 1.00748]
        running_max = np.maximum.accumulate(cumulative)  # [1.1, 1.1, 1.1, 1.1, 1.1]
        drawdown = (cumulative - running_max) / running_max
        expected_max_dd = abs(np.min(drawdown))
        
        assert abs(max_dd - expected_max_dd) < 1e-10
    
    def test_win_rate_calculation(self, evaluator):
        """Test win rate calculation."""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.005])  # 3 wins out of 5
        
        win_rate = evaluator._calculate_win_rate(returns)
        
        assert win_rate == 0.6  # 3/5 = 0.6
    
    def test_profit_factor_calculation(self, evaluator):
        """Test profit factor calculation."""
        returns = np.array([0.02, -0.01, 0.03, -0.015])  # Profits: 0.05, Losses: -0.025
        
        profit_factor = evaluator._calculate_profit_factor(returns)
        
        expected_pf = 0.05 / 0.025  # 2.0
        assert abs(profit_factor - expected_pf) < 1e-10
    
    def test_metrics_to_dict(self, evaluator, sample_data):
        """Test metrics conversion to dictionary."""
        predictions, targets = sample_data
        
        metrics = evaluator.evaluate_predictions(predictions, targets)
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        expected_keys = [
            'mse', 'mae', 'rmse', 'directional_accuracy', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'profit_factor', 'calmar_ratio'
        ]
        
        for key in expected_keys:
            assert key in metrics_dict
            assert isinstance(metrics_dict[key], (int, float))


class TestValidationEdgeCases:
    """Test edge cases in validation."""
    
    def test_empty_predictions(self):
        """Test evaluation with empty predictions."""
        evaluator = PerformanceEvaluator()
        
        predictions = np.array([])
        targets = np.array([])
        
        # Should handle gracefully
        metrics = evaluator.evaluate_predictions(predictions, targets)
        
        # Most metrics should be 0 or NaN, but shouldn't crash
        assert isinstance(metrics, ValidationMetrics)
    
    def test_zero_predictions(self):
        """Test evaluation with all zero predictions."""
        evaluator = PerformanceEvaluator()
        
        predictions = np.zeros(10)
        targets = np.random.randn(10) * 0.01
        
        metrics = evaluator.evaluate_predictions(predictions, targets)
        
        # Should handle gracefully
        assert isinstance(metrics, ValidationMetrics)
        assert metrics.mse >= 0.0
    
    def test_zero_volatility(self):
        """Test evaluation with zero volatility."""
        evaluator = PerformanceEvaluator()
        
        predictions = np.ones(10) * 0.01  # Constant predictions
        targets = np.ones(10) * 0.01     # Constant targets
        
        metrics = evaluator.evaluate_predictions(predictions, targets)
        
        assert metrics.mse == 0.0
        assert metrics.mae == 0.0
        # Sharpe ratio might be undefined due to zero volatility


class TestLossGradients:
    """Test gradient flow through loss functions."""
    
    def test_loss_gradients(self):
        """Test that all loss functions produce gradients."""
        predictions = torch.randn(10, 1, requires_grad=True)
        targets = torch.randn(10, 1)
        
        loss_functions = [
            DirectionalAccuracyLoss(),
            SharpeRatioLoss(),
            HuberLoss(),
            QuantileLoss(),
        ]
        
        for loss_fn in loss_functions:
            if predictions.grad is not None:
                predictions.grad.zero_()
            
            loss = loss_fn(predictions, targets)
            loss.backward()
            
            # Check that gradients exist and are not all zero
            assert predictions.grad is not None
            assert not torch.allclose(predictions.grad, torch.zeros_like(predictions.grad))
    
    def test_combined_loss_gradients(self):
        """Test gradients for combined loss."""
        predictions = torch.randn(10, 1, requires_grad=True)
        targets = torch.randn(10, 1)
        
        loss_configs = {
            'mse': {'weight': 1.0, 'params': {}},
            'directional': {'weight': 0.5, 'params': {}}
        }
        
        combined_loss = CombinedLoss(loss_configs)
        
        total_loss, _ = combined_loss(predictions, targets)
        total_loss.backward()
        
        assert predictions.grad is not None
        assert not torch.allclose(predictions.grad, torch.zeros_like(predictions.grad))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])