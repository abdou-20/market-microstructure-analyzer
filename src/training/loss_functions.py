"""
Custom Loss Functions for Market Microstructure Analysis

This module provides specialized loss functions for financial prediction tasks,
including direction-aware losses and risk-adjusted metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DirectionalAccuracyLoss(nn.Module):
    """
    Loss function that focuses on directional accuracy rather than magnitude.
    
    This loss is particularly useful for trading applications where the direction
    of price movement is more important than the exact magnitude.
    """
    
    def __init__(self, 
                 margin: float = 0.0,
                 weight_positive: float = 1.0,
                 weight_negative: float = 1.0):
        """
        Initialize directional accuracy loss.
        
        Args:
            margin: Minimum threshold for directional prediction
            weight_positive: Weight for positive direction predictions
            weight_negative: Weight for negative direction predictions
        """
        super().__init__()
        self.margin = margin
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute directional accuracy loss.
        
        Args:
            predictions: Model predictions
            targets: True targets
            
        Returns:
            Directional accuracy loss
        """
        # Compute signs
        pred_signs = torch.sign(predictions)
        target_signs = torch.sign(targets)
        
        # Apply margin
        pred_signs = torch.where(torch.abs(predictions) < self.margin, 
                               torch.zeros_like(pred_signs), pred_signs)
        target_signs = torch.where(torch.abs(targets) < self.margin, 
                                 torch.zeros_like(target_signs), target_signs)
        
        # Compute directional accuracy
        correct_directions = (pred_signs == target_signs).float()
        
        # Apply class weights
        weights = torch.where(target_signs > 0, self.weight_positive, self.weight_negative)
        weights = torch.where(target_signs == 0, 1.0, weights)  # Neutral weight for zero targets
        
        # Compute weighted loss (we want to minimize incorrect directions)
        loss = weights * (1.0 - correct_directions)
        
        return loss.mean()


class SharpeRatioLoss(nn.Module):
    """
    Loss function based on the Sharpe ratio of predicted returns.
    
    This loss encourages predictions that would lead to higher risk-adjusted returns.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.0,
                 lookback_window: int = 252):
        """
        Initialize Sharpe ratio loss.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation
            lookback_window: Window for computing rolling Sharpe ratio
        """
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.lookback_window = lookback_window
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute negative Sharpe ratio as loss.
        
        Args:
            predictions: Model predictions (expected returns)
            targets: True returns
            
        Returns:
            Negative Sharpe ratio loss
        """
        # Simulate trading returns based on predictions and actual targets
        # Simple strategy: go long if prediction > 0, short if prediction < 0
        position_signs = torch.sign(predictions)
        trading_returns = position_signs * targets
        
        # Remove zero positions
        non_zero_mask = position_signs != 0
        if non_zero_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        active_returns = trading_returns[non_zero_mask]
        
        # Compute Sharpe ratio
        mean_return = active_returns.mean()
        std_return = active_returns.std()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        sharpe_ratio = (mean_return - self.risk_free_rate) / (std_return + epsilon)
        
        # Return negative Sharpe ratio as loss (we want to maximize Sharpe)
        return -sharpe_ratio


class HuberLoss(nn.Module):
    """
    Huber loss that is more robust to outliers than MSE.
    
    Combines the best properties of MSE and MAE losses.
    """
    
    def __init__(self, delta: float = 1.0):
        """
        Initialize Huber loss.
        
        Args:
            delta: Threshold for switching between MSE and MAE
        """
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss.
        
        Args:
            predictions: Model predictions
            targets: True targets
            
        Returns:
            Huber loss
        """
        residual = torch.abs(predictions - targets)
        condition = residual < self.delta
        
        loss = torch.where(condition,
                          0.5 * residual ** 2,
                          self.delta * residual - 0.5 * self.delta ** 2)
        
        return loss.mean()


class QuantileLoss(nn.Module):
    """
    Quantile loss for probabilistic predictions.
    
    Useful for predicting confidence intervals and risk estimates.
    """
    
    def __init__(self, quantile: float = 0.5):
        """
        Initialize quantile loss.
        
        Args:
            quantile: Target quantile (0.5 for median)
        """
        super().__init__()
        assert 0.0 < quantile < 1.0
        self.quantile = quantile
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            predictions: Model predictions
            targets: True targets
            
        Returns:
            Quantile loss
        """
        residual = targets - predictions
        loss = torch.where(residual > 0,
                          self.quantile * residual,
                          (self.quantile - 1) * residual)
        
        return loss.mean()


class TrendAwareLoss(nn.Module):
    """
    Loss function that is aware of market trends and adapts accordingly.
    
    Penalizes more heavily when predictions go against strong trends.
    """
    
    def __init__(self, 
                 trend_window: int = 10,
                 trend_threshold: float = 0.01,
                 trend_penalty: float = 2.0):
        """
        Initialize trend-aware loss.
        
        Args:
            trend_window: Window for computing trend
            trend_threshold: Threshold for detecting strong trends
            trend_penalty: Penalty multiplier for going against trends
        """
        super().__init__()
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.trend_penalty = trend_penalty
        
        # Base loss
        self.base_loss = nn.MSELoss(reduction='none')
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                historical_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute trend-aware loss.
        
        Args:
            predictions: Model predictions
            targets: True targets
            historical_targets: Historical target values for trend computation
            
        Returns:
            Trend-aware loss
        """
        base_loss = self.base_loss(predictions, targets)
        
        if historical_targets is None:
            return base_loss.mean()
        
        # Compute trend from historical data
        if historical_targets.size(0) < self.trend_window:
            return base_loss.mean()
        
        recent_targets = historical_targets[-self.trend_window:]
        trend = (recent_targets[-1] - recent_targets[0]) / self.trend_window
        
        # Detect strong trends
        strong_uptrend = trend > self.trend_threshold
        strong_downtrend = trend < -self.trend_threshold
        
        # Apply penalties
        penalty_weights = torch.ones_like(base_loss)
        
        if strong_uptrend:
            # Penalize predictions that go against uptrend
            wrong_direction = predictions < 0
            penalty_weights[wrong_direction] *= self.trend_penalty
        elif strong_downtrend:
            # Penalize predictions that go against downtrend
            wrong_direction = predictions > 0
            penalty_weights[wrong_direction] *= self.trend_penalty
        
        weighted_loss = base_loss * penalty_weights
        return weighted_loss.mean()


class VolatilityAwareLoss(nn.Module):
    """
    Loss function that adjusts based on market volatility.
    
    More lenient during high volatility periods, stricter during low volatility.
    """
    
    def __init__(self, 
                 volatility_window: int = 20,
                 base_loss_type: str = 'mse'):
        """
        Initialize volatility-aware loss.
        
        Args:
            volatility_window: Window for computing volatility
            base_loss_type: Base loss function ('mse', 'mae', 'huber')
        """
        super().__init__()
        self.volatility_window = volatility_window
        
        if base_loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif base_loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        elif base_loss_type == 'huber':
            self.base_loss = HuberLoss()
        else:
            raise ValueError(f"Unknown base loss type: {base_loss_type}")
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                historical_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute volatility-aware loss.
        
        Args:
            predictions: Model predictions
            targets: True targets
            historical_targets: Historical target values for volatility computation
            
        Returns:
            Volatility-aware loss
        """
        if isinstance(self.base_loss, HuberLoss):
            base_loss = self.base_loss(predictions, targets)
            base_loss = base_loss.expand_as(predictions)  # Ensure same shape
        else:
            base_loss = self.base_loss(predictions, targets)
        
        if historical_targets is None:
            return base_loss.mean()
        
        # Compute volatility from historical data
        if historical_targets.size(0) < self.volatility_window:
            return base_loss.mean()
        
        recent_targets = historical_targets[-self.volatility_window:]
        volatility = recent_targets.std()
        
        # Adjust loss based on volatility
        # Higher volatility -> lower penalty (more forgiving)
        # Lower volatility -> higher penalty (more strict)
        volatility_adjustment = 1.0 / (1.0 + volatility.item())
        
        adjusted_loss = base_loss * volatility_adjustment
        return adjusted_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function that combines multiple loss types.
    
    Allows for multi-objective optimization with different loss components.
    """
    
    def __init__(self, loss_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize combined loss.
        
        Args:
            loss_configs: Dictionary of loss configurations
                Example: {
                    'mse': {'weight': 1.0, 'params': {}},
                    'directional': {'weight': 0.5, 'params': {'margin': 0.001}},
                    'sharpe': {'weight': 0.3, 'params': {}}
                }
        """
        super().__init__()
        
        self.loss_functions = nn.ModuleDict()
        self.loss_weights = {}
        
        for loss_name, config in loss_configs.items():
            weight = config.get('weight', 1.0)
            params = config.get('params', {})
            
            self.loss_weights[loss_name] = weight
            
            if loss_name == 'mse':
                self.loss_functions[loss_name] = nn.MSELoss()
            elif loss_name == 'mae':
                self.loss_functions[loss_name] = nn.L1Loss()
            elif loss_name == 'huber':
                self.loss_functions[loss_name] = HuberLoss(**params)
            elif loss_name == 'directional':
                self.loss_functions[loss_name] = DirectionalAccuracyLoss(**params)
            elif loss_name == 'sharpe':
                self.loss_functions[loss_name] = SharpeRatioLoss(**params)
            elif loss_name == 'quantile':
                self.loss_functions[loss_name] = QuantileLoss(**params)
            elif loss_name == 'trend_aware':
                self.loss_functions[loss_name] = TrendAwareLoss(**params)
            elif loss_name == 'volatility_aware':
                self.loss_functions[loss_name] = VolatilityAwareLoss(**params)
            else:
                raise ValueError(f"Unknown loss function: {loss_name}")
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            targets: True targets
            **kwargs: Additional arguments for specific loss functions
            
        Returns:
            Tuple of (total_loss, individual_losses)
        """
        total_loss = 0.0
        individual_losses = {}
        
        for loss_name, loss_fn in self.loss_functions.items():
            weight = self.loss_weights[loss_name]
            
            # Compute individual loss
            if loss_name in ['trend_aware', 'volatility_aware']:
                loss_value = loss_fn(predictions, targets, **kwargs)
            else:
                loss_value = loss_fn(predictions, targets)
            
            individual_losses[loss_name] = loss_value
            total_loss += weight * loss_value
        
        return total_loss, individual_losses


def create_loss_function(loss_config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss functions from configuration.
    
    Args:
        loss_config: Loss function configuration
        
    Returns:
        Loss function instance
    """
    loss_type = loss_config.get('loss_function', 'mse')
    loss_params = loss_config.get('loss_params', {})
    
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return HuberLoss(**loss_params)
    elif loss_type == 'directional':
        return DirectionalAccuracyLoss(**loss_params)
    elif loss_type == 'sharpe':
        return SharpeRatioLoss(**loss_params)
    elif loss_type == 'quantile':
        return QuantileLoss(**loss_params)
    elif loss_type == 'trend_aware':
        return TrendAwareLoss(**loss_params)
    elif loss_type == 'volatility_aware':
        return VolatilityAwareLoss(**loss_params)
    elif loss_type == 'combined':
        loss_configs = loss_params.get('loss_configs', {})
        return CombinedLoss(loss_configs)
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    
    batch_size = 32
    predictions = torch.randn(batch_size, 1) * 0.01  # Small price movements
    targets = torch.randn(batch_size, 1) * 0.01
    
    print("Testing loss functions...")
    
    # Test MSE
    mse_loss = nn.MSELoss()
    mse_value = mse_loss(predictions, targets)
    print(f"MSE Loss: {mse_value.item():.6f}")
    
    # Test Directional Accuracy Loss
    directional_loss = DirectionalAccuracyLoss(margin=0.001)
    dir_value = directional_loss(predictions, targets)
    print(f"Directional Loss: {dir_value.item():.6f}")
    
    # Test Sharpe Ratio Loss
    sharpe_loss = SharpeRatioLoss()
    sharpe_value = sharpe_loss(predictions, targets)
    print(f"Sharpe Loss: {sharpe_value.item():.6f}")
    
    # Test Huber Loss
    huber_loss = HuberLoss(delta=0.005)
    huber_value = huber_loss(predictions, targets)
    print(f"Huber Loss: {huber_value.item():.6f}")
    
    # Test Quantile Loss
    quantile_loss = QuantileLoss(quantile=0.5)
    quantile_value = quantile_loss(predictions, targets)
    print(f"Quantile Loss: {quantile_value.item():.6f}")
    
    # Test Combined Loss
    loss_configs = {
        'mse': {'weight': 1.0, 'params': {}},
        'directional': {'weight': 0.5, 'params': {'margin': 0.001}},
        'sharpe': {'weight': 0.3, 'params': {}}
    }
    combined_loss = CombinedLoss(loss_configs)
    total_loss, individual_losses = combined_loss(predictions, targets)
    
    print(f"Combined Loss: {total_loss.item():.6f}")
    print("Individual losses:")
    for name, loss_val in individual_losses.items():
        print(f"  {name}: {loss_val.item():.6f}")
    
    print("\n✅ Loss functions test passed!")