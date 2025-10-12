"""
Validation Framework for Market Microstructure Models

This module provides comprehensive validation strategies including
walk-forward validation, cross-validation, and performance evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing.data_loader import WalkForwardDataModule, OrderBookDataModule
from utils.logger import get_experiment_logger

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    mse: float
    mae: float
    rmse: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mse': self.mse,
            'mae': self.mae,
            'rmse': self.rmse,
            'directional_accuracy': self.directional_accuracy,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'calmar_ratio': self.calmar_ratio
        }


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for trading models.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize performance evaluator.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def evaluate_predictions(self, 
                           predictions: np.ndarray, 
                           targets: np.ndarray,
                           timestamps: Optional[np.ndarray] = None) -> ValidationMetrics:
        """
        Evaluate model predictions comprehensively.
        
        Args:
            predictions: Model predictions
            targets: True targets
            timestamps: Optional timestamps for time-series analysis
            
        Returns:
            ValidationMetrics object
        """
        # Basic regression metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        pred_directions = np.sign(predictions)
        target_directions = np.sign(targets)
        directional_accuracy = np.mean(pred_directions == target_directions)
        
        # Trading performance metrics
        trading_returns = self._simulate_trading_returns(predictions, targets)
        sharpe_ratio = self._calculate_sharpe_ratio(trading_returns)
        max_drawdown = self._calculate_max_drawdown(trading_returns)
        win_rate = self._calculate_win_rate(trading_returns)
        profit_factor = self._calculate_profit_factor(trading_returns)
        calmar_ratio = self._calculate_calmar_ratio(trading_returns, max_drawdown)
        
        return ValidationMetrics(
            mse=mse,
            mae=mae,
            rmse=rmse,
            directional_accuracy=directional_accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio
        )
    
    def _simulate_trading_returns(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Simulate trading returns based on predictions and actual outcomes.
        
        Args:
            predictions: Model predictions
            targets: True targets
            
        Returns:
            Array of trading returns
        """
        # Simple strategy: go long if prediction > 0, short if prediction < 0
        position_signs = np.sign(predictions)
        trading_returns = position_signs * targets
        
        # Only consider non-zero positions
        non_zero_mask = position_signs != 0
        if non_zero_mask.sum() == 0:
            return np.array([0.0])
        
        return trading_returns[non_zero_mask]
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.abs(np.min(drawdown))
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate (percentage of profitable trades)."""
        if len(returns) == 0:
            return 0.0
        
        winning_trades = np.sum(returns > 0)
        return winning_trades / len(returns)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = np.abs(np.sum(returns[returns < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) == 0 or max_drawdown == 0:
            return 0.0
        
        annualized_return = np.mean(returns) * 252
        return annualized_return / max_drawdown


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    
    This validator implements realistic backtesting by training on historical data
    and testing on future periods, advancing the window through time.
    """
    
    def __init__(self,
                 model_class: type,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initialize walk-forward validator.
        
        Args:
            model_class: Model class to instantiate
            config: Configuration dictionary
            device: Training device
        """
        self.model_class = model_class
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = PerformanceEvaluator()
        
        # Results storage
        self.validation_results = []
        self.models = []
        
    def validate(self, 
                 data_module: WalkForwardDataModule,
                 num_splits: Optional[int] = None) -> List[ValidationMetrics]:
        """
        Perform walk-forward validation.
        
        Args:
            data_module: Walk-forward data module
            num_splits: Number of splits to validate (None for all)
            
        Returns:
            List of validation metrics for each split
        """
        total_splits = data_module.get_num_splits()
        if num_splits is None:
            num_splits = total_splits
        
        num_splits = min(num_splits, total_splits)
        
        logger.info(f"Starting walk-forward validation with {num_splits} splits")
        
        for split_idx in range(num_splits):
            logger.info(f"Processing split {split_idx + 1}/{num_splits}")
            
            # Get data for this split
            train_loader, test_loader = data_module.get_split(split_idx)
            split_info = data_module.get_split_info(split_idx)
            
            # Create fresh model for this split
            model = self.model_class(self.config)
            model.to(self.device)
            
            # Train model
            trained_model = self._train_model(model, train_loader, split_idx)
            
            # Evaluate model
            metrics = self._evaluate_model(trained_model, test_loader, split_idx)
            
            # Store results
            self.validation_results.append({
                'split_idx': split_idx,
                'split_info': split_info,
                'metrics': metrics
            })
            self.models.append(trained_model)
            
            logger.info(f"Split {split_idx + 1} - Metrics: {metrics.to_dict()}")
        
        return [result['metrics'] for result in self.validation_results]
    
    def _train_model(self, 
                     model: nn.Module, 
                     train_loader,
                     split_idx: int) -> nn.Module:
        """Train model for one split."""
        from training.trainer import ModelTrainer
        
        # Create trainer with minimal epochs for walk-forward
        config = self.config.copy()
        config['training']['num_epochs'] = config.get('training', {}).get('walk_forward_epochs', 20)
        config['training']['early_stopping'] = True
        config['training']['patience'] = 5
        
        trainer = ModelTrainer(
            model, 
            config, 
            device=self.device,
            experiment_name=f"walk_forward_split_{split_idx}"
        )
        
        # Train model
        trainer.train(train_loader, num_epochs=config['training']['num_epochs'])
        
        # Cleanup trainer resources
        trainer.cleanup()
        
        return model
    
    def _evaluate_model(self, 
                       model: nn.Module, 
                       test_loader,
                       split_idx: int) -> ValidationMetrics:
        """Evaluate model on test set."""
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 2:
                    features, targets = batch
                else:
                    features, targets, _ = batch
                
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                output = model(features)
                predictions = output['predictions']
                
                all_predictions.append(predictions.cpu().numpy().flatten())
                all_targets.append(targets.cpu().numpy().flatten())
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        # Evaluate performance
        return self.evaluator.evaluate_predictions(predictions, targets)
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across all splits."""
        if not self.validation_results:
            return {}
        
        metrics_list = [result['metrics'] for result in self.validation_results]
        
        # Calculate mean and std for each metric
        aggregate_metrics = {}
        metric_names = metrics_list[0].to_dict().keys()
        
        for metric_name in metric_names:
            values = [metrics.to_dict()[metric_name] for metrics in metrics_list]
            aggregate_metrics[f'{metric_name}_mean'] = np.mean(values)
            aggregate_metrics[f'{metric_name}_std'] = np.std(values)
            aggregate_metrics[f'{metric_name}_min'] = np.min(values)
            aggregate_metrics[f'{metric_name}_max'] = np.max(values)
        
        return aggregate_metrics
    
    def save_results(self, output_path: Union[str, Path]):
        """Save validation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for saving
        results_data = []
        for result in self.validation_results:
            split_data = {
                'split_idx': result['split_idx'],
                'split_info': result['split_info'],
                'metrics': result['metrics'].to_dict()
            }
            results_data.append(split_data)
        
        # Add aggregate metrics
        aggregate_metrics = self.get_aggregate_metrics()
        
        # Save to JSON
        import json
        with open(output_path, 'w') as f:
            json.dump({
                'validation_results': results_data,
                'aggregate_metrics': aggregate_metrics,
                'config': self.config
            }, f, indent=2, default=str)
        
        logger.info(f"Saved validation results to {output_path}")


class CrossValidator:
    """
    Cross-validation for model evaluation.
    
    Note: Traditional k-fold cross-validation is not appropriate for time series
    due to temporal dependencies. This class provides time-aware alternatives.
    """
    
    def __init__(self,
                 model_class: type,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initialize cross-validator.
        
        Args:
            model_class: Model class to instantiate
            config: Configuration dictionary
            device: Training device
        """
        self.model_class = model_class
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = PerformanceEvaluator()
    
    def time_series_split_validation(self,
                                   data_module: OrderBookDataModule,
                                   n_splits: int = 5) -> List[ValidationMetrics]:
        """
        Perform time series split validation.
        
        This method splits data into n_splits where each split uses increasingly
        more data for training and tests on the next time period.
        
        Args:
            data_module: Data module with time series data
            n_splits: Number of splits
            
        Returns:
            List of validation metrics
        """
        logger.info(f"Starting time series split validation with {n_splits} splits")
        
        # Get all data
        all_features = []
        all_targets = []
        all_timestamps = []
        
        # Combine train, val, test data
        for loader in [data_module.train_dataloader(shuffle=False), 
                      data_module.val_dataloader(), 
                      data_module.test_dataloader()]:
            for batch in loader:
                if len(batch) == 2:
                    features, targets = batch
                    timestamps = None
                else:
                    features, targets, metadata = batch
                    timestamps = [m.get('target_timestamp') for m in metadata] if metadata else None
                
                all_features.append(features)
                all_targets.append(targets)
                if timestamps:
                    all_timestamps.extend(timestamps)
        
        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Create time-ordered splits
        total_samples = len(all_features)
        split_size = total_samples // n_splits
        
        validation_results = []
        
        for split_idx in range(n_splits - 1):  # n_splits - 1 because we need data for testing
            # Define training and testing indices
            train_end = (split_idx + 1) * split_size
            test_start = train_end
            test_end = min(test_start + split_size, total_samples)
            
            if test_end <= test_start:
                break
            
            # Create data loaders for this split
            train_features = all_features[:train_end]
            train_targets = all_targets[:train_end]
            test_features = all_features[test_start:test_end]
            test_targets = all_targets[test_start:test_end]
            
            # Create datasets and loaders
            from torch.utils.data import TensorDataset, DataLoader
            
            train_dataset = TensorDataset(train_features, train_targets)
            test_dataset = TensorDataset(test_features, test_targets)
            
            batch_size = self.config.get('training', {}).get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Train and evaluate model
            model = self.model_class(self.config)
            model.to(self.device)
            
            trained_model = self._train_model(model, train_loader, split_idx)
            metrics = self._evaluate_model(trained_model, test_loader, split_idx)
            
            validation_results.append(metrics)
            
            logger.info(f"Split {split_idx + 1}/{n_splits - 1} - Metrics: {metrics.to_dict()}")
        
        return validation_results
    
    def _train_model(self, model: nn.Module, train_loader, split_idx: int) -> nn.Module:
        """Train model for one split."""
        from training.trainer import ModelTrainer
        
        # Create trainer with reduced epochs for cross-validation
        config = self.config.copy()
        config['training']['num_epochs'] = config.get('training', {}).get('cv_epochs', 30)
        
        trainer = ModelTrainer(
            model, 
            config, 
            device=self.device,
            experiment_name=f"cv_split_{split_idx}"
        )
        
        # Train model
        trainer.train(train_loader, num_epochs=config['training']['num_epochs'])
        trainer.cleanup()
        
        return model
    
    def _evaluate_model(self, model: nn.Module, test_loader, split_idx: int) -> ValidationMetrics:
        """Evaluate model on test set."""
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                features, targets = batch
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                output = model(features)
                predictions = output['predictions']
                
                all_predictions.append(predictions.cpu().numpy().flatten())
                all_targets.append(targets.cpu().numpy().flatten())
        
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        return self.evaluator.evaluate_predictions(predictions, targets)


def plot_validation_results(validation_results: List[ValidationMetrics], 
                          output_path: Optional[Union[str, Path]] = None):
    """
    Plot validation results.
    
    Args:
        validation_results: List of validation metrics
        output_path: Optional path to save plot
    """
    if not validation_results:
        logger.warning("No validation results to plot")
        return
    
    # Convert to DataFrame
    results_data = [metrics.to_dict() for metrics in validation_results]
    df = pd.DataFrame(results_data)
    df['split'] = range(len(df))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Validation Results Across Splits', fontsize=16)
    
    # Plot metrics
    axes[0, 0].plot(df['split'], df['directional_accuracy'], marker='o')
    axes[0, 0].set_title('Directional Accuracy')
    axes[0, 0].set_xlabel('Split')
    axes[0, 0].set_ylabel('Accuracy')
    
    axes[0, 1].plot(df['split'], df['sharpe_ratio'], marker='o', color='orange')
    axes[0, 1].set_title('Sharpe Ratio')
    axes[0, 1].set_xlabel('Split')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    
    axes[1, 0].plot(df['split'], df['max_drawdown'], marker='o', color='red')
    axes[1, 0].set_title('Maximum Drawdown')
    axes[1, 0].set_xlabel('Split')
    axes[1, 0].set_ylabel('Max Drawdown')
    
    axes[1, 1].plot(df['split'], df['win_rate'], marker='o', color='green')
    axes[1, 1].set_title('Win Rate')
    axes[1, 1].set_xlabel('Split')
    axes[1, 1].set_ylabel('Win Rate')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved validation plot to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test validation framework
    from ..models.transformer_model import create_transformer_model
    from ..data_processing.data_loader import OrderBookDataModule
    from ..data_processing.order_book_parser import create_synthetic_order_book_data
    from ..data_processing.feature_engineering import FeatureEngineering
    
    print("Testing validation framework...")
    
    # Create synthetic data
    snapshots = create_synthetic_order_book_data(num_snapshots=1000)
    feature_engineer = FeatureEngineering(lookback_window=10, prediction_horizon=5)
    feature_vectors = feature_engineer.extract_features(snapshots)
    
    # Create data module
    data_module = OrderBookDataModule(
        feature_vectors=feature_vectors,
        sequence_length=20,
        batch_size=16,
        scaler_type='standard'
    )
    
    # Test config
    config = {
        'input_dim': 46,
        'model': {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'output_size': 1
        },
        'training': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 16,
            'num_epochs': 3,
            'cv_epochs': 2,
            'loss_function': 'mse'
        }
    }
    
    # Test performance evaluator
    evaluator = PerformanceEvaluator()
    predictions = np.random.randn(100) * 0.01
    targets = np.random.randn(100) * 0.01
    
    metrics = evaluator.evaluate_predictions(predictions, targets)
    print(f"Test metrics: {metrics.to_dict()}")
    
    # Test cross-validator
    cross_validator = CrossValidator(
        model_class=lambda config: create_transformer_model(config),
        config=config
    )
    
    cv_results = cross_validator.time_series_split_validation(data_module, n_splits=3)
    print(f"Cross-validation completed with {len(cv_results)} splits")
    
    for i, result in enumerate(cv_results):
        print(f"Split {i + 1}: Directional Accuracy = {result.directional_accuracy:.4f}")
    
    print("✅ Validation framework test passed!")