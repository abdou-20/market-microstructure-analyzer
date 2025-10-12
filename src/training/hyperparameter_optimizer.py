"""
Hyperparameter Optimization for Market Microstructure Models

This module provides comprehensive hyperparameter optimization using Optuna,
with support for multi-objective optimization, early pruning, and advanced
search strategies.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import pickle
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler, CmaEsSampler, QMCSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

from src.training.trainer import ModelTrainer, TrainingConfig
from src.models.transformer_model import create_transformer_model
from src.models.lstm_model import create_lstm_model
from src.training.loss_functions import create_loss_function

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    # Study configuration
    study_name: str = "market_microstructure_optimization"
    storage_url: Optional[str] = None  # For distributed optimization
    n_trials: int = 100
    timeout: Optional[int] = None  # Seconds
    
    # Optimization configuration
    direction: str = "minimize"  # or "maximize" 
    sampler: str = "tpe"  # tpe, cmaes, qmc
    pruner: str = "median"  # median, halving, hyperband
    
    # Multi-objective weights
    loss_weight: float = 0.4
    correlation_weight: float = 0.3
    sharpe_weight: float = 0.2
    stability_weight: float = 0.1
    
    # Training configuration
    max_epochs: int = 50
    patience: int = 10
    min_trials_for_pruning: int = 5
    
    # Data configuration
    cv_splits: int = 3
    test_ratio: float = 0.2
    
    # Search spaces (will be defined per model type)
    search_spaces: Dict[str, Dict[str, Any]] = None
    
    # Output configuration
    output_dir: str = "outputs/optimization"
    save_best_models: bool = True
    save_all_trials: bool = False


class HyperparameterObjective:
    """Objective function for hyperparameter optimization."""
    
    def __init__(self, 
                 model_type: str,
                 data_module,
                 config: OptimizationConfig,
                 device: Optional[torch.device] = None):
        """
        Initialize optimization objective.
        
        Args:
            model_type: Type of model to optimize
            data_module: Data module for training
            config: Optimization configuration
            device: Training device
        """
        self.model_type = model_type
        self.data_module = data_module
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trial tracking
        self.trial_results = []
        self.best_trial_score = float('inf') if config.direction == "minimize" else float('-inf')
        
        logger.info(f"Initialized optimization for {model_type} on {self.device}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters based on model type."""
        
        if self.model_type == "transformer":
            return self._suggest_transformer_params(trial)
        elif self.model_type == "lstm":
            return self._suggest_lstm_params(trial)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _suggest_transformer_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest transformer hyperparameters."""
        
        # Model architecture
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
        num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
        num_layers = trial.suggest_int('num_layers', 2, 8)
        
        # Ensure d_model is divisible by num_heads
        while d_model % num_heads != 0:
            num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
        
        # Regularization
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        
        # Training parameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # Optimizer
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        
        # Scheduler
        scheduler = trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'step', 'none'])
        
        # Loss function
        loss_type = trial.suggest_categorical('loss_function', ['mse', 'huber', 'combined'])
        
        # Loss-specific parameters
        loss_params = {}
        if loss_type == 'huber':
            loss_params['delta'] = trial.suggest_float('huber_delta', 0.1, 2.0)
        elif loss_type == 'combined':
            loss_params['loss_configs'] = {
                'mse': {'weight': trial.suggest_float('mse_weight', 0.1, 1.0), 'params': {}},
                'directional': {
                    'weight': trial.suggest_float('directional_weight', 0.1, 1.0),
                    'params': {'margin': trial.suggest_float('directional_margin', 0.0001, 0.01)}
                },
                'sharpe': {'weight': trial.suggest_float('sharpe_weight', 0.1, 0.5), 'params': {}}
            }
        
        return {
            'model_params': {
                'd_model': d_model,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout,
                'output_size': 1
            },
            'training_params': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'weight_decay': weight_decay,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'loss_function': loss_type,
                'loss_params': loss_params,
                'num_epochs': self.config.max_epochs,
                'patience': self.config.patience
            }
        }
    
    def _suggest_lstm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest LSTM hyperparameters."""
        
        # Model architecture
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        bidirectional = trial.suggest_categorical('bidirectional', [True, False])
        
        # Regularization
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        
        # Training parameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # Optimizer
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        
        # Scheduler
        scheduler = trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'step', 'none'])
        
        # Loss function
        loss_type = trial.suggest_categorical('loss_function', ['mse', 'huber', 'combined'])
        
        # Loss-specific parameters
        loss_params = {}
        if loss_type == 'huber':
            loss_params['delta'] = trial.suggest_float('huber_delta', 0.1, 2.0)
        elif loss_type == 'combined':
            loss_params['loss_configs'] = {
                'mse': {'weight': trial.suggest_float('mse_weight', 0.1, 1.0), 'params': {}},
                'directional': {
                    'weight': trial.suggest_float('directional_weight', 0.1, 1.0),
                    'params': {'margin': trial.suggest_float('directional_margin', 0.0001, 0.01)}
                },
                'sharpe': {'weight': trial.suggest_float('sharpe_weight', 0.1, 0.5), 'params': {}}
            }
        
        return {
            'model_params': {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'bidirectional': bidirectional,
                'dropout': dropout,
                'output_size': 1
            },
            'training_params': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'weight_decay': weight_decay,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'loss_function': loss_type,
                'loss_params': loss_params,
                'num_epochs': self.config.max_epochs,
                'patience': self.config.patience
            }
        }
    
    def create_model(self, model_params: Dict[str, Any]) -> nn.Module:
        """Create model with given parameters."""
        base_config = {
            'input_dim': self.data_module.feature_dim,
            'sequence_length': self.data_module.sequence_length,
            'prediction_horizon': 1
        }
        base_config.update(model_params)
        
        if self.model_type == "transformer":
            return create_transformer_model(base_config)
        elif self.model_type == "lstm":
            return create_lstm_model(base_config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def evaluate_model(self, 
                      model: nn.Module,
                      train_loader,
                      val_loader,
                      training_params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model with given parameters."""
        
        # Create training config
        training_config = {
            'training': training_params,
            'experiment': {
                'output_dir': str(self.output_dir / 'temp_training'),
                'log_to_tensorboard': False
            }
        }
        
        # Create trainer
        trainer = ModelTrainer(model, training_config, self.device, experiment_name="optimization")
        
        try:
            # Train model
            training_history = trainer.train(train_loader, val_loader)
            
            if not training_history:
                return {'loss': float('inf'), 'correlation': 0.0, 'sharpe': 0.0, 'stability': 0.0}
            
            # Get final metrics
            final_metrics = training_history[-1]
            
            # Calculate additional metrics
            model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 2:
                        features, targets = batch
                    else:
                        features, targets, _ = batch
                    
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(features)
                    if isinstance(outputs, dict):
                        predictions = outputs['predictions']
                    else:
                        predictions = outputs
                    
                    val_predictions.append(predictions.cpu().numpy())
                    val_targets.append(targets.cpu().numpy())
            
            val_predictions = np.concatenate(val_predictions, axis=0).flatten()
            val_targets = np.concatenate(val_targets, axis=0).flatten()
            
            # Calculate comprehensive metrics
            mse = mean_squared_error(val_targets, val_predictions)
            mae = mean_absolute_error(val_targets, val_predictions)
            r2 = r2_score(val_targets, val_predictions)
            
            # Correlation
            correlation, _ = pearsonr(val_predictions, val_targets)
            correlation = correlation if not np.isnan(correlation) else 0.0
            
            # Sharpe-like metric
            returns = np.diff(val_predictions)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
            else:
                sharpe = 0.0
            
            # Training stability (inverse of loss variance)
            loss_history = [m.val_loss for m in training_history if m.val_loss is not None]
            if len(loss_history) > 3:
                loss_std = np.std(loss_history[-10:])  # Last 10 epochs
                stability = 1.0 / (1.0 + loss_std)
            else:
                stability = 0.5
            
            metrics = {
                'loss': final_metrics.val_loss or final_metrics.train_loss,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'correlation': abs(correlation),
                'sharpe': sharpe,
                'stability': stability,
                'epochs_trained': len(training_history)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'loss': float('inf'), 'correlation': 0.0, 'sharpe': 0.0, 'stability': 0.0}
        
        finally:
            trainer.cleanup()
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        
        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial)
        
        # Create model
        model = self.create_model(params['model_params'])
        
        # Get data loaders with suggested batch size
        batch_size = params['training_params']['batch_size']
        
        # Update data module batch size
        original_batch_size = self.data_module.batch_size
        self.data_module.batch_size = batch_size
        
        try:
            train_loader = self.data_module.train_dataloader()
            val_loader = self.data_module.val_dataloader()
            
            # Evaluate model
            metrics = self.evaluate_model(model, train_loader, val_loader, params['training_params'])
            
            # Calculate composite score
            composite_score = (
                self.config.loss_weight * metrics['loss'] +
                self.config.correlation_weight * (1.0 - metrics['correlation']) +
                self.config.sharpe_weight * max(0, -metrics['sharpe']) +
                self.config.stability_weight * (1.0 - metrics['stability'])
            )
            
            # Store trial results
            trial_result = {
                'trial_number': trial.number,
                'params': params,
                'metrics': metrics,
                'composite_score': composite_score,
                'timestamp': datetime.now().isoformat()
            }
            self.trial_results.append(trial_result)
            
            # Log intermediate values for pruning
            trial.report(composite_score, step=metrics.get('epochs_trained', 1))
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # Save best model if requested
            if self.config.save_best_models:
                is_best = False
                if self.config.direction == "minimize":
                    is_best = composite_score < self.best_trial_score
                else:
                    is_best = composite_score > self.best_trial_score
                
                if is_best:
                    self.best_trial_score = composite_score
                    model_path = self.output_dir / f"best_model_trial_{trial.number}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'params': params,
                        'metrics': metrics,
                        'trial_number': trial.number
                    }, model_path)
            
            # Log progress
            logger.info(
                f"Trial {trial.number}: Score={composite_score:.6f}, "
                f"Loss={metrics['loss']:.6f}, Corr={metrics['correlation']:.4f}, "
                f"Sharpe={metrics['sharpe']:.4f}"
            )
            
            return composite_score
            
        finally:
            # Restore original batch size
            self.data_module.batch_size = original_batch_size


class HyperparameterOptimizer:
    """Main hyperparameter optimization coordinator."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create study
        self.study = self._create_study()
        
        logger.info(f"Initialized hyperparameter optimizer: {config.study_name}")
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study."""
        
        # Create sampler
        if self.config.sampler == "tpe":
            sampler = TPESampler(seed=42)
        elif self.config.sampler == "cmaes":
            sampler = CmaEsSampler(seed=42)
        elif self.config.sampler == "qmc":
            sampler = QMCSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler: {self.config.sampler}")
        
        # Create pruner
        if self.config.pruner == "median":
            pruner = MedianPruner(n_startup_trials=self.config.min_trials_for_pruning)
        elif self.config.pruner == "halving":
            pruner = SuccessiveHalvingPruner()
        elif self.config.pruner == "hyperband":
            pruner = HyperbandPruner()
        else:
            raise ValueError(f"Unknown pruner: {self.config.pruner}")
        
        # Create study
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.config.storage_url
        )
        
        return study
    
    def optimize(self, 
                model_type: str,
                data_module,
                device: Optional[torch.device] = None) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Args:
            model_type: Type of model to optimize
            data_module: Data module for training
            device: Training device
            
        Returns:
            Completed study
        """
        logger.info(f"Starting optimization for {model_type}")
        start_time = time.time()
        
        # Create objective
        objective = HyperparameterObjective(model_type, data_module, self.config, device)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            callbacks=[self._progress_callback]
        )
        
        optimization_time = time.time() - start_time
        
        # Save results
        self._save_results(objective, optimization_time)
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value: {self.study.best_value:.6f}")
        
        return self.study
    
    def _progress_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Progress callback for optimization."""
        if trial.number % 10 == 0:
            logger.info(f"Completed {trial.number} trials. Best value: {study.best_value:.6f}")
    
    def _save_results(self, objective: HyperparameterObjective, optimization_time: float):
        """Save optimization results."""
        
        # Save study
        study_path = self.output_dir / "study.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        
        # Save trial results
        results_path = self.output_dir / "trial_results.json"
        with open(results_path, 'w') as f:
            json.dump(objective.trial_results, f, indent=2)
        
        # Save summary
        summary = {
            'study_name': self.config.study_name,
            'n_trials': len(self.study.trials),
            'best_trial_number': self.study.best_trial.number,
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'optimization_time': optimization_time,
            'config': self.config.__dict__
        }
        
        summary_path = self.output_dir / "optimization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create trials dataframe and save
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(self.output_dir / "trials_dataframe.csv", index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Optimization history
            values = [trial.value for trial in self.study.trials if trial.value is not None]
            ax1.plot(values)
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Objective Value')
            ax1.set_title('Optimization History')
            ax1.grid(True)
            
            # Parameter importance
            if len(self.study.trials) > 10:
                try:
                    importance = optuna.importance.get_param_importances(self.study)
                    params = list(importance.keys())[:10]  # Top 10
                    importances = list(importance.values())[:10]
                    
                    ax2.barh(params, importances)
                    ax2.set_xlabel('Importance')
                    ax2.set_title('Parameter Importance')
                except Exception:
                    ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                            ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'Not enough trials\nfor importance analysis', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.output_dir / "optimization_history.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization history plot saved to {save_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")


def optimize_model_hyperparameters(model_type: str,
                                 data_module,
                                 config: Optional[OptimizationConfig] = None,
                                 device: Optional[torch.device] = None) -> optuna.Study:
    """
    Convenience function to optimize model hyperparameters.
    
    Args:
        model_type: Type of model to optimize
        data_module: Data module for training
        config: Optimization configuration
        device: Training device
        
    Returns:
        Completed study
    """
    if config is None:
        config = OptimizationConfig()
    
    optimizer = HyperparameterOptimizer(config)
    study = optimizer.optimize(model_type, data_module, device)
    
    # Generate plots
    optimizer.plot_optimization_history()
    
    return study


if __name__ == "__main__":
    # Test hyperparameter optimization
    print("Testing Hyperparameter Optimization...")
    
    # This would normally use real data
    from src.data_processing.data_loader import OrderBookDataModule
    from src.data_processing.order_book_parser import create_synthetic_order_book_data
    from src.data_processing.feature_engineering import FeatureEngineering
    
    # Create synthetic data
    snapshots = create_synthetic_order_book_data(num_snapshots=500)
    feature_engineer = FeatureEngineering(lookback_window=10, prediction_horizon=5)
    feature_vectors = feature_engineer.extract_features(snapshots)
    
    # Create data module
    data_module = OrderBookDataModule(
        feature_vectors=feature_vectors,
        sequence_length=20,
        batch_size=16,
        scaler_type='standard'
    )
    
    # Create optimization config
    config = OptimizationConfig(
        study_name="test_optimization",
        n_trials=5,  # Small number for testing
        max_epochs=3,
        output_dir="test_optimization_results"
    )
    
    # Test optimization
    study = optimize_model_hyperparameters(
        model_type="transformer",
        data_module=data_module,
        config=config
    )
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    
    print("✅ Hyperparameter optimization test passed!")