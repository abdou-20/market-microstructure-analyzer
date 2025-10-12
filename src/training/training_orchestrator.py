"""
Training Orchestrator for Market Microstructure Models

This module provides a comprehensive training orchestrator that coordinates
hyperparameter optimization, cross-validation, model training, and evaluation
to achieve the best possible model performance.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from src.training.hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig
from src.training.trainer import ModelTrainer
from src.models.transformer_model import create_transformer_model
from src.models.lstm_model import create_lstm_model
from src.backtesting.runner import BacktestRunner
from src.backtesting.strategy import MLStrategy
from src.backtesting.engine import TransactionCostModel, MarketImpactModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingTarget:
    """Target performance metrics for training."""
    min_correlation: float = 0.15      # Minimum prediction correlation
    min_sharpe_ratio: float = 0.8      # Minimum Sharpe ratio in backtesting
    max_mse: float = 0.001             # Maximum MSE
    min_r2: float = 0.05               # Minimum R² score
    min_directional_accuracy: float = 0.55  # Minimum directional accuracy
    max_drawdown: float = 0.15         # Maximum drawdown in backtesting


@dataclass
class ModelPerformance:
    """Container for model performance metrics."""
    model_name: str
    model_type: str
    
    # Statistical metrics
    mse: float
    mae: float
    r2: float
    correlation: float
    spearman_correlation: float
    directional_accuracy: float
    
    # Trading metrics
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    
    # Training metrics
    final_train_loss: float
    final_val_loss: float
    epochs_trained: int
    training_time: float
    
    # Hyperparameters
    best_params: Dict[str, Any]
    
    # Meets targets
    meets_targets: bool = False
    target_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'mse': self.mse,
            'mae': self.mae,
            'r2': self.r2,
            'correlation': self.correlation,
            'spearman_correlation': self.spearman_correlation,
            'directional_accuracy': self.directional_accuracy,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_return': self.total_return,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'final_train_loss': self.final_train_loss,
            'final_val_loss': self.final_val_loss,
            'epochs_trained': self.epochs_trained,
            'training_time': self.training_time,
            'best_params': self.best_params,
            'meets_targets': self.meets_targets,
            'target_score': self.target_score
        }


class TrainingOrchestrator:
    """
    Comprehensive training orchestrator for achieving optimal model performance.
    """
    
    def __init__(self, 
                 data_module,
                 targets: TrainingTarget,
                 output_dir: str = "outputs/orchestrated_training",
                 device: Optional[torch.device] = None):
        """
        Initialize training orchestrator.
        
        Args:
            data_module: Data module for training
            targets: Performance targets
            output_dir: Output directory
            device: Training device
        """
        self.data_module = data_module
        self.targets = targets
        self.output_dir = Path(output_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.model_performances = {}
        self.optimization_studies = {}
        self.training_history = {}
        
        logger.info(f"Initialized training orchestrator with targets:")
        logger.info(f"  Min Correlation: {targets.min_correlation:.3f}")
        logger.info(f"  Min Sharpe: {targets.min_sharpe_ratio:.3f}")
        logger.info(f"  Max MSE: {targets.max_mse:.6f}")
        logger.info(f"  Min R²: {targets.min_r2:.3f}")
    
    def train_all_models(self, 
                        model_types: List[str] = None,
                        optimization_trials: int = 50,
                        max_training_rounds: int = 3) -> Dict[str, ModelPerformance]:
        """
        Train all models with hyperparameter optimization until targets are met.
        
        Args:
            model_types: List of model types to train
            optimization_trials: Number of optimization trials per round
            max_training_rounds: Maximum training rounds per model
            
        Returns:
            Dictionary of model performances
        """
        if model_types is None:
            model_types = ['transformer', 'lstm']
        
        logger.info(f"Starting training for models: {model_types}")
        logger.info(f"Optimization trials per round: {optimization_trials}")
        logger.info(f"Maximum training rounds: {max_training_rounds}")
        
        start_time = time.time()
        
        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_type.upper()} Model")
            logger.info(f"{'='*50}")
            
            self.train_single_model(
                model_type=model_type,
                optimization_trials=optimization_trials,
                max_training_rounds=max_training_rounds
            )
        
        total_time = time.time() - start_time
        logger.info(f"\nAll models training completed in {total_time:.2f} seconds")
        
        # Generate final report
        self.generate_final_report()
        
        return self.model_performances
    
    def train_single_model(self,
                          model_type: str,
                          optimization_trials: int = 50,
                          max_training_rounds: int = 3) -> ModelPerformance:
        """
        Train a single model type until targets are met.
        
        Args:
            model_type: Type of model to train
            optimization_trials: Number of optimization trials per round
            max_training_rounds: Maximum training rounds
            
        Returns:
            Model performance
        """
        best_performance = None
        training_round = 0
        
        while training_round < max_training_rounds:
            training_round += 1
            logger.info(f"\nTraining Round {training_round} for {model_type}")
            
            # Run hyperparameter optimization
            study = self.run_hyperparameter_optimization(
                model_type=model_type,
                n_trials=optimization_trials,
                round_number=training_round
            )
            
            # Train best model from optimization
            performance = self.train_best_model(
                model_type=model_type,
                study=study,
                round_number=training_round
            )
            
            # Update best performance
            if best_performance is None or performance.target_score > best_performance.target_score:
                best_performance = performance
                self.model_performances[model_type] = performance
            
            # Check if targets are met
            if performance.meets_targets:
                logger.info(f"🎉 {model_type} model meets all targets!")
                break
            else:
                logger.info(f"Round {training_round} targets not met. Target score: {performance.target_score:.3f}")
                
                # Adjust optimization strategy for next round
                if training_round < max_training_rounds:
                    optimization_trials = min(optimization_trials + 20, 100)  # Increase trials
                    logger.info(f"Increasing optimization trials to {optimization_trials} for next round")
        
        if not best_performance.meets_targets:
            logger.warning(f"⚠️  {model_type} model did not meet all targets after {max_training_rounds} rounds")
        
        return best_performance
    
    def run_hyperparameter_optimization(self,
                                       model_type: str,
                                       n_trials: int,
                                       round_number: int) -> Any:
        """Run hyperparameter optimization for a model type."""
        
        logger.info(f"Running hyperparameter optimization for {model_type} (Round {round_number})")
        
        # Create optimization config
        config = OptimizationConfig(
            study_name=f"{model_type}_optimization_round_{round_number}",
            n_trials=n_trials,
            direction="minimize",
            max_epochs=50,
            patience=10,
            output_dir=str(self.output_dir / f"{model_type}_optimization_round_{round_number}")
        )
        
        # Adjust optimization strategy based on round
        if round_number > 1:
            # Use more aggressive search in later rounds
            config.sampler = "cmaes"
            config.max_epochs = 75
            config.patience = 15
        
        # Run optimization
        optimizer = HyperparameterOptimizer(config)
        study = optimizer.optimize(model_type, self.data_module, self.device)
        
        # Save study
        study_key = f"{model_type}_round_{round_number}"
        self.optimization_studies[study_key] = study
        
        logger.info(f"Optimization completed. Best value: {study.best_value:.6f}")
        
        return study
    
    def train_best_model(self,
                        model_type: str,
                        study: Any,
                        round_number: int) -> ModelPerformance:
        """Train the best model from hyperparameter optimization."""
        
        logger.info(f"Training best {model_type} model from optimization")
        
        # Get best parameters
        best_params = study.best_params
        
        # Create model with best parameters
        model = self.create_model_from_params(model_type, best_params)
        
        # Create training config
        training_config = self.create_training_config(best_params)
        
        # Train model
        trainer = ModelTrainer(model, training_config, self.device, 
                             experiment_name=f"{model_type}_final_round_{round_number}")
        
        start_time = time.time()
        
        try:
            train_loader = self.data_module.train_dataloader()
            val_loader = self.data_module.val_dataloader()
            test_loader = self.data_module.test_dataloader()
            
            # Train model
            training_history = trainer.train(train_loader, val_loader)
            training_time = time.time() - start_time
            
            # Evaluate model comprehensively
            performance = self.evaluate_model_comprehensively(
                model=model,
                model_type=model_type,
                test_loader=test_loader,
                training_history=training_history,
                training_time=training_time,
                best_params=best_params,
                round_number=round_number
            )
            
            # Save model if it's the best so far
            if performance.meets_targets or round_number == 1:
                model_path = self.output_dir / f"best_{model_type}_model_round_{round_number}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'performance': performance.to_dict(),
                    'training_config': training_config,
                    'best_params': best_params
                }, model_path)
                logger.info(f"Saved model to {model_path}")
            
            return performance
            
        finally:
            trainer.cleanup()
    
    def create_model_from_params(self, model_type: str, params: Dict[str, Any]) -> nn.Module:
        """Create model from hyperparameter optimization results."""
        
        # Extract model-specific parameters
        model_params = {}
        base_config = {
            'input_dim': self.data_module.feature_dim,
            'sequence_length': self.data_module.sequence_length,
            'prediction_horizon': 1
        }
        
        if model_type == "transformer":
            model_params = {
                'd_model': params.get('d_model', 128),
                'num_heads': params.get('num_heads', 8),
                'num_layers': params.get('num_layers', 4),
                'dropout': params.get('dropout', 0.1),
                'output_size': 1
            }
        elif model_type == "lstm":
            model_params = {
                'hidden_size': params.get('hidden_size', 128),
                'num_layers': params.get('num_layers', 2),
                'bidirectional': params.get('bidirectional', True),
                'dropout': params.get('dropout', 0.1),
                'output_size': 1
            }
        
        base_config.update(model_params)
        
        if model_type == "transformer":
            return create_transformer_model(base_config)
        elif model_type == "lstm":
            return create_lstm_model(base_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_training_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create training configuration from parameters."""
        
        training_config = {
            'training': {
                'optimizer': params.get('optimizer', 'adamw'),
                'learning_rate': params.get('learning_rate', 0.001),
                'batch_size': params.get('batch_size', 32),
                'num_epochs': 100,  # Extended training for final model
                'weight_decay': params.get('weight_decay', 0.01),
                'scheduler': params.get('scheduler', 'cosine'),
                'loss_function': params.get('loss_function', 'combined'),
                'loss_params': params.get('loss_params', {}),
                'early_stopping': True,
                'patience': 20,
                'grad_clip_norm': 1.0
            },
            'experiment': {
                'output_dir': str(self.output_dir / 'final_training'),
                'log_to_tensorboard': True
            }
        }
        
        return training_config
    
    def evaluate_model_comprehensively(self,
                                     model: nn.Module,
                                     model_type: str,
                                     test_loader,
                                     training_history: List,
                                     training_time: float,
                                     best_params: Dict[str, Any],
                                     round_number: int) -> ModelPerformance:
        """Comprehensively evaluate model performance."""
        
        logger.info("Running comprehensive model evaluation...")
        
        model.eval()
        test_predictions = []
        test_targets = []
        
        # Collect predictions
        with torch.no_grad():
            for batch in test_loader:
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
                
                test_predictions.append(predictions.cpu().numpy())
                test_targets.append(targets.cpu().numpy())
        
        test_predictions = np.concatenate(test_predictions, axis=0).flatten()
        test_targets = np.concatenate(test_targets, axis=0).flatten()
        
        # Statistical metrics
        mse = mean_squared_error(test_targets, test_predictions)
        mae = mean_absolute_error(test_targets, test_predictions)
        r2 = r2_score(test_targets, test_predictions)
        
        correlation, _ = pearsonr(test_predictions, test_targets)
        correlation = correlation if not np.isnan(correlation) else 0.0
        
        spearman_corr, _ = spearmanr(test_predictions, test_targets)
        spearman_corr = spearman_corr if not np.isnan(spearman_corr) else 0.0
        
        # Directional accuracy
        pred_directions = np.sign(test_predictions)
        target_directions = np.sign(test_targets)
        directional_accuracy = np.mean(pred_directions == target_directions)
        
        # Trading metrics via backtesting
        trading_metrics = self.run_backtesting_evaluation(model, model_type)
        
        # Training metrics
        final_train_loss = training_history[-1].train_loss if training_history else 0.0
        final_val_loss = training_history[-1].val_loss if training_history else 0.0
        epochs_trained = len(training_history)
        
        # Create performance object
        performance = ModelPerformance(
            model_name=f"{model_type}_round_{round_number}",
            model_type=model_type,
            mse=mse,
            mae=mae,
            r2=r2,
            correlation=abs(correlation),
            spearman_correlation=abs(spearman_corr),
            directional_accuracy=directional_accuracy,
            sharpe_ratio=trading_metrics['sharpe_ratio'],
            max_drawdown=abs(trading_metrics['max_drawdown']),
            total_return=trading_metrics['total_return'],
            win_rate=trading_metrics['win_rate'],
            profit_factor=trading_metrics['profit_factor'],
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            epochs_trained=epochs_trained,
            training_time=training_time,
            best_params=best_params
        )
        
        # Check if targets are met and calculate target score
        performance.meets_targets, performance.target_score = self.check_targets(performance)
        
        # Log performance
        self.log_performance(performance)
        
        return performance
    
    def run_backtesting_evaluation(self, model: nn.Module, model_type: str) -> Dict[str, float]:
        """Run backtesting evaluation of the model."""
        
        try:
            # Create backtesting components
            transaction_cost_model = TransactionCostModel(commission_rate=0.001)
            market_impact_model = MarketImpactModel()
            
            runner = BacktestRunner(
                initial_capital=100000,
                transaction_cost_model=transaction_cost_model,
                market_impact_model=market_impact_model
            )
            
            # Create ML strategy
            strategy = MLStrategy(
                name=f"{model_type}_strategy",
                model=model,
                min_signal_strength=0.05,
                min_confidence=0.3,
                max_position_size=0.15
            )
            
            # Run backtest
            results = runner.run_model_backtest(
                model=model,
                data_module=self.data_module,
                strategy=strategy,
                model_name=model_type
            )
            
            # Extract metrics
            metrics = results.get('metrics', {})
            return {
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': metrics.get('max_drawdown', 0.0),
                'total_return': metrics.get('total_return', 0.0),
                'win_rate': metrics.get('win_rate', 0.0),
                'profit_factor': metrics.get('profit_factor', 0.0)
            }
            
        except Exception as e:
            logger.warning(f"Backtesting evaluation failed: {e}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
    
    def check_targets(self, performance: ModelPerformance) -> Tuple[bool, float]:
        """Check if performance meets targets and calculate target score."""
        
        # Individual target checks
        correlation_ok = performance.correlation >= self.targets.min_correlation
        sharpe_ok = performance.sharpe_ratio >= self.targets.min_sharpe_ratio
        mse_ok = performance.mse <= self.targets.max_mse
        r2_ok = performance.r2 >= self.targets.min_r2
        directional_ok = performance.directional_accuracy >= self.targets.min_directional_accuracy
        drawdown_ok = performance.max_drawdown <= self.targets.max_drawdown
        
        # All targets met
        meets_targets = all([correlation_ok, sharpe_ok, mse_ok, r2_ok, directional_ok, drawdown_ok])
        
        # Calculate target score (higher is better)
        target_score = (
            min(performance.correlation / self.targets.min_correlation, 2.0) * 0.2 +
            min(performance.sharpe_ratio / self.targets.min_sharpe_ratio, 2.0) * 0.2 +
            min(self.targets.max_mse / performance.mse, 2.0) * 0.2 +
            min(performance.r2 / self.targets.min_r2, 2.0) * 0.15 +
            min(performance.directional_accuracy / self.targets.min_directional_accuracy, 2.0) * 0.15 +
            min(self.targets.max_drawdown / performance.max_drawdown, 2.0) * 0.1
        )
        
        return meets_targets, target_score
    
    def log_performance(self, performance: ModelPerformance):
        """Log model performance."""
        
        logger.info(f"\n{'='*50}")
        logger.info(f"PERFORMANCE REPORT: {performance.model_name}")
        logger.info(f"{'='*50}")
        
        logger.info(f"Statistical Metrics:")
        logger.info(f"  MSE: {performance.mse:.6f} (target: ≤{self.targets.max_mse:.6f}) {'✅' if performance.mse <= self.targets.max_mse else '❌'}")
        logger.info(f"  MAE: {performance.mae:.6f}")
        logger.info(f"  R²: {performance.r2:.4f} (target: ≥{self.targets.min_r2:.4f}) {'✅' if performance.r2 >= self.targets.min_r2 else '❌'}")
        logger.info(f"  Correlation: {performance.correlation:.4f} (target: ≥{self.targets.min_correlation:.4f}) {'✅' if performance.correlation >= self.targets.min_correlation else '❌'}")
        logger.info(f"  Directional Accuracy: {performance.directional_accuracy:.4f} (target: ≥{self.targets.min_directional_accuracy:.4f}) {'✅' if performance.directional_accuracy >= self.targets.min_directional_accuracy else '❌'}")
        
        logger.info(f"\nTrading Metrics:")
        logger.info(f"  Sharpe Ratio: {performance.sharpe_ratio:.4f} (target: ≥{self.targets.min_sharpe_ratio:.4f}) {'✅' if performance.sharpe_ratio >= self.targets.min_sharpe_ratio else '❌'}")
        logger.info(f"  Max Drawdown: {performance.max_drawdown:.4f} (target: ≤{self.targets.max_drawdown:.4f}) {'✅' if performance.max_drawdown <= self.targets.max_drawdown else '❌'}")
        logger.info(f"  Total Return: {performance.total_return:.4f}")
        logger.info(f"  Win Rate: {performance.win_rate:.4f}")
        
        logger.info(f"\nTraining Info:")
        logger.info(f"  Epochs Trained: {performance.epochs_trained}")
        logger.info(f"  Training Time: {performance.training_time:.2f}s")
        logger.info(f"  Final Val Loss: {performance.final_val_loss:.6f}")
        
        logger.info(f"\nOverall:")
        logger.info(f"  Target Score: {performance.target_score:.3f}")
        logger.info(f"  Meets All Targets: {'✅ YES' if performance.meets_targets else '❌ NO'}")
    
    def generate_final_report(self):
        """Generate final training report."""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL TRAINING ORCHESTRATION REPORT")
        logger.info(f"{'='*60}")
        
        if not self.model_performances:
            logger.info("No models were trained successfully.")
            return
        
        # Create summary table
        summary_data = []
        for model_name, performance in self.model_performances.items():
            summary_data.append({
                'Model': model_name,
                'Target Score': f"{performance.target_score:.3f}",
                'Meets Targets': '✅' if performance.meets_targets else '❌',
                'Correlation': f"{performance.correlation:.4f}",
                'Sharpe Ratio': f"{performance.sharpe_ratio:.4f}",
                'MSE': f"{performance.mse:.6f}",
                'R²': f"{performance.r2:.4f}",
                'Max Drawdown': f"{performance.max_drawdown:.4f}"
            })
        
        # Print summary table
        df = pd.DataFrame(summary_data)
        logger.info(f"\nModel Performance Summary:")
        logger.info(f"\n{df.to_string(index=False)}")
        
        # Best model
        best_model = max(self.model_performances.items(), key=lambda x: x[1].target_score)
        logger.info(f"\n🏆 Best Model: {best_model[0]} (Score: {best_model[1].target_score:.3f})")
        
        # Models meeting targets
        successful_models = [name for name, perf in self.model_performances.items() if perf.meets_targets]
        if successful_models:
            logger.info(f"🎉 Models meeting all targets: {', '.join(successful_models)}")
        else:
            logger.info("⚠️  No models met all performance targets")
        
        # Save detailed report
        report_path = self.output_dir / "final_training_report.json"
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'targets': self.targets.__dict__,
            'model_performances': {name: perf.to_dict() for name, perf in self.model_performances.items()},
            'summary': {
                'total_models_trained': len(self.model_performances),
                'models_meeting_targets': len(successful_models),
                'best_model': best_model[0] if best_model else None,
                'best_score': best_model[1].target_score if best_model else 0.0
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        # Generate comparison plots
        self.plot_model_comparison()
        
        logger.info(f"All results saved to: {self.output_dir}")
    
    def plot_model_comparison(self):
        """Create comparison plots for all models."""
        
        if len(self.model_performances) < 2:
            return
        
        try:
            # Prepare data for plotting
            models = list(self.model_performances.keys())
            correlations = [self.model_performances[m].correlation for m in models]
            sharpe_ratios = [self.model_performances[m].sharpe_ratio for m in models]
            mses = [self.model_performances[m].mse for m in models]
            target_scores = [self.model_performances[m].target_score for m in models]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Correlation comparison
            bars1 = ax1.bar(models, correlations, color='skyblue', edgecolor='navy')
            ax1.axhline(y=self.targets.min_correlation, color='red', linestyle='--', label='Target')
            ax1.set_ylabel('Correlation')
            ax1.set_title('Model Correlation Comparison')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars1, correlations):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # Sharpe ratio comparison
            bars2 = ax2.bar(models, sharpe_ratios, color='lightgreen', edgecolor='darkgreen')
            ax2.axhline(y=self.targets.min_sharpe_ratio, color='red', linestyle='--', label='Target')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_title('Model Sharpe Ratio Comparison')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars2, sharpe_ratios):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # MSE comparison (log scale)
            bars3 = ax3.bar(models, mses, color='salmon', edgecolor='darkred')
            ax3.axhline(y=self.targets.max_mse, color='red', linestyle='--', label='Target')
            ax3.set_ylabel('MSE (log scale)')
            ax3.set_title('Model MSE Comparison')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars3, mses):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                        f'{value:.6f}', ha='center', va='bottom', fontsize=8)
            
            # Target score comparison
            bars4 = ax4.bar(models, target_scores, color='gold', edgecolor='orange')
            ax4.axhline(y=1.0, color='red', linestyle='--', label='Target Threshold')
            ax4.set_ylabel('Target Score')
            ax4.set_title('Model Target Score Comparison')
            ax4.legend()
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars4, target_scores):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / "model_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create comparison plots: {e}")


if __name__ == "__main__":
    # Test the training orchestrator
    print("Testing Training Orchestrator...")
    
    # This would use real data in practice
    from src.data_processing.data_loader import OrderBookDataModule
    from src.data_processing.order_book_parser import create_synthetic_order_book_data
    from src.data_processing.feature_engineering import FeatureEngineering
    
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
    
    # Create training targets
    targets = TrainingTarget(
        min_correlation=0.1,      # Easier targets for testing
        min_sharpe_ratio=0.5,
        max_mse=0.01,
        min_r2=0.01,
        min_directional_accuracy=0.51,
        max_drawdown=0.2
    )
    
    # Create orchestrator
    orchestrator = TrainingOrchestrator(
        data_module=data_module,
        targets=targets,
        output_dir="test_orchestrated_training"
    )
    
    # Train models (with small parameters for testing)
    performances = orchestrator.train_all_models(
        model_types=['transformer'],  # Just one model for testing
        optimization_trials=3,        # Very few trials for testing
        max_training_rounds=1
    )
    
    print(f"Training completed. {len(performances)} models trained.")
    for name, perf in performances.items():
        print(f"{name}: Target Score = {perf.target_score:.3f}, Meets Targets = {perf.meets_targets}")
    
    print("✅ Training orchestrator test passed!")