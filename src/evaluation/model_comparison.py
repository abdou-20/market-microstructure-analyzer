"""
Fair Model Comparison Framework

This module provides a framework for fairly comparing traditional microstructure
models with deep learning models on identical data splits and evaluation metrics.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.traditional_models import TraditionalModelEnsemble, ModelPrediction
from src.training.validation import ValidationMetrics, PerformanceEvaluator
from src.data_processing.data_loader import OrderBookDataModule
from src.utils.logger import get_experiment_logger

logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonResult:
    """Results from comparing models."""
    model_name: str
    model_type: str  # 'traditional' or 'deep_learning'
    metrics: ValidationMetrics
    predictions: np.ndarray
    targets: np.ndarray
    execution_time: float
    additional_info: Optional[Dict[str, Any]] = None


class FairComparisonFramework:
    """
    Framework for fair comparison between traditional and deep learning models.
    
    Ensures:
    - Same data splits for all models
    - Same evaluation metrics
    - Consistent preprocessing
    - Fair timing comparisons
    """
    
    def __init__(self, 
                 output_dir: Union[str, Path] = "outputs/model_comparison",
                 random_seed: int = 42):
        """
        Initialize comparison framework.
        
        Args:
            output_dir: Directory to save comparison results
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.evaluator = PerformanceEvaluator()
        self.comparison_results = {}
        
        # Data storage for consistent evaluation
        self.test_features = None
        self.test_targets = None
        self.test_metadata = None
        
        logger.info(f"Initialized fair comparison framework with output_dir: {output_dir}")
    
    def prepare_data(self, data_module: OrderBookDataModule):
        """
        Prepare consistent test data for all models.
        
        Args:
            data_module: Data module with train/val/test splits
        """
        logger.info("Preparing consistent test data for all models...")
        
        # Extract test data
        test_loader = data_module.test_dataloader()
        
        all_features = []
        all_targets = []
        all_metadata = []
        
        for batch in test_loader:
            if len(batch) == 2:
                features, targets = batch
                metadata = None
            else:
                features, targets, metadata = batch
            
            all_features.append(features)
            all_targets.append(targets)
            if metadata is not None:
                all_metadata.extend(metadata)
        
        self.test_features = torch.cat(all_features, dim=0)
        self.test_targets = torch.cat(all_targets, dim=0)
        self.test_metadata = all_metadata if all_metadata else None
        
        logger.info(f"Prepared test data: {self.test_features.shape[0]} samples")
    
    def evaluate_traditional_models(self, 
                                  traditional_ensemble: TraditionalModelEnsemble,
                                  current_prices: Optional[np.ndarray] = None) -> Dict[str, ModelComparisonResult]:
        """
        Evaluate traditional models on test data.
        
        Args:
            traditional_ensemble: Ensemble of traditional models
            current_prices: Current prices for each sample (if available)
            
        Returns:
            Dictionary of model results
        """
        logger.info("Evaluating traditional models...")
        
        if self.test_features is None:
            raise ValueError("Must call prepare_data() first")
        
        results = {}
        
        # Convert features to numpy for traditional models
        features_np = self.test_features.numpy()
        targets_np = self.test_targets.numpy().flatten()
        
        # Extract relevant features for traditional models
        # Assuming features contain order flow and volume information
        # This needs to be adapted based on actual feature structure
        order_flows = self._extract_order_flows(features_np)
        volumes = self._extract_volumes(features_np)
        
        if current_prices is None:
            current_prices = self._extract_current_prices(features_np)
        
        # Evaluate ensemble
        start_time = datetime.now()
        ensemble_predictions = []
        individual_predictions = {}
        
        for i in range(len(order_flows)):
            # Get predictions from all models
            predictions = traditional_ensemble.predict(
                order_flow=order_flows[i],
                volume=volumes[i],
                current_price=current_prices[i]
            )
            
            # Store individual predictions
            for model_name, pred in predictions.items():
                if model_name not in individual_predictions:
                    individual_predictions[model_name] = []
                individual_predictions[model_name].append(pred.prediction)
            
            # Get ensemble prediction
            ensemble_pred = traditional_ensemble.get_ensemble_prediction(predictions)
            ensemble_predictions.append(ensemble_pred.prediction)
        
        ensemble_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate ensemble
        ensemble_predictions = np.array(ensemble_predictions)
        ensemble_metrics = self.evaluator.evaluate_predictions(ensemble_predictions, targets_np)
        
        results['Traditional_Ensemble'] = ModelComparisonResult(
            model_name='Traditional_Ensemble',
            model_type='traditional',
            metrics=ensemble_metrics,
            predictions=ensemble_predictions,
            targets=targets_np,
            execution_time=ensemble_time,
            additional_info={
                'num_models': len(traditional_ensemble.models),
                'ensemble_method': traditional_ensemble.ensemble_method
            }
        )
        
        # Evaluate individual traditional models
        for model_name, preds in individual_predictions.items():
            preds = np.array(preds)
            metrics = self.evaluator.evaluate_predictions(preds, targets_np)
            
            results[f'Traditional_{model_name}'] = ModelComparisonResult(
                model_name=f'Traditional_{model_name}',
                model_type='traditional',
                metrics=metrics,
                predictions=preds,
                targets=targets_np,
                execution_time=ensemble_time / len(individual_predictions),  # Approximate
                additional_info={'model_type': model_name}
            )
        
        logger.info(f"Evaluated {len(results)} traditional models")
        return results
    
    def evaluate_deep_learning_model(self, 
                                   model: torch.nn.Module,
                                   device: torch.device = None,
                                   model_name: str = "DeepLearning") -> ModelComparisonResult:
        """
        Evaluate deep learning model on test data.
        
        Args:
            model: Trained PyTorch model
            device: Device for inference
            model_name: Name for the model
            
        Returns:
            Model comparison result
        """
        logger.info(f"Evaluating deep learning model: {model_name}...")
        
        if self.test_features is None:
            raise ValueError("Must call prepare_data() first")
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.to(device)
        model.eval()
        
        # Inference
        start_time = datetime.now()
        predictions = []
        
        # Process in batches to avoid memory issues
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(self.test_features), batch_size):
                batch_features = self.test_features[i:i+batch_size].to(device)
                
                output = model(batch_features)
                batch_predictions = output['predictions'].cpu().numpy().flatten()
                predictions.extend(batch_predictions)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        predictions = np.array(predictions)
        targets = self.test_targets.numpy().flatten()
        
        # Evaluate
        metrics = self.evaluator.evaluate_predictions(predictions, targets)
        
        # Get model info if available
        additional_info = {}
        if hasattr(model, 'get_model_info'):
            additional_info.update(model.get_model_info())
        if hasattr(model, 'count_parameters'):
            additional_info['total_parameters'] = model.count_parameters()
        
        return ModelComparisonResult(
            model_name=model_name,
            model_type='deep_learning',
            metrics=metrics,
            predictions=predictions,
            targets=targets,
            execution_time=execution_time,
            additional_info=additional_info
        )
    
    def add_comparison_result(self, result: ModelComparisonResult):
        """Add a comparison result to the framework."""
        self.comparison_results[result.model_name] = result
        logger.info(f"Added comparison result for {result.model_name}")
    
    def compare_models(self, 
                      traditional_ensemble: TraditionalModelEnsemble,
                      deep_learning_models: List[Tuple[torch.nn.Module, str]],
                      data_module: OrderBookDataModule) -> Dict[str, ModelComparisonResult]:
        """
        Compare traditional and deep learning models.
        
        Args:
            traditional_ensemble: Traditional model ensemble
            deep_learning_models: List of (model, name) tuples
            data_module: Data module for consistent evaluation
            
        Returns:
            Complete comparison results
        """
        logger.info("Starting comprehensive model comparison...")
        
        # Prepare consistent data
        self.prepare_data(data_module)
        
        # Evaluate traditional models
        traditional_results = self.evaluate_traditional_models(traditional_ensemble)
        for name, result in traditional_results.items():
            self.add_comparison_result(result)
        
        # Evaluate deep learning models
        for model, name in deep_learning_models:
            dl_result = self.evaluate_deep_learning_model(model, model_name=name)
            self.add_comparison_result(dl_result)
        
        logger.info(f"Comparison completed for {len(self.comparison_results)} models")
        return self.comparison_results
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """
        Generate comprehensive comparison report.
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.comparison_results:
            raise ValueError("No comparison results available")
        
        # Create comparison DataFrame
        report_data = []
        
        for model_name, result in self.comparison_results.items():
            metrics_dict = result.metrics.to_dict()
            
            row = {
                'Model': model_name,
                'Type': result.model_type,
                'Execution_Time': result.execution_time,
                **metrics_dict
            }
            
            # Add additional info
            if result.additional_info:
                for key, value in result.additional_info.items():
                    if isinstance(value, (int, float, str)):
                        row[f'Info_{key}'] = value
            
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        # Sort by performance metric (e.g., Sharpe ratio)
        if 'sharpe_ratio' in df.columns:
            df = df.sort_values('sharpe_ratio', ascending=False)
        
        return df
    
    def plot_comparison_results(self, save_plots: bool = True) -> List[plt.Figure]:
        """
        Create visualization plots for model comparison.
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            List of matplotlib figures
        """
        if not self.comparison_results:
            raise ValueError("No comparison results available")
        
        figures = []
        
        # 1. Performance metrics comparison
        fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig1.suptitle('Model Performance Comparison', fontsize=16)
        
        # Prepare data
        models = list(self.comparison_results.keys())
        metrics = ['directional_accuracy', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        metric_titles = ['Directional Accuracy', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        colors = ['blue' if 'Traditional' in m else 'red' for m in models]
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[i//2, i%2]
            values = [self.comparison_results[m].metrics.to_dict()[metric] for m in models]
            
            bars = ax.bar(range(len(models)), values, color=colors, alpha=0.7)
            ax.set_title(title)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        figures.append(fig1)
        
        # 2. Execution time comparison
        fig2, ax = plt.subplots(figsize=(12, 6))
        execution_times = [self.comparison_results[m].execution_time for m in models]
        
        bars = ax.bar(range(len(models)), execution_times, color=colors, alpha=0.7)
        ax.set_title('Model Execution Time Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time in zip(bars, execution_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{time:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        figures.append(fig2)
        
        # 3. Prediction scatter plots
        fig3, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig3.suptitle('Predictions vs Targets', fontsize=16)
        
        model_names = list(self.comparison_results.keys())[:4]  # Show top 4 models
        for i, model_name in enumerate(model_names):
            ax = axes[i//2, i%2]
            result = self.comparison_results[model_name]
            
            ax.scatter(result.targets, result.predictions, alpha=0.5, s=1)
            ax.plot([result.targets.min(), result.targets.max()], 
                   [result.targets.min(), result.targets.max()], 'r--', alpha=0.8)
            ax.set_title(f'{model_name} (R² = {result.metrics.directional_accuracy:.3f})')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures.append(fig3)
        
        # Save plots if requested
        if save_plots:
            for i, fig in enumerate(figures):
                plot_path = self.output_dir / f'comparison_plot_{i+1}.png'
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot to {plot_path}")
        
        return figures
    
    def save_results(self, filename: str = "model_comparison_results.json"):
        """
        Save comparison results to file.
        
        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Prepare data for serialization
        results_data = {}
        for model_name, result in self.comparison_results.items():
            results_data[model_name] = {
                'model_name': result.model_name,
                'model_type': result.model_type,
                'metrics': result.metrics.to_dict(),
                'execution_time': result.execution_time,
                'predictions_stats': {
                    'mean': float(np.mean(result.predictions)),
                    'std': float(np.std(result.predictions)),
                    'min': float(np.min(result.predictions)),
                    'max': float(np.max(result.predictions))
                },
                'additional_info': result.additional_info
            }
        
        # Save metadata
        metadata = {
            'comparison_timestamp': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'num_test_samples': len(self.test_targets) if self.test_targets is not None else 0,
            'test_data_shape': list(self.test_features.shape) if self.test_features is not None else None
        }
        
        output_data = {
            'metadata': metadata,
            'results': results_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Saved comparison results to {output_path}")
        
        # Also save CSV report
        report_df = self.generate_comparison_report()
        csv_path = self.output_dir / "model_comparison_report.csv"
        report_df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison report to {csv_path}")
    
    def _extract_order_flows(self, features: np.ndarray) -> np.ndarray:
        """
        Extract order flow information from features.
        
        This is a placeholder - needs to be adapted based on actual feature structure.
        """
        # Assuming order flow is in the first column or can be computed
        # This needs to be customized based on the actual feature engineering
        if features.shape[-1] > 10:  # Assume OFI is available
            return features[:, -1, 10]  # Example: OFI at last timestamp
        else:
            return np.random.normal(0, 100, features.shape[0])  # Placeholder
    
    def _extract_volumes(self, features: np.ndarray) -> np.ndarray:
        """Extract volume information from features."""
        # Placeholder implementation
        if features.shape[-1] > 5:
            return np.abs(features[:, -1, 5])  # Example: volume at last timestamp
        else:
            return np.abs(np.random.normal(1000, 200, features.shape[0]))  # Placeholder
    
    def _extract_current_prices(self, features: np.ndarray) -> np.ndarray:
        """Extract current price information from features."""
        # Placeholder implementation
        if features.shape[-1] > 0:
            return features[:, -1, 0]  # Example: first feature at last timestamp
        else:
            return np.random.normal(50000, 100, features.shape[0])  # Placeholder


def run_comprehensive_comparison(traditional_ensemble: TraditionalModelEnsemble,
                               deep_learning_models: List[Tuple[torch.nn.Module, str]],
                               data_module: OrderBookDataModule,
                               output_dir: str = "outputs/model_comparison") -> FairComparisonFramework:
    """
    Run a comprehensive model comparison.
    
    Args:
        traditional_ensemble: Traditional model ensemble
        deep_learning_models: List of (model, name) tuples
        data_module: Data module for evaluation
        output_dir: Output directory for results
        
    Returns:
        Comparison framework with results
    """
    # Create comparison framework
    framework = FairComparisonFramework(output_dir=output_dir)
    
    # Run comparison
    framework.compare_models(traditional_ensemble, deep_learning_models, data_module)
    
    # Generate report and plots
    report_df = framework.generate_comparison_report()
    print("Model Comparison Report:")
    print(report_df.to_string(index=False))
    
    # Create visualizations
    framework.plot_comparison_results(save_plots=True)
    
    # Save results
    framework.save_results()
    
    return framework


if __name__ == "__main__":
    # Test the comparison framework
    from src.models.traditional_models import TraditionalModelEnsemble
    from src.models.transformer_model import create_transformer_model
    from src.data_processing.data_loader import OrderBookDataModule
    from src.data_processing.order_book_parser import create_synthetic_order_book_data
    from src.data_processing.feature_engineering import FeatureEngineering
    
    print("Testing model comparison framework...")
    
    # Create synthetic data
    snapshots = create_synthetic_order_book_data(num_snapshots=1000)
    feature_engineer = FeatureEngineering(lookback_window=10, prediction_horizon=5)
    feature_vectors = feature_engineer.extract_features(snapshots)
    
    # Create data module
    data_module = OrderBookDataModule(
        feature_vectors=feature_vectors,
        sequence_length=20,
        batch_size=32,
        scaler_type='standard'
    )
    
    # Create traditional ensemble
    traditional_ensemble = TraditionalModelEnsemble(
        use_glosten_milgrom=True,
        use_kyle_lambda=True,
        use_momentum=True,
        use_mean_reversion=True,
        use_vwap=True
    )
    
    # Create simple deep learning model for testing
    config = {
        'input_dim': 46,
        'model': {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'output_size': 1
        }
    }
    
    dl_model = create_transformer_model(config)
    
    # Run comparison
    framework = FairComparisonFramework(output_dir="test_comparison")
    framework.compare_models(
        traditional_ensemble=traditional_ensemble,
        deep_learning_models=[(dl_model, "Transformer_Test")],
        data_module=data_module
    )
    
    # Generate report
    report = framework.generate_comparison_report()
    print("\nComparison Report:")
    print(report)
    
    # Save results
    framework.save_results()
    
    print("✅ Model comparison framework test passed!")