"""
Hybrid Training Pipeline

This module extends the training pipeline to support both traditional microstructure
models and deep learning models, enabling fair comparison and ensemble approaches.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
import json

from src.training.trainer import ModelTrainer, TrainingMetrics
from src.training.validation import ValidationMetrics, PerformanceEvaluator
from src.models.traditional_models import TraditionalModelEnsemble, ModelPrediction
from src.evaluation.model_comparison import FairComparisonFramework, ModelComparisonResult
from src.evaluation.traditional_evaluation import TraditionalModelEvaluator
from src.data_processing.data_loader import OrderBookDataModule
from src.utils.logger import get_experiment_logger

logger = logging.getLogger(__name__)


class HybridTrainingPipeline:
    """
    Hybrid training pipeline that trains and evaluates both traditional
    and deep learning models on the same data.
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 output_dir: Union[str, Path] = "outputs/hybrid_training",
                 experiment_name: str = "hybrid_microstructure_training"):
        """
        Initialize hybrid training pipeline.
        
        Args:
            config: Training configuration
            output_dir: Output directory for results
            experiment_name: Name for experiment tracking
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.traditional_ensemble = None
        self.deep_learning_models = []
        self.comparison_framework = None
        self.traditional_evaluator = None
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        self.comparison_results = {}
        
        # Setup experiment tracking
        self.experiment_logger = get_experiment_logger(experiment_name)
        
        logger.info(f"Initialized hybrid training pipeline with output_dir: {output_dir}")
    
    def setup_traditional_models(self, 
                                traditional_config: Optional[Dict[str, Any]] = None) -> TraditionalModelEnsemble:
        """
        Setup traditional microstructure models.
        
        Args:
            traditional_config: Configuration for traditional models
            
        Returns:
            Traditional model ensemble
        """
        logger.info("Setting up traditional models...")
        
        if traditional_config is None:
            traditional_config = self.config.get('traditional_models', {})
        
        # Create ensemble with specified models
        self.traditional_ensemble = TraditionalModelEnsemble(
            use_glosten_milgrom=traditional_config.get('use_glosten_milgrom', True),
            use_kyle_lambda=traditional_config.get('use_kyle_lambda', True),
            use_momentum=traditional_config.get('use_momentum', True),
            use_mean_reversion=traditional_config.get('use_mean_reversion', True),
            use_vwap=traditional_config.get('use_vwap', True),
            ensemble_method=traditional_config.get('ensemble_method', 'weighted_average')
        )
        
        logger.info(f"Created traditional ensemble with {len(self.traditional_ensemble.models)} models")
        return self.traditional_ensemble
    
    def train_deep_learning_model(self,
                                model: nn.Module,
                                data_module: OrderBookDataModule,
                                model_name: str = "DeepLearning") -> Dict[str, Any]:
        """
        Train a deep learning model.
        
        Args:
            model: PyTorch model to train
            data_module: Data module for training
            model_name: Name for the model
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training deep learning model: {model_name}")
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            config=self.config,
            device=self.device,
            experiment_name=f"{self.experiment_name}_{model_name}"
        )
        
        # Get data loaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        # Train model
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config.get('training', {}).get('num_epochs', 100)
        )
        
        # Save training results
        results = {
            'model_name': model_name,
            'model_type': 'deep_learning',
            'training_history': [metrics.to_dict() for metrics in training_history],
            'best_val_loss': trainer.best_val_loss,
            'final_model_path': trainer.best_model_path
        }
        
        # Store trained model
        self.deep_learning_models.append((model, model_name))
        self.training_results[model_name] = results
        
        # Cleanup trainer
        trainer.cleanup()
        
        logger.info(f"Completed training for {model_name}")
        return results
    
    def evaluate_traditional_models(self,
                                  data_module: OrderBookDataModule) -> Dict[str, Any]:
        """
        Evaluate traditional models on test data.
        
        Args:
            data_module: Data module for evaluation
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating traditional models...")
        
        if self.traditional_ensemble is None:
            raise ValueError("Traditional models not setup. Call setup_traditional_models() first.")
        
        # Setup evaluator
        self.traditional_evaluator = TraditionalModelEvaluator(
            output_dir=self.output_dir / "traditional_evaluation"
        )
        
        # Extract test data
        test_loader = data_module.test_dataloader()
        
        all_features = []
        all_targets = []
        
        for batch in test_loader:
            if len(batch) == 2:
                features, targets = batch
            else:
                features, targets, _ = batch
            
            all_features.append(features)
            all_targets.append(targets)
        
        test_features = torch.cat(all_features, dim=0).numpy()
        test_targets = torch.cat(all_targets, dim=0).numpy().flatten()
        
        # Extract market microstructure features
        order_flows = self._extract_order_flows(test_features)
        volumes = self._extract_volumes(test_features)
        prices = self._extract_current_prices(test_features)
        
        # Evaluate individual models
        evaluation_results = {}
        
        # Evaluate Glosten-Milgrom if present
        if 'glosten_milgrom' in self.traditional_ensemble.models:
            gm_model = self.traditional_ensemble.models['glosten_milgrom']
            gm_diagnostics = self.traditional_evaluator.evaluate_glosten_milgrom_model(
                gm_model, order_flows, volumes, prices, test_targets
            )
            evaluation_results['glosten_milgrom'] = gm_diagnostics
        
        # Evaluate Kyle's Lambda if present
        if 'kyle_lambda' in self.traditional_ensemble.models:
            kyle_model = self.traditional_ensemble.models['kyle_lambda']
            kyle_diagnostics = self.traditional_evaluator.evaluate_kyle_lambda_model(
                kyle_model, order_flows, volumes, test_targets
            )
            evaluation_results['kyle_lambda'] = kyle_diagnostics
        
        # Evaluate baseline models if present
        baseline_models = getattr(self.traditional_ensemble, 'baseline_models', None)
        if baseline_models is not None:
            baseline_diagnostics = self.traditional_evaluator.evaluate_baseline_models(
                baseline_models, prices, volumes, test_targets
            )
            evaluation_results.update(baseline_diagnostics)
        
        # Evaluate ensemble
        ensemble_diagnostics = self.traditional_evaluator.evaluate_ensemble(
            self.traditional_ensemble, order_flows, volumes, prices, test_targets
        )
        evaluation_results['ensemble'] = ensemble_diagnostics
        
        self.evaluation_results['traditional'] = evaluation_results
        
        logger.info(f"Completed evaluation of {len(evaluation_results)} traditional models")
        return evaluation_results
    
    def run_comprehensive_comparison(self,
                                   data_module: OrderBookDataModule) -> Dict[str, ModelComparisonResult]:
        """
        Run comprehensive comparison between all models.
        
        Args:
            data_module: Data module for comparison
            
        Returns:
            Comparison results
        """
        logger.info("Running comprehensive model comparison...")
        
        # Setup comparison framework
        self.comparison_framework = FairComparisonFramework(
            output_dir=self.output_dir / "model_comparison"
        )
        
        # Run comparison
        comparison_results = self.comparison_framework.compare_models(
            traditional_ensemble=self.traditional_ensemble,
            deep_learning_models=self.deep_learning_models,
            data_module=data_module
        )
        
        self.comparison_results = comparison_results
        
        # Generate and save comparison report
        report_df = self.comparison_framework.generate_comparison_report()
        self._log_comparison_summary(report_df)
        
        # Create visualizations
        self.comparison_framework.plot_comparison_results(save_plots=True)
        
        # Save results
        self.comparison_framework.save_results()
        
        logger.info("Completed comprehensive model comparison")
        return comparison_results
    
    def run_full_pipeline(self,
                         models_to_train: List[Tuple[nn.Module, str]],
                         data_module: OrderBookDataModule,
                         traditional_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete hybrid training and evaluation pipeline.
        
        Args:
            models_to_train: List of (model, name) tuples for deep learning models
            data_module: Data module for training and evaluation
            traditional_config: Configuration for traditional models
            
        Returns:
            Complete pipeline results
        """
        logger.info("Starting full hybrid training pipeline...")
        
        pipeline_results = {
            'pipeline_start_time': datetime.now().isoformat(),
            'config': self.config
        }
        
        # Step 1: Setup traditional models
        self.setup_traditional_models(traditional_config)
        pipeline_results['traditional_models_setup'] = True
        
        # Step 2: Train deep learning models
        for model, model_name in models_to_train:
            training_results = self.train_deep_learning_model(model, data_module, model_name)
            pipeline_results[f'training_{model_name}'] = training_results
        
        # Step 3: Evaluate traditional models
        traditional_evaluation = self.evaluate_traditional_models(data_module)
        pipeline_results['traditional_evaluation'] = traditional_evaluation
        
        # Step 4: Run comprehensive comparison
        comparison_results = self.run_comprehensive_comparison(data_module)
        pipeline_results['comparison_results'] = comparison_results
        
        # Step 5: Generate final summary
        final_summary = self._generate_final_summary()
        pipeline_results['final_summary'] = final_summary
        
        # Save complete results
        self._save_pipeline_results(pipeline_results)
        
        pipeline_results['pipeline_end_time'] = datetime.now().isoformat()
        
        logger.info("Completed full hybrid training pipeline")
        return pipeline_results
    
    def _extract_order_flows(self, features: np.ndarray) -> np.ndarray:
        """Extract order flow information from features."""
        # This is a placeholder implementation
        # Should be customized based on actual feature engineering
        if features.shape[-1] > 10:
            return features[:, -1, 10]  # Assume OFI is at index 10
        else:
            return np.random.normal(0, 100, features.shape[0])
    
    def _extract_volumes(self, features: np.ndarray) -> np.ndarray:
        """Extract volume information from features."""
        if features.shape[-1] > 5:
            return np.abs(features[:, -1, 5])  # Assume volume is at index 5
        else:
            return np.abs(np.random.normal(1000, 200, features.shape[0]))
    
    def _extract_current_prices(self, features: np.ndarray) -> np.ndarray:
        """Extract current price information from features."""
        if features.shape[-1] > 0:
            return features[:, -1, 0]  # Assume price is at index 0
        else:
            return np.random.normal(50000, 100, features.shape[0])
    
    def _log_comparison_summary(self, report_df: pd.DataFrame):
        """Log comparison summary to experiment logger."""
        logger.info("Model Comparison Summary:")
        logger.info(f"\n{report_df.to_string(index=False)}")
        
        # Log key metrics to experiment logger
        for _, row in report_df.iterrows():
            model_name = row['Model']
            self.experiment_logger.log_metric(f"{model_name}_directional_accuracy", row['directional_accuracy'])
            self.experiment_logger.log_metric(f"{model_name}_sharpe_ratio", row['sharpe_ratio'])
            self.experiment_logger.log_metric(f"{model_name}_execution_time", row['Execution_Time'])
    
    def _generate_final_summary(self) -> Dict[str, Any]:
        """Generate final summary of pipeline results."""
        summary = {
            'total_models_evaluated': len(self.comparison_results),
            'traditional_models_count': len([r for r in self.comparison_results.values() if r.model_type == 'traditional']),
            'deep_learning_models_count': len([r for r in self.comparison_results.values() if r.model_type == 'deep_learning']),
        }
        
        if self.comparison_results:
            # Find best model by Sharpe ratio
            best_model = max(self.comparison_results.items(), 
                           key=lambda x: x[1].metrics.sharpe_ratio)
            
            summary['best_model'] = {
                'name': best_model[0],
                'type': best_model[1].model_type,
                'sharpe_ratio': best_model[1].metrics.sharpe_ratio,
                'directional_accuracy': best_model[1].metrics.directional_accuracy,
                'execution_time': best_model[1].execution_time
            }
            
            # Calculate average performance by type
            traditional_models = [r for r in self.comparison_results.values() if r.model_type == 'traditional']
            dl_models = [r for r in self.comparison_results.values() if r.model_type == 'deep_learning']
            
            if traditional_models:
                summary['traditional_avg_sharpe'] = np.mean([m.metrics.sharpe_ratio for m in traditional_models])
                summary['traditional_avg_accuracy'] = np.mean([m.metrics.directional_accuracy for m in traditional_models])
            
            if dl_models:
                summary['deep_learning_avg_sharpe'] = np.mean([m.metrics.sharpe_ratio for m in dl_models])
                summary['deep_learning_avg_accuracy'] = np.mean([m.metrics.directional_accuracy for m in dl_models])
        
        return summary
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save complete pipeline results."""
        # Convert non-serializable objects
        serializable_results = self._make_serializable(results)
        
        # Save to JSON
        results_path = self.output_dir / "hybrid_pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Saved pipeline results to {results_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)


def create_hybrid_pipeline(config: Dict[str, Any], 
                         output_dir: str = "outputs/hybrid_training",
                         experiment_name: str = "hybrid_experiment") -> HybridTrainingPipeline:
    """
    Create a hybrid training pipeline with the given configuration.
    
    Args:
        config: Training configuration
        output_dir: Output directory
        experiment_name: Experiment name
        
    Returns:
        Configured hybrid training pipeline
    """
    return HybridTrainingPipeline(
        config=config,
        output_dir=output_dir,
        experiment_name=experiment_name
    )


if __name__ == "__main__":
    # Test the hybrid training pipeline
    from src.models.transformer_model import create_transformer_model
    from src.models.lstm_model import create_lstm_model
    from src.data_processing.data_loader import OrderBookDataModule
    from src.data_processing.order_book_parser import create_synthetic_order_book_data
    from src.data_processing.feature_engineering import FeatureEngineering
    
    print("Testing hybrid training pipeline...")
    
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
    
    # Configuration
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
            'batch_size': 32,
            'num_epochs': 5,  # Reduced for testing
            'loss_function': 'mse'
        },
        'traditional_models': {
            'use_glosten_milgrom': True,
            'use_kyle_lambda': True,
            'use_momentum': True,
            'use_mean_reversion': True,
            'use_vwap': True
        }
    }
    
    # Create models to train
    transformer_model = create_transformer_model(config)
    lstm_model = create_lstm_model(config)
    
    models_to_train = [
        (transformer_model, "Transformer"),
        (lstm_model, "LSTM")
    ]
    
    # Create and run pipeline
    pipeline = create_hybrid_pipeline(
        config=config,
        output_dir="test_hybrid_output",
        experiment_name="test_hybrid"
    )
    
    # Run subset of pipeline for testing
    pipeline.setup_traditional_models()
    print("✅ Traditional models setup completed")
    
    # Test single deep learning model training
    training_results = pipeline.train_deep_learning_model(
        transformer_model, data_module, "Test_Transformer"
    )
    print("✅ Deep learning model training completed")
    
    # Test traditional model evaluation
    traditional_eval = pipeline.evaluate_traditional_models(data_module)
    print("✅ Traditional model evaluation completed")
    
    print("✅ Hybrid training pipeline test passed!")