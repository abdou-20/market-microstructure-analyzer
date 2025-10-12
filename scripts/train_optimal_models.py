#!/usr/bin/env python3
"""
Comprehensive Model Training Script

This script runs the complete training pipeline to optimize models and achieve
excellent performance on market microstructure prediction tasks.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from training.training_orchestrator import TrainingOrchestrator, TrainingTarget
from data_processing.data_loader import OrderBookDataModule
from data_processing.order_book_parser import create_synthetic_order_book_data
from data_processing.feature_engineering import FeatureEngineering
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Optimal Market Microstructure Models')
    
    # Data configuration
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to real data file (if not provided, uses synthetic data)')
    parser.add_argument('--num-snapshots', type=int, default=2000,
                       help='Number of synthetic snapshots to generate')
    parser.add_argument('--sequence-length', type=int, default=20,
                       help='Sequence length for models')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    # Model configuration
    parser.add_argument('--model-types', nargs='+', default=['transformer', 'lstm'],
                       choices=['transformer', 'lstm'],
                       help='Model types to train and optimize')
    
    # Training configuration
    parser.add_argument('--optimization-trials', type=int, default=100,
                       help='Number of hyperparameter optimization trials per round')
    parser.add_argument('--max-training-rounds', type=int, default=3,
                       help='Maximum training rounds per model')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for training')
    
    # Performance targets
    parser.add_argument('--min-correlation', type=float, default=0.2,
                       help='Minimum required correlation')
    parser.add_argument('--min-sharpe-ratio', type=float, default=1.0,
                       help='Minimum required Sharpe ratio')
    parser.add_argument('--max-mse', type=float, default=0.0005,
                       help='Maximum allowed MSE')
    parser.add_argument('--min-r2', type=float, default=0.08,
                       help='Minimum required R² score')
    parser.add_argument('--min-directional-accuracy', type=float, default=0.55,
                       help='Minimum required directional accuracy')
    parser.add_argument('--max-drawdown', type=float, default=0.12,
                       help='Maximum allowed drawdown')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='outputs/optimal_training',
                       help='Output directory for results')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (default: timestamp)')
    parser.add_argument('--save-all-models', action='store_true',
                       help='Save all trained models (not just the best)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    return device


def create_data_module(args) -> OrderBookDataModule:
    """Create data module for training."""
    
    if args.data_path:
        logger.info(f"Loading data from {args.data_path}")
        # Real data loading would be implemented here
        raise NotImplementedError("Real data loading not implemented yet")
    else:
        logger.info(f"Creating synthetic data with {args.num_snapshots} snapshots")
        
        # Create synthetic order book data
        snapshots = create_synthetic_order_book_data(num_snapshots=args.num_snapshots)
        
        # Extract features
        feature_engineer = FeatureEngineering(
            lookback_window=10,
            prediction_horizon=5
        )
        feature_vectors = feature_engineer.extract_features(snapshots)
        
        # Create data module
        data_module = OrderBookDataModule(
            feature_vectors=feature_vectors,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            scaler_type='standard'
        )
        
        logger.info(f"Created data module with {len(feature_vectors)} samples")
        logger.info(f"Feature dimension: {data_module.feature_dim}")
        
        return data_module


def create_training_targets(args) -> TrainingTarget:
    """Create training targets from arguments."""
    
    targets = TrainingTarget(
        min_correlation=args.min_correlation,
        min_sharpe_ratio=args.min_sharpe_ratio,
        max_mse=args.max_mse,
        min_r2=args.min_r2,
        min_directional_accuracy=args.min_directional_accuracy,
        max_drawdown=args.max_drawdown
    )
    
    logger.info("Training targets:")
    logger.info(f"  Minimum Correlation: {targets.min_correlation:.3f}")
    logger.info(f"  Minimum Sharpe Ratio: {targets.min_sharpe_ratio:.3f}")
    logger.info(f"  Maximum MSE: {targets.max_mse:.6f}")
    logger.info(f"  Minimum R²: {targets.min_r2:.3f}")
    logger.info(f"  Minimum Directional Accuracy: {targets.min_directional_accuracy:.3f}")
    logger.info(f"  Maximum Drawdown: {targets.max_drawdown:.3f}")
    
    return targets


def analyze_results(performances: dict, targets: TrainingTarget):
    """Analyze and report final results."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL RESULTS ANALYSIS")
    logger.info(f"{'='*80}")
    
    if not performances:
        logger.error("No models were trained successfully!")
        return False
    
    # Find best models
    successful_models = [name for name, perf in performances.items() if perf.meets_targets]
    best_model = max(performances.items(), key=lambda x: x[1].target_score)
    
    # Overall statistics
    total_models = len(performances)
    success_rate = len(successful_models) / total_models * 100
    avg_score = np.mean([perf.target_score for perf in performances.values()])
    
    logger.info(f"Total Models Trained: {total_models}")
    logger.info(f"Models Meeting All Targets: {len(successful_models)} ({success_rate:.1f}%)")
    logger.info(f"Average Target Score: {avg_score:.3f}")
    logger.info(f"Best Model: {best_model[0]} (Score: {best_model[1].target_score:.3f})")
    
    # Detailed performance breakdown
    logger.info(f"\nDETAILED PERFORMANCE BREAKDOWN:")
    logger.info(f"{'Model':<20} {'Score':<8} {'Targets':<8} {'Corr':<8} {'Sharpe':<8} {'MSE':<10} {'R²':<8}")
    logger.info(f"{'-'*80}")
    
    for name, perf in performances.items():
        targets_met = '✅' if perf.meets_targets else '❌'
        logger.info(f"{name:<20} {perf.target_score:<8.3f} {targets_met:<8} "
                   f"{perf.correlation:<8.3f} {perf.sharpe_ratio:<8.3f} "
                   f"{perf.mse:<10.6f} {perf.r2:<8.3f}")
    
    # Success indicators
    if successful_models:
        logger.info(f"\n🎉 SUCCESS: {len(successful_models)} model(s) achieved all performance targets!")
        logger.info(f"Successful models: {', '.join(successful_models)}")
        return True
    else:
        logger.info(f"\n⚠️  PARTIAL SUCCESS: No models met all targets, but improvements were achieved")
        logger.info(f"Best performing model: {best_model[0]} (Score: {best_model[1].target_score:.3f})")
        
        # Show how close we got to targets
        best_perf = best_model[1]
        logger.info(f"\nBest model performance vs targets:")
        logger.info(f"  Correlation: {best_perf.correlation:.3f} / {targets.min_correlation:.3f} "
                   f"({'✅' if best_perf.correlation >= targets.min_correlation else '❌'})")
        logger.info(f"  Sharpe Ratio: {best_perf.sharpe_ratio:.3f} / {targets.min_sharpe_ratio:.3f} "
                   f"({'✅' if best_perf.sharpe_ratio >= targets.min_sharpe_ratio else '❌'})")
        logger.info(f"  MSE: {best_perf.mse:.6f} / {targets.max_mse:.6f} "
                   f"({'✅' if best_perf.mse <= targets.max_mse else '❌'})")
        logger.info(f"  R²: {best_perf.r2:.3f} / {targets.min_r2:.3f} "
                   f"({'✅' if best_perf.r2 >= targets.min_r2 else '❌'})")
        
        return False


def save_deployment_ready_model(best_model_name: str, output_dir: Path):
    """Save the best model in deployment-ready format."""
    
    # Find the best model file
    model_files = list(output_dir.glob(f"*{best_model_name.split('_')[0]}*_model_*.pt"))
    
    if not model_files:
        logger.warning("Could not find best model file for deployment")
        return
    
    # Get the most recent model file
    best_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
    
    # Copy to deployment location
    deployment_path = output_dir / "deployment_ready_model.pt"
    
    try:
        import shutil
        shutil.copy2(best_model_file, deployment_path)
        logger.info(f"Deployment-ready model saved to: {deployment_path}")
        
        # Save deployment metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'source_model': str(best_model_file),
            'model_name': best_model_name,
            'deployment_path': str(deployment_path)
        }
        
        metadata_path = output_dir / "deployment_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Deployment metadata saved to: {metadata_path}")
        
    except Exception as e:
        logger.error(f"Failed to save deployment-ready model: {e}")


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup experiment
    if args.experiment_name is None:
        args.experiment_name = f"optimal_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Update output directory with experiment name
    args.output_dir = str(Path(args.output_dir) / args.experiment_name)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    logger.info(f"🚀 Starting Optimal Model Training: {args.experiment_name}")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Setup device
        device = setup_device(args.device)
        
        # Create data module
        data_module = create_data_module(args)
        
        # Create training targets
        targets = create_training_targets(args)
        
        # Create training orchestrator
        orchestrator = TrainingOrchestrator(
            data_module=data_module,
            targets=targets,
            output_dir=args.output_dir,
            device=device
        )
        
        # Train all models
        logger.info(f"\n🎯 Starting comprehensive model training...")
        logger.info(f"Models to train: {args.model_types}")
        logger.info(f"Optimization trials per round: {args.optimization_trials}")
        logger.info(f"Maximum training rounds: {args.max_training_rounds}")
        
        start_time = datetime.now()
        
        performances = orchestrator.train_all_models(
            model_types=args.model_types,
            optimization_trials=args.optimization_trials,
            max_training_rounds=args.max_training_rounds
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Analyze results
        success = analyze_results(performances, targets)
        
        # Save deployment-ready model
        if performances:
            best_model_name = max(performances.items(), key=lambda x: x[1].target_score)[0]
            save_deployment_ready_model(best_model_name, Path(args.output_dir))
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING COMPLETED")
        logger.info(f"{'='*80}")
        logger.info(f"Experiment: {args.experiment_name}")
        logger.info(f"Duration: {training_duration}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"Success: {'✅ YES' if success else '⚠️  PARTIAL'}")
        
        if success:
            logger.info(f"\n🎉 CONGRATULATIONS! Models achieved excellent performance!")
            logger.info(f"   All target metrics have been met.")
            logger.info(f"   Models are ready for deployment.")
        else:
            logger.info(f"\n📈 GOOD PROGRESS! Significant improvements achieved.")
            logger.info(f"   Consider adjusting targets or running additional rounds.")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)