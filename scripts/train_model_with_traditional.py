"""
Enhanced Training Script with Traditional Model Support

This script extends the original training script to support traditional model
comparison and hybrid training approaches.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.transformer_model import create_transformer_model
from models.lstm_model import create_lstm_model, create_hybrid_model
from models.traditional_models import TraditionalModelEnsemble
from data_processing.data_loader import OrderBookDataModule
from data_processing.order_book_parser import create_synthetic_order_book_data
from data_processing.feature_engineering import FeatureEngineering
from training.trainer import ModelTrainer
from training.hybrid_trainer import HybridTrainingPipeline
from evaluation.model_comparison import FairComparisonFramework
from utils.config import ConfigManager
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Market Microstructure Models')
    
    # Model configuration
    parser.add_argument('--model-type', type=str, default='transformer',
                       choices=['transformer', 'lstm', 'hybrid'],
                       help='Type of model to train')
    
    # Data configuration
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to order book data file')
    parser.add_argument('--synthetic-data', action='store_true',
                       help='Use synthetic data for training')
    parser.add_argument('--num-snapshots', type=int, default=5000,
                       help='Number of synthetic snapshots to generate')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for training')
    
    # Validation configuration
    parser.add_argument('--validation-type', type=str, default='standard',
                       choices=['standard', 'walk-forward', 'cross-validation'],
                       help='Type of validation to use')
    
    # Traditional model options
    parser.add_argument('--compare-traditional', action='store_true',
                       help='Include traditional model comparison')
    parser.add_argument('--hybrid-training', action='store_true',
                       help='Use hybrid training pipeline')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for the experiment')
    
    # Other options
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (reduced data and epochs)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    return device


def load_or_create_data(args) -> OrderBookDataModule:
    """Load or create training data."""
    if args.data_path:
        logger.info(f"Loading data from {args.data_path}")
        # Implementation for loading real data would go here
        raise NotImplementedError("Real data loading not implemented yet")
    else:
        logger.info(f"Creating synthetic data with {args.num_snapshots} snapshots")
        
        # Reduce data size in test mode
        num_snapshots = args.num_snapshots
        if args.test_mode:
            num_snapshots = min(500, num_snapshots)
        
        # Create synthetic order book data
        snapshots = create_synthetic_order_book_data(num_snapshots=num_snapshots)
        
        # Extract features
        feature_engineer = FeatureEngineering(
            lookback_window=10,
            prediction_horizon=5
        )
        feature_vectors = feature_engineer.extract_features(snapshots)
        
        # Create data module
        data_module = OrderBookDataModule(
            feature_vectors=feature_vectors,
            sequence_length=20,
            batch_size=args.batch_size,
            scaler_type='standard'
        )
        
        logger.info(f"Created data module with {len(feature_vectors)} samples")
        return data_module


def create_model(args, config: dict):
    """Create model based on configuration."""
    if args.model_type == 'transformer':
        return create_transformer_model(config)
    elif args.model_type == 'lstm':
        return create_lstm_model(config)
    elif args.model_type == 'hybrid':
        return create_hybrid_model(config)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def train_standard_model(args, config: dict, data_module: OrderBookDataModule, device: torch.device):
    """Train a standard deep learning model."""
    logger.info(f"Training standard {args.model_type} model...")
    
    # Create model
    model = create_model(args, config)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        config=config,
        device=device,
        experiment_name=args.experiment_name or f"{args.model_type}_training"
    )
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Train model
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs
    )
    
    # Cleanup
    trainer.cleanup()
    
    return model, training_history


def run_traditional_comparison(args, config: dict, data_module: OrderBookDataModule, 
                             trained_models: list):
    """Run comparison with traditional models."""
    logger.info("Running traditional model comparison...")
    
    # Create traditional ensemble
    traditional_ensemble = TraditionalModelEnsemble(
        use_glosten_milgrom=True,
        use_kyle_lambda=True,
        use_momentum=True,
        use_mean_reversion=True,
        use_vwap=True
    )
    
    # Create comparison framework
    comparison_framework = FairComparisonFramework(
        output_dir=Path(args.output_dir) / "model_comparison"
    )
    
    # Run comparison
    comparison_results = comparison_framework.compare_models(
        traditional_ensemble=traditional_ensemble,
        deep_learning_models=trained_models,
        data_module=data_module
    )
    
    # Generate report
    report_df = comparison_framework.generate_comparison_report()
    logger.info("Model Comparison Results:")
    logger.info(f"\n{report_df.to_string(index=False)}")
    
    # Create visualizations
    comparison_framework.plot_comparison_results(save_plots=True)
    
    # Save results
    comparison_framework.save_results()
    
    return comparison_results


def run_hybrid_training(args, config: dict, data_module: OrderBookDataModule):
    """Run hybrid training pipeline."""
    logger.info("Running hybrid training pipeline...")
    
    # Create hybrid pipeline
    pipeline = HybridTrainingPipeline(
        config=config,
        output_dir=Path(args.output_dir) / "hybrid_training",
        experiment_name=args.experiment_name or "hybrid_training"
    )
    
    # Create models to train
    models_to_train = []
    
    # Always include the primary model type
    primary_model = create_model(args, config)
    models_to_train.append((primary_model, f"Primary_{args.model_type}"))
    
    # Add additional models for comparison
    if args.model_type != 'transformer':
        transformer_model = create_transformer_model(config)
        models_to_train.append((transformer_model, "Transformer"))
    
    if args.model_type != 'lstm':
        lstm_model = create_lstm_model(config)
        models_to_train.append((lstm_model, "LSTM"))
    
    # Run full pipeline
    pipeline_results = pipeline.run_full_pipeline(
        models_to_train=models_to_train,
        data_module=data_module,
        traditional_config=config.get('traditional_models', {})
    )
    
    return pipeline_results


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=log_level)
    
    logger.info("Starting Market Microstructure Model Training")
    
    # Setup device
    device = setup_device(args.device)
    
    # Load configuration
    config_manager = ConfigManager()
    if args.config:
        config = config_manager.load_config(args.config)
    else:
        config = config_manager.get_default_config()
    
    # Override config with command line arguments
    config['training']['num_epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.learning_rate
    config['experiment']['output_dir'] = args.output_dir
    
    # Adjust for test mode
    if args.test_mode:
        logger.info("Running in test mode - reducing epochs and data size")
        config['training']['num_epochs'] = min(5, args.epochs)
        config['training']['early_stopping'] = True
        config['training']['patience'] = 3
    
    # Load or create data
    data_module = load_or_create_data(args)
    
    # Choose training approach
    if args.hybrid_training:
        # Run hybrid training pipeline
        results = run_hybrid_training(args, config, data_module)
        logger.info("Hybrid training pipeline completed successfully")
        
    else:
        # Train standard model
        model, training_history = train_standard_model(args, config, data_module, device)
        trained_models = [(model, args.model_type)]
        
        # Optional traditional model comparison
        if args.compare_traditional:
            comparison_results = run_traditional_comparison(
                args, config, data_module, trained_models
            )
            logger.info("Traditional model comparison completed")
        
        logger.info("Standard training completed successfully")
    
    # Final summary
    output_dir = Path(args.output_dir)
    logger.info(f"Training completed. Results saved to: {output_dir.absolute()}")
    
    # List key output files
    key_files = [
        output_dir / "final_model.pt",
        output_dir / "training_history.json",
        output_dir / "model_comparison_report.csv"
    ]
    
    existing_files = [f for f in key_files if f.exists()]
    if existing_files:
        logger.info("Key output files:")
        for file_path in existing_files:
            logger.info(f"  - {file_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if not parse_arguments().quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)