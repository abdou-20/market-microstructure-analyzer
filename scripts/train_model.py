#!/usr/bin/env python3
"""
Training Script for Market Microstructure Models

This script provides a command-line interface for training transformer and LSTM models
on order book data with various configuration options.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import logging
from typing import Dict, Any, Optional

# Add src to path
src_path = str(Path(__file__).parent.parent / 'src')
sys.path.insert(0, src_path)

from utils.logger import setup_logging
from utils.config import ConfigManager
from data_processing.order_book_parser import create_synthetic_order_book_data, OrderBookParser
from data_processing.feature_engineering import FeatureEngineering
from data_processing.data_loader import OrderBookDataModule, WalkForwardDataModule
from models.transformer_model import create_transformer_model
from models.lstm_model import create_lstm_model, create_hybrid_model
from training.trainer import ModelTrainer
from training.validation import WalkForwardValidator, CrossValidator


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Market Microstructure Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model-type", 
        choices=["transformer", "lstm", "hybrid"],
        default="transformer",
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    
    # Data configuration
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to order book data file"
    )
    
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use synthetic data for training"
    )
    
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=5000,
        help="Number of synthetic snapshots to generate"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for training"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for models and logs"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for this experiment"
    )
    
    # Validation configuration
    parser.add_argument(
        "--validation-type",
        choices=["standard", "walk-forward", "cross-validation"],
        default="standard",
        help="Type of validation to perform"
    )
    
    parser.add_argument(
        "--walk-forward-splits",
        type=int,
        default=5,
        help="Number of walk-forward splits"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output"
    )
    
    # Model-specific arguments
    parser.add_argument(
        "--d-model",
        type=int,
        help="Transformer model dimension"
    )
    
    parser.add_argument(
        "--num-heads",
        type=int,
        help="Number of attention heads"
    )
    
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of transformer/LSTM layers"
    )
    
    parser.add_argument(
        "--hidden-size",
        type=int,
        help="LSTM hidden size"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate"
    )
    
    # Feature engineering
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="Input sequence length"
    )
    
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        help="Prediction horizon"
    )
    
    parser.add_argument(
        "--lookback-window",
        type=int,
        help="Lookback window for feature engineering"
    )
    
    # Advanced options
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save attention weights for visualization"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with minimal data"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup training device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
    
    return device


def load_or_generate_data(args: argparse.Namespace) -> tuple:
    """Load data from file or generate synthetic data."""
    if args.synthetic_data or not args.data_path:
        print(f"Generating {args.num_snapshots} synthetic order book snapshots...")
        
        num_snapshots = 100 if args.test_mode else args.num_snapshots
        snapshots = create_synthetic_order_book_data(
            symbol="BTC-USD",
            num_snapshots=num_snapshots,
            start_price=50000.0,
            volatility=0.001,
            num_levels=10
        )
        
        print(f"Generated {len(snapshots)} snapshots")
        return snapshots, "synthetic"
    
    else:
        print(f"Loading order book data from {args.data_path}")
        
        parser = OrderBookParser(max_levels=10, validate_data=True)
        
        if args.data_path.endswith('.csv'):
            snapshots = list(parser.parse_csv(args.data_path))
        elif args.data_path.endswith('.json'):
            snapshots = list(parser.parse_json(args.data_path))
        else:
            raise ValueError(f"Unsupported file format: {args.data_path}")
        
        print(f"Loaded {len(snapshots)} snapshots")
        return snapshots, "real"


def create_feature_vectors(snapshots: list, args: argparse.Namespace) -> list:
    """Extract features from order book snapshots."""
    print("Extracting features from order book data...")
    
    lookback_window = args.lookback_window or 10
    prediction_horizon = args.prediction_horizon or 5
    
    if args.test_mode:
        lookback_window = min(lookback_window, 5)
        prediction_horizon = min(prediction_horizon, 3)
    
    feature_engineer = FeatureEngineering(
        lookback_window=lookback_window,
        prediction_horizon=prediction_horizon,
        max_levels=10
    )
    
    feature_vectors = feature_engineer.extract_features(snapshots)
    print(f"Extracted {len(feature_vectors)} feature vectors")
    
    if len(feature_vectors) == 0:
        raise ValueError("No feature vectors extracted. Check data quality and parameters.")
    
    return feature_vectors


def create_data_module(feature_vectors: list, args: argparse.Namespace, config: Dict[str, Any]) -> OrderBookDataModule:
    """Create data module for training."""
    print("Creating data module...")
    
    sequence_length = args.sequence_length or config.get('data', {}).get('sequence_length', 50)
    batch_size = args.batch_size or config.get('training', {}).get('batch_size', 32)
    
    if args.test_mode:
        sequence_length = min(sequence_length, 20)
        batch_size = min(batch_size, 16)
    
    data_module = OrderBookDataModule(
        feature_vectors=feature_vectors,
        sequence_length=sequence_length,
        prediction_horizon=args.prediction_horizon or 5,
        batch_size=batch_size,
        scaler_type='standard',
        num_workers=0  # Avoid multiprocessing issues
    )
    
    print(f"Data module created with sequence length: {sequence_length}, batch size: {batch_size}")
    return data_module


def create_model(args: argparse.Namespace, config: Dict[str, Any], input_dim: int) -> torch.nn.Module:
    """Create model based on arguments and configuration."""
    print(f"Creating {args.model_type} model...")
    
    # Update config with command line arguments
    model_config = config.get('model', {}).copy()
    
    if args.d_model:
        model_config['d_model'] = args.d_model
    if args.num_heads:
        model_config['num_heads'] = args.num_heads
    if args.num_layers:
        model_config['num_layers'] = args.num_layers
    if args.hidden_size:
        model_config['hidden_size'] = args.hidden_size
    if args.dropout is not None:
        model_config['dropout'] = args.dropout
    
    # Reduce model size for test mode
    if args.test_mode:
        if args.model_type == "transformer":
            model_config['d_model'] = min(model_config.get('d_model', 256), 64)
            model_config['num_heads'] = min(model_config.get('num_heads', 8), 4)
            model_config['num_layers'] = min(model_config.get('num_layers', 6), 2)
        elif args.model_type == "lstm":
            model_config['hidden_size'] = min(model_config.get('hidden_size', 256), 64)
            model_config['num_lstm_layers'] = min(model_config.get('num_lstm_layers', 2), 2)
    
    # Update config with input dimension
    config['input_dim'] = input_dim
    config['model'] = model_config
    
    # Create model
    if args.model_type == "transformer":
        model = create_transformer_model(config)
    elif args.model_type == "lstm":
        model = create_lstm_model(config)
    elif args.model_type == "hybrid":
        model = create_hybrid_model(config)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"Model created with {model.count_parameters()} parameters")
    return model


def train_model(model: torch.nn.Module, 
                data_module: OrderBookDataModule,
                args: argparse.Namespace,
                config: Dict[str, Any],
                device: torch.device) -> ModelTrainer:
    """Train the model."""
    print("Starting model training...")
    
    # Update training config with command line arguments
    training_config = config.get('training', {}).copy()
    
    if args.epochs:
        training_config['num_epochs'] = args.epochs
    if args.batch_size:
        training_config['batch_size'] = args.batch_size
    if args.learning_rate:
        training_config['learning_rate'] = args.learning_rate
    
    # Reduce epochs for test mode
    if args.test_mode:
        training_config['num_epochs'] = min(training_config.get('num_epochs', 100), 5)
        training_config['early_stopping'] = True
        training_config['patience'] = 3
    
    config['training'] = training_config
    
    # Update experiment config
    experiment_config = config.get('experiment', {}).copy()
    experiment_config['output_dir'] = args.output_dir
    if args.experiment_name:
        experiment_config['experiment_name'] = args.experiment_name
    config['experiment'] = experiment_config
    
    # Create trainer
    experiment_name = args.experiment_name or f"{args.model_type}_training"
    trainer = ModelTrainer(model, config, device=device, experiment_name=experiment_name)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader() if len(data_module.val_dataset) > 0 else None
    
    print(f"Training with {len(train_loader)} batches per epoch")
    if val_loader:
        print(f"Validation with {len(val_loader)} batches")
    
    # Train model
    training_history = trainer.train(train_loader, val_loader)
    
    print(f"Training completed after {len(training_history)} epochs")
    if training_history:
        final_metrics = training_history[-1]
        print(f"Final train loss: {final_metrics.train_loss:.6f}")
        if final_metrics.val_loss:
            print(f"Final val loss: {final_metrics.val_loss:.6f}")
    
    return trainer


def run_validation(model: torch.nn.Module,
                  feature_vectors: list,
                  args: argparse.Namespace,
                  config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run validation if requested."""
    if args.validation_type == "standard":
        return None
    
    print(f"Running {args.validation_type} validation...")
    
    if args.validation_type == "walk-forward":
        # Create walk-forward data module
        wf_data_module = WalkForwardDataModule(
            feature_vectors=feature_vectors,
            train_window_days=30,
            test_window_days=5,
            step_days=5,
            sequence_length=args.sequence_length or 50,
            batch_size=args.batch_size or 32
        )
        
        # Run walk-forward validation
        from models.transformer_model import OrderBookTransformer
        from models.lstm_model import OrderBookLSTM
        
        model_class = type(model)
        validator = WalkForwardValidator(model_class, config)
        
        num_splits = min(args.walk_forward_splits, wf_data_module.get_num_splits())
        if args.test_mode:
            num_splits = min(num_splits, 2)
        
        validation_results = validator.validate(wf_data_module, num_splits)
        
        # Save results
        output_path = Path(args.output_dir) / "validation_results.json"
        validator.save_results(output_path)
        
        print(f"Walk-forward validation completed with {len(validation_results)} splits")
        return validator.get_aggregate_metrics()
    
    elif args.validation_type == "cross-validation":
        # Create standard data module for cross-validation
        data_module = create_data_module(feature_vectors, args, config)
        
        model_class = type(model)
        cv_validator = CrossValidator(model_class, config)
        
        n_splits = 3 if args.test_mode else 5
        cv_results = cv_validator.time_series_split_validation(data_module, n_splits)
        
        print(f"Cross-validation completed with {len(cv_results)} splits")
        return {"cv_results": [result.to_dict() for result in cv_results]}
    
    return None


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_to_console=not args.quiet,
        log_to_file=True,
        log_dir=Path(args.output_dir) / "logs"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Market Microstructure Model Training")
    
    try:
        # Setup device
        device = setup_device(args.device)
        logger.info(f"Using device: {device}")
        
        # Load configuration
        config_manager = ConfigManager()
        if args.config:
            config_manager.load_config(args.config)
        else:
            config_manager.load_config()  # Load defaults
        
        config = config_manager.to_dict()
        logger.info("Configuration loaded")
        
        # Load or generate data
        snapshots, data_type = load_or_generate_data(args)
        logger.info(f"Using {data_type} data with {len(snapshots)} snapshots")
        
        # Extract features
        feature_vectors = create_feature_vectors(snapshots, args)
        
        # Create data module
        data_module = create_data_module(feature_vectors, args, config)
        
        # Get input dimension from data
        input_dim = len(data_module.get_feature_names())
        logger.info(f"Input dimension: {input_dim}")
        
        # Create model
        model = create_model(args, config, input_dim)
        model.to(device)
        
        # Train model
        trainer = train_model(model, data_module, args, config, device)
        
        # Run validation if requested
        validation_results = run_validation(model, feature_vectors, args, config)
        if validation_results:
            logger.info(f"Validation results: {validation_results}")
        
        # Save final model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        final_model_path = output_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'args': vars(args),
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
            'feature_names': data_module.get_feature_names()
        }, final_model_path)
        
        logger.info(f"Final model saved to {final_model_path}")
        
        # Cleanup
        trainer.cleanup()
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
        print(f"🤖 Model type: {args.model_type}")
        print(f"💾 Model parameters: {model.count_parameters():,}")
        
        if trainer.best_model_path:
            print(f"🏆 Best model: {trainer.best_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()