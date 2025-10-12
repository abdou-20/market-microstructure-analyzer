"""
Training Pipeline for Market Microstructure Models

This module provides a comprehensive training pipeline with support for
various models, loss functions, optimizers, and training strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from dataclasses import dataclass, asdict
import json
import pickle

from utils.logger import get_experiment_logger
from utils.config import ConfigManager
from training.loss_functions import create_loss_function

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 1e-6,
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
    
    def __call__(self, current_value: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value
            model: Model to potentially store weights
            epoch: Current epoch
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict({k: v.to(next(model.parameters()).device) 
                                     for k, v in self.best_weights.items()})
                logger.info(f"Restored best weights from epoch {self.best_epoch}")
            return True
        
        return False


class ModelTrainer:
    """
    Comprehensive model trainer for market microstructure models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None,
                 experiment_name: str = "market_microstructure_training"):
        """
        Initialize model trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Training device
            experiment_name: Name for experiment tracking
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = experiment_name
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_function = self._create_loss_function()
        self.early_stopping = self._create_early_stopping()
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Experiment tracking
        self.experiment_logger = get_experiment_logger(experiment_name)
        self.tensorboard_writer = None
        
        # Setup logging
        self._setup_experiment_tracking()
        
        logger.info(f"Initialized trainer for {model.__class__.__name__} on {self.device}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        training_config = self.config.get('training', {})
        optimizer_type = training_config.get('optimizer', 'adam').lower()
        learning_rate = training_config.get('learning_rate', 0.001)
        weight_decay = training_config.get('weight_decay', 0.01)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = training_config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=learning_rate, 
                           weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        training_config = self.config.get('training', {})
        scheduler_type = training_config.get('scheduler', 'none').lower()
        
        if scheduler_type == 'none':
            return None
        elif scheduler_type == 'cosine':
            T_max = training_config.get('num_epochs', 100)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'step':
            step_size = training_config.get('step_size', 30)
            gamma = training_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'exponential':
            gamma = training_config.get('gamma', 0.95)
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        elif scheduler_type == 'plateau':
            patience = training_config.get('scheduler_patience', 10)
            factor = training_config.get('scheduler_factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=patience, factor=factor)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function from config."""
        training_config = self.config.get('training', {})
        return create_loss_function(training_config)
    
    def _create_early_stopping(self) -> Optional[EarlyStopping]:
        """Create early stopping from config."""
        training_config = self.config.get('training', {})
        
        if not training_config.get('early_stopping', True):
            return None
        
        patience = training_config.get('patience', 10)
        min_delta = training_config.get('min_delta', 1e-6)
        
        return EarlyStopping(patience=patience, min_delta=min_delta)
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking."""
        training_config = self.config.get('training', {})
        experiment_config = self.config.get('experiment', {})
        
        # Setup TensorBoard
        if experiment_config.get('log_to_tensorboard', True):
            log_dir = Path(experiment_config.get('output_dir', 'outputs')) / 'tensorboard'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir)
        
        # Log hyperparameters
        hyperparams = {
            'model_type': self.model.__class__.__name__,
            'optimizer': training_config.get('optimizer', 'adam'),
            'learning_rate': training_config.get('learning_rate', 0.001),
            'batch_size': training_config.get('batch_size', 32),
            'num_epochs': training_config.get('num_epochs', 100),
            'loss_function': training_config.get('loss_function', 'mse'),
        }
        
        if hasattr(self.model, 'get_model_info'):
            hyperparams.update(self.model.get_model_info())
        
        self.experiment_logger.log_hyperparameters(hyperparams)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct_directions = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 2:
                features, targets = batch
                metadata = None
            else:
                features, targets, metadata = batch
            
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(features)
            predictions = output['predictions']
            
            # Compute loss
            loss = self.loss_function(predictions.squeeze(), targets.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            training_config = self.config.get('training', {})
            grad_clip_norm = training_config.get('grad_clip_norm')
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Compute directional accuracy
            pred_directions = torch.sign(predictions.squeeze())
            target_directions = torch.sign(targets.squeeze())
            correct_directions += (pred_directions == target_directions).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct_directions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_directions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    features, targets = batch
                else:
                    features, targets, _ = batch
                
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                output = self.model(features)
                predictions = output['predictions']
                
                # Compute loss
                loss = self.loss_function(predictions.squeeze(), targets.squeeze())
                
                # Update metrics
                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Compute directional accuracy
                pred_directions = torch.sign(predictions.squeeze())
                target_directions = torch.sign(targets.squeeze())
                correct_directions += (pred_directions == target_directions).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct_directions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None) -> List[TrainingMetrics]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            
        Returns:
            List of training metrics per epoch
        """
        if num_epochs is None:
            num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_accuracy = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_accuracy = None, None
            if val_loader is not None:
                val_loss, val_accuracy = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Create metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            
            self.training_history.append(metrics)
            
            # Log metrics
            self._log_metrics(metrics)
            
            # Model checkpointing
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)
            
            # Regular checkpointing
            training_config = self.config.get('training', {})
            if training_config.get('save_checkpoints', True):
                checkpoint_interval = training_config.get('checkpoint_every_n_epochs', 5)
                if (epoch + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(epoch)
            
            # Early stopping
            if self.early_stopping is not None:
                stop_metric = val_loss if val_loss is not None else train_loss
                if self.early_stopping(stop_metric, self.model, epoch):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Progress logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self._log_progress(metrics)
        
        logger.info("Training completed")
        return self.training_history
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics to various tracking systems."""
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('Loss/Train', metrics.train_loss, metrics.epoch)
            if metrics.val_loss is not None:
                self.tensorboard_writer.add_scalar('Loss/Validation', metrics.val_loss, metrics.epoch)
            if metrics.train_accuracy is not None:
                self.tensorboard_writer.add_scalar('Accuracy/Train', metrics.train_accuracy, metrics.epoch)
            if metrics.val_accuracy is not None:
                self.tensorboard_writer.add_scalar('Accuracy/Validation', metrics.val_accuracy, metrics.epoch)
            self.tensorboard_writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.epoch)
        
        # Experiment logger
        self.experiment_logger.log_metric('train_loss', metrics.train_loss, metrics.epoch)
        if metrics.val_loss is not None:
            self.experiment_logger.log_metric('val_loss', metrics.val_loss, metrics.epoch)
        if metrics.train_accuracy is not None:
            self.experiment_logger.log_metric('train_accuracy', metrics.train_accuracy, metrics.epoch)
        if metrics.val_accuracy is not None:
            self.experiment_logger.log_metric('val_accuracy', metrics.val_accuracy, metrics.epoch)
    
    def _log_progress(self, metrics: TrainingMetrics):
        """Log training progress."""
        val_info = ""
        if metrics.val_loss is not None:
            val_info = f" - Val Loss: {metrics.val_loss:.6f}"
        if metrics.val_accuracy is not None:
            val_info += f" - Val Acc: {metrics.val_accuracy:.4f}"
        
        logger.info(
            f"Epoch {metrics.epoch + 1} - "
            f"Train Loss: {metrics.train_loss:.6f} - "
            f"Train Acc: {metrics.train_accuracy:.4f}"
            f"{val_info} - "
            f"LR: {metrics.learning_rate:.6f} - "
            f"Time: {metrics.epoch_time:.2f}s"
        )
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        experiment_config = self.config.get('experiment', {})
        output_dir = Path(experiment_config.get('output_dir', 'outputs'))
        checkpoints_dir = output_dir / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoints_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoints_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
        return checkpoint
    
    def save_training_history(self, file_path: Optional[Union[str, Path]] = None):
        """Save training history to file."""
        if file_path is None:
            experiment_config = self.config.get('experiment', {})
            output_dir = Path(experiment_config.get('output_dir', 'outputs'))
            file_path = output_dir / 'training_history.json'
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        history_data = [metrics.to_dict() for metrics in self.training_history]
        
        with open(file_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Saved training history to {file_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        
        # Save final metrics
        self.experiment_logger.save_metrics()
        
        # Save training history
        self.save_training_history()


if __name__ == "__main__":
    # Test the trainer
    from ..models.transformer_model import create_transformer_model
    from ..data_processing.data_loader import OrderBookDataModule
    from ..data_processing.order_book_parser import create_synthetic_order_book_data
    from ..data_processing.feature_engineering import FeatureEngineering
    
    print("Testing model trainer...")
    
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
    
    # Create model
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
            'num_epochs': 5,
            'loss_function': 'mse',
            'early_stopping': True,
            'patience': 3
        },
        'experiment': {
            'output_dir': 'test_outputs',
            'log_to_tensorboard': False
        }
    }
    
    model = create_transformer_model(config)
    
    # Create trainer
    trainer = ModelTrainer(model, config, experiment_name="test_training")
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Train model
    training_history = trainer.train(train_loader, val_loader, num_epochs=3)
    
    print(f"Training completed with {len(training_history)} epochs")
    print(f"Final train loss: {training_history[-1].train_loss:.6f}")
    if training_history[-1].val_loss:
        print(f"Final val loss: {training_history[-1].val_loss:.6f}")
    
    # Cleanup
    trainer.cleanup()
    
    print("✅ Trainer test passed!")