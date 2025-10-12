"""
Configuration Management Module

This module provides centralized configuration management for the 
Market Microstructure Analyzer project using YAML configuration files.
"""

from typing import Dict, Any, Optional, Union
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data processing configuration."""
    max_levels: int = 10
    validate_data: bool = True
    sort_levels: bool = True
    sequence_length: int = 50
    prediction_horizon: int = 1
    lookback_window: int = 10
    scaler_type: str = 'standard'  # 'standard', 'minmax', 'robust', 'none'
    
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    
    # Feature engineering
    extract_ofi: bool = True
    extract_spread_features: bool = True
    extract_volume_features: bool = True
    extract_price_features: bool = True
    extract_microstructure_features: bool = True
    extract_technical_features: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = 'transformer'  # 'transformer', 'lstm', 'hybrid'
    
    # Transformer specific
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_length: int = 512
    
    # LSTM specific
    hidden_size: int = 256
    num_lstm_layers: int = 2
    bidirectional: bool = True
    
    # Common
    output_size: int = 1
    activation: str = 'relu'
    batch_norm: bool = True
    layer_norm: bool = True
    
    # Attention mechanism
    use_attention: bool = True
    attention_type: str = 'multihead'  # 'multihead', 'self', 'cross'


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    scheduler: str = 'cosine'  # 'cosine', 'step', 'exponential', 'none'
    
    # Loss function
    loss_function: str = 'mse'  # 'mse', 'mae', 'huber', 'custom'
    
    # Regularization
    weight_decay: float = 0.01
    grad_clip_norm: Optional[float] = 1.0
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-6
    
    # Validation
    val_check_interval: int = 1
    val_metric: str = 'val_loss'  # 'val_loss', 'val_accuracy', 'val_sharpe'
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_every_n_epochs: int = 5
    save_top_k: int = 3


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    # Backtesting engine
    engine_type: str = 'event_driven'
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # As fraction of capital
    
    # Transaction costs
    fixed_fee: float = 0.0  # Fixed fee per trade
    percentage_fee: float = 0.001  # 0.1% fee
    spread_cost: bool = True  # Account for spread crossing
    
    # Market impact
    use_market_impact: bool = True
    impact_model: str = 'sqrt'  # 'sqrt', 'linear', 'none'
    impact_coefficient: float = 0.1
    
    # Slippage
    use_slippage: bool = True
    slippage_bps: float = 1.0  # Basis points
    
    # Risk management
    max_drawdown_pct: float = 15.0
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    
    # Walk-forward validation
    use_walk_forward: bool = True
    train_window_days: int = 30
    test_window_days: int = 5
    step_days: int = 5


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    experiment_name: str = "market_microstructure_experiment"
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    
    # Logging
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_to_wandb: bool = False
    log_to_tensorboard: bool = True
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Outputs
    output_dir: str = "outputs"
    save_predictions: bool = True
    save_attention_weights: bool = True
    save_feature_importance: bool = True


class ConfigManager:
    """
    Central configuration manager for the project.
    
    Handles loading, saving, and validation of configuration files.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize config manager.
        
        Args:
            config_dir: Directory containing config files
        """
        if config_dir is None:
            # Default to configs directory in project root
            self.config_dir = Path(__file__).parent.parent.parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)
        
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize with default configs
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.backtest_config = BacktestConfig()
        self.experiment_config = ExperimentConfig()
        
    def load_config(self, config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to config file. If None, loads from default locations.
            
        Returns:
            Combined configuration dictionary
        """
        if config_file is None:
            # Load from separate config files
            self._load_data_config()
            self._load_model_config()
            self._load_training_config()
            self._load_backtest_config()
            self._load_experiment_config()
        else:
            # Load from single config file
            self._load_from_file(config_file)
        
        return self.to_dict()
    
    def _load_data_config(self):
        """Load data configuration."""
        config_file = self.config_dir / "data_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.data_config = DataConfig(**config_dict)
            logger.info(f"Loaded data config from {config_file}")
        else:
            logger.info("Using default data config")
    
    def _load_model_config(self):
        """Load model configuration."""
        config_file = self.config_dir / "model_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.model_config = ModelConfig(**config_dict)
            logger.info(f"Loaded model config from {config_file}")
        else:
            logger.info("Using default model config")
    
    def _load_training_config(self):
        """Load training configuration."""
        config_file = self.config_dir / "training_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.training_config = TrainingConfig(**config_dict)
            logger.info(f"Loaded training config from {config_file}")
        else:
            logger.info("Using default training config")
    
    def _load_backtest_config(self):
        """Load backtest configuration."""
        config_file = self.config_dir / "backtest_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.backtest_config = BacktestConfig(**config_dict)
            logger.info(f"Loaded backtest config from {config_file}")
        else:
            logger.info("Using default backtest config")
    
    def _load_experiment_config(self):
        """Load experiment configuration."""
        config_file = self.config_dir / "experiment_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.experiment_config = ExperimentConfig(**config_dict)
            logger.info(f"Loaded experiment config from {config_file}")
        else:
            logger.info("Using default experiment config")
    
    def _load_from_file(self, config_file: Union[str, Path]):
        """Load configuration from a single file."""
        config_file = Path(config_file)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        if 'data' in config_dict:
            self.data_config = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            self.model_config = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            self.training_config = TrainingConfig(**config_dict['training'])
        if 'backtest' in config_dict:
            self.backtest_config = BacktestConfig(**config_dict['backtest'])
        if 'experiment' in config_dict:
            self.experiment_config = ExperimentConfig(**config_dict['experiment'])
        
        logger.info(f"Loaded config from {config_file}")
    
    def save_config(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Save current configuration to YAML files.
        
        Args:
            output_dir: Directory to save configs. If None, saves to config_dir.
        """
        if output_dir is None:
            output_dir = self.config_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Save individual config files
        self._save_config_file(output_dir / "data_config.yaml", asdict(self.data_config))
        self._save_config_file(output_dir / "model_config.yaml", asdict(self.model_config))
        self._save_config_file(output_dir / "training_config.yaml", asdict(self.training_config))
        self._save_config_file(output_dir / "backtest_config.yaml", asdict(self.backtest_config))
        self._save_config_file(output_dir / "experiment_config.yaml", asdict(self.experiment_config))
        
        # Save combined config file
        combined_config = {
            'data': asdict(self.data_config),
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'backtest': asdict(self.backtest_config),
            'experiment': asdict(self.experiment_config)
        }
        self._save_config_file(output_dir / "config.yaml", combined_config)
        
        logger.info(f"Saved configuration to {output_dir}")
    
    def _save_config_file(self, file_path: Path, config_dict: Dict[str, Any]):
        """Save a configuration dictionary to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all configurations to dictionary."""
        return {
            'data': asdict(self.data_config),
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'backtest': asdict(self.backtest_config),
            'experiment': asdict(self.experiment_config)
        }
    
    def update_config(self, config_updates: Dict[str, Any]):
        """
        Update configuration values.
        
        Args:
            config_updates: Dictionary with config updates
        """
        if 'data' in config_updates:
            for key, value in config_updates['data'].items():
                if hasattr(self.data_config, key):
                    setattr(self.data_config, key, value)
        
        if 'model' in config_updates:
            for key, value in config_updates['model'].items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
        
        if 'training' in config_updates:
            for key, value in config_updates['training'].items():
                if hasattr(self.training_config, key):
                    setattr(self.training_config, key, value)
        
        if 'backtest' in config_updates:
            for key, value in config_updates['backtest'].items():
                if hasattr(self.backtest_config, key):
                    setattr(self.backtest_config, key, value)
        
        if 'experiment' in config_updates:
            for key, value in config_updates['experiment'].items():
                if hasattr(self.experiment_config, key):
                    setattr(self.experiment_config, key, value)
        
        logger.info("Updated configuration")
    
    def validate_config(self) -> Dict[str, bool]:
        """
        Validate configuration values.
        
        Returns:
            Dictionary with validation results for each config section
        """
        validation_results = {}
        
        # Validate data config
        validation_results['data'] = self._validate_data_config()
        validation_results['model'] = self._validate_model_config()
        validation_results['training'] = self._validate_training_config()
        validation_results['backtest'] = self._validate_backtest_config()
        validation_results['experiment'] = self._validate_experiment_config()
        
        return validation_results
    
    def _validate_data_config(self) -> bool:
        """Validate data configuration."""
        try:
            assert self.data_config.max_levels > 0
            assert 0 < self.data_config.train_ratio < 1
            assert 0 < self.data_config.val_ratio < 1
            assert self.data_config.train_ratio + self.data_config.val_ratio < 1
            assert self.data_config.sequence_length > 0
            assert self.data_config.prediction_horizon > 0
            assert self.data_config.scaler_type in ['standard', 'minmax', 'robust', 'none']
            return True
        except AssertionError:
            logger.error("Data config validation failed")
            return False
    
    def _validate_model_config(self) -> bool:
        """Validate model configuration."""
        try:
            assert self.model_config.model_type in ['transformer', 'lstm', 'hybrid']
            assert self.model_config.d_model > 0
            assert self.model_config.num_heads > 0
            assert self.model_config.num_layers > 0
            assert 0 <= self.model_config.dropout <= 1
            assert self.model_config.hidden_size > 0
            return True
        except AssertionError:
            logger.error("Model config validation failed")
            return False
    
    def _validate_training_config(self) -> bool:
        """Validate training configuration."""
        try:
            assert self.training_config.batch_size > 0
            assert self.training_config.learning_rate > 0
            assert self.training_config.num_epochs > 0
            assert self.training_config.optimizer in ['adam', 'sgd', 'adamw']
            assert self.training_config.patience > 0
            return True
        except AssertionError:
            logger.error("Training config validation failed")
            return False
    
    def _validate_backtest_config(self) -> bool:
        """Validate backtest configuration."""
        try:
            assert self.backtest_config.initial_capital > 0
            assert 0 < self.backtest_config.max_position_size <= 1
            assert self.backtest_config.percentage_fee >= 0
            assert self.backtest_config.impact_model in ['sqrt', 'linear', 'none']
            return True
        except AssertionError:
            logger.error("Backtest config validation failed")
            return False
    
    def _validate_experiment_config(self) -> bool:
        """Validate experiment configuration."""
        try:
            assert self.experiment_config.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
            assert self.experiment_config.random_seed >= 0
            return True
        except AssertionError:
            logger.error("Experiment config validation failed")
            return False


def load_config_from_args(args) -> ConfigManager:
    """
    Load configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ConfigManager instance
    """
    config_manager = ConfigManager()
    
    if hasattr(args, 'config') and args.config:
        config_manager.load_config(args.config)
    else:
        config_manager.load_config()
    
    # Override with command line arguments
    config_updates = {}
    
    # Data config overrides
    if hasattr(args, 'batch_size') and args.batch_size:
        config_updates.setdefault('training', {})['batch_size'] = args.batch_size
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config_updates.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    if hasattr(args, 'epochs') and args.epochs:
        config_updates.setdefault('training', {})['num_epochs'] = args.epochs
    
    if config_updates:
        config_manager.update_config(config_updates)
    
    return config_manager


if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager()
    
    # Load default configs
    config_dict = config_manager.load_config()
    print("Loaded configuration:")
    print(yaml.dump(config_dict, default_flow_style=False, indent=2))
    
    # Validate configuration
    validation_results = config_manager.validate_config()
    print(f"Validation results: {validation_results}")
    
    # Save configuration
    config_manager.save_config()
    print("Saved configuration files")