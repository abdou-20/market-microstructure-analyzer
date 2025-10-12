"""
Logging Utilities Module

This module provides centralized logging configuration for the 
Market Microstructure Analyzer project.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import os


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the record
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


class ExperimentLogger:
    """
    Enhanced logger for ML experiments with metrics tracking.
    """
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        """
        Initialize experiment logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_dir = log_dir
        self.metrics: Dict[str, Any] = {}
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {'value': value, 'timestamp': datetime.now().isoformat()}
        if step is not None:
            metric_entry['step'] = step
        
        self.metrics[name].append(metric_entry)
        
        # Also log to regular logger
        if step is not None:
            self.logger.info(f"Metric {name}: {value} (step {step})")
        else:
            self.logger.info(f"Metric {name}: {value}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.logger.info(f"Hyperparameters: {json.dumps(params, indent=2)}")
        
        # Save to file if log_dir is specified
        if self.log_dir:
            params_file = self.log_dir / "hyperparameters.json"
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=2)
    
    def save_metrics(self):
        """Save metrics to file."""
        if self.log_dir and self.metrics:
            metrics_file = self.log_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            self.logger.info(f"Saved metrics to {metrics_file}")
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Get summary statistics for a metric."""
        if metric_name not in self.metrics:
            return None
        
        values = [entry['value'] for entry in self.metrics[metric_name]]
        
        if not values:
            return None
        
        return {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'last': values[-1],
            'count': len(values)
        }


def setup_logging(
    log_level: str = 'INFO',
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_file: Optional[Path] = None,
    log_format: str = 'standard',
    log_dir: Optional[Path] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_file: Specific log file path
        log_format: Format type ('standard', 'json', 'colored')
        log_dir: Directory for log files
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Create log directory if specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    if log_format == 'json':
        formatter = JSONFormatter()
    elif log_format == 'colored':
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:  # standard
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_file is None:
            if log_dir:
                log_file = log_dir / f"market_microstructure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            else:
                log_file = Path(f"market_microstructure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Add info about logging setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete - Level: {log_level}, Console: {log_to_console}, File: {log_to_file}")
    
    return root_logger


def get_experiment_logger(experiment_name: str, log_dir: Optional[Path] = None) -> ExperimentLogger:
    """
    Get an experiment logger instance.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for experiment logs
        
    Returns:
        ExperimentLogger instance
    """
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    return ExperimentLogger(experiment_name, log_dir)


def log_system_info():
    """Log system information for debugging."""
    import platform
    import psutil
    import torch
    
    logger = logging.getLogger(__name__)
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("CUDA available: False")
    
    logger.info("=== End System Information ===")


def log_config(config_dict: Dict[str, Any]):
    """Log configuration dictionary."""
    logger = logging.getLogger(__name__)
    logger.info("=== Configuration ===")
    logger.info(json.dumps(config_dict, indent=2, default=str))
    logger.info("=== End Configuration ===")


class LoggerContextManager:
    """Context manager for temporary logger configuration."""
    
    def __init__(self, logger_name: str, level: str = 'INFO', extra_fields: Optional[Dict[str, Any]] = None):
        """
        Initialize context manager.
        
        Args:
            logger_name: Name of the logger
            level: Logging level for this context
            extra_fields: Additional fields to include in log records
        """
        self.logger_name = logger_name
        self.level = level
        self.extra_fields = extra_fields or {}
        self.logger = logging.getLogger(logger_name)
        self.original_level = None
    
    def __enter__(self):
        """Enter the context."""
        self.original_level = self.logger.level
        self.logger.setLevel(getattr(logging, self.level.upper()))
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)


def with_logging(func):
    """Decorator to add logging to functions."""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    
    return wrapper


if __name__ == "__main__":
    # Example usage
    
    # Setup basic logging
    setup_logging(
        log_level='INFO',
        log_to_console=True,
        log_to_file=True,
        log_format='colored',
        log_dir=Path("logs")
    )
    
    # Test basic logging
    logger = logging.getLogger(__name__)
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test experiment logger
    exp_logger = get_experiment_logger("test_experiment", Path("logs/experiments"))
    exp_logger.log_metric("accuracy", 0.85, step=1)
    exp_logger.log_metric("loss", 0.15, step=1)
    exp_logger.log_hyperparameters({"learning_rate": 0.001, "batch_size": 32})
    exp_logger.save_metrics()
    
    # Log system info
    log_system_info()
    
    # Test context manager
    with LoggerContextManager("test_context", "DEBUG") as ctx_logger:
        ctx_logger.debug("This debug message will be shown")
    
    print("Logging test completed")